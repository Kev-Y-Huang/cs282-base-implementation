# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team and Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" The distiller to distil the student.
    Adapted in part from Facebook, Inc XLM model (https://github.com/facebookresearch/XLM)
"""
import math
import os
import time

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import BatchSampler, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from grouped_batch_sampler import GroupedBatchSampler, create_lengths_groups
from lm_seqs_dataset import LmSeqsDataset
from transformers import get_linear_schedule_with_warmup
from utils import logger

import json
import random

import numpy as np
import torch

# from distiller import Distiller
from lm_seqs_dataset import LmSeqsDataset

from utils import logger

from collections import defaultdict

def deserialize_variable_name(variable_name):
    deserialized_variables = [] 
    params = variable_name.split("$")

    # get layer variable 
    layer = int(params[1][-1])
    
    # get head range 
    head_vars = params[2].split(":")
    head_l = int(head_vars[1].strip("["))
    head_r = int(head_vars[2].strip("]"))

    # get head nodes 
    nodes = params[3].split(":")
    nodes_l = int(nodes[0].strip("["))
    nodes_r = int(nodes[1].strip("]"))

    # iterate over all heads
    for i in range(head_l, head_r):
        var = (layer, i, slice(nodes_l, nodes_r))
        deserialized_variables.append(var)

    return deserialized_variables

class CausalDistiller:
    def __init__(
        self, params: dict, dataset: LmSeqsDataset, 
        token_probs: torch.tensor, student: nn.Module, teacher: nn.Module
    ):  
        logger.info("Initializing Distiller")
        self.params = params
        self.dump_path = params.dump_path
        self.multi_gpu = params.multi_gpu
        self.fp16 = params.fp16

        self.student = student
        self.teacher = teacher
        
        self.deserialized_variable_mappings = defaultdict(list)
        def load_variable_names(m):
            deserialized_variables = [] 
            for variable in m:
                deserialized_variables.append(deserialize_variable_name(variable))
            return deserialized_variables
        
        # node mappings
        # neuron mapping logic is taken from the paper's implementation, modified to fit out usecase
        with open(params.neuron_mapping) as json_file:
            logger.info(f"Loading neuron mapping {params.neuron_mapping}")
            neuron_mapping_json = json.load(json_file)
            names = neuron_mapping_json["interchange_variable_mappings"][0]

            t_deserialized_variables = load_variable_names(names["teacher_variable_names"])
            s_deserialized_variables = load_variable_names(names["student_variable_names"])
            self.deserialized_variable_mappings["teacher"].extend(t_deserialized_variables)
            self.deserialized_variable_mappings["student"].extend(s_deserialized_variables)        

        print(self.deserialized_variable_mappings)

        self.student_config = student.config
        self.vocab_size = student.config.vocab_size

        # overwrite slightly on this.
        if params.local_rank == -1:
            sampler = RandomSampler(dataset)
        else:
            sampler = DistributedSampler(dataset)
            
        if params.group_by_size:
            groups = create_lengths_groups(lengths=dataset.x1_lengths, k=params.max_model_input_size)
            sampler = GroupedBatchSampler(sampler=sampler, group_ids=groups, batch_size=params.batch_size)
        else:
            sampler = BatchSampler(sampler=sampler, batch_size=params.batch_size, drop_last=False)
  
        self.dataloader = DataLoader(
            dataset=dataset, batch_sampler=sampler, collate_fn=dataset.batch_sequences,
        )

        self.temperature = params.temperature
        assert self.temperature > 0.0

        self.alpha_ce = params.alpha_ce
        self.alpha_mlm = params.alpha_mlm
        self.alpha_clm = params.alpha_clm
        self.alpha_mse = params.alpha_mse
        self.alpha_cos = params.alpha_cos
        self.alpha_causal_ce = params.alpha_causal_ce

        self.mlm = params.mlm
        if self.mlm:
            logger.info("Using MLM loss for LM step.")
            self.mlm_mask_prop = params.mlm_mask_prop
            assert 0.0 <= self.mlm_mask_prop <= 1.0
            assert params.word_mask + params.word_keep + params.word_rand == 1.0
            self.pred_probs = torch.FloatTensor([params.word_mask, params.word_keep, params.word_rand])
            self.pred_probs = self.pred_probs.to(torch.device("cuda"), non_blocking=True) if params.n_gpu > 0 else self.pred_probs
            self.token_probs = token_probs.to(torch.device("cuda"), non_blocking=True) if params.n_gpu > 0 else token_probs
            if self.fp16:
                self.pred_probs = self.pred_probs.half()
                self.token_probs = self.token_probs.half()
        else:
            logger.info("Using CLM loss for LM step.")
        
        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.n_sequences_epoch = 0
        self.total_loss_epoch = 0
        self.last_loss = 0
        self.last_loss_ce = 0
        self.last_loss_mlm = 0
        self.last_loss_clm = 0
        if self.alpha_mse > 0.0:
            self.last_loss_mse = 0
        if self.alpha_cos > 0.0:
            self.last_loss_cos = 0

        self.last_loss_causal_ce = 0
        self.last_log = 0

        self.ce_loss_fct = nn.KLDivLoss(reduction="batchmean")
        self.lm_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        if self.alpha_mse > 0.0:
            self.mse_loss_fct = nn.MSELoss(reduction="sum")
        if self.alpha_cos > 0.0:
            self.cosine_loss_fct = nn.CosineEmbeddingLoss(reduction="mean")

        logger.info("--- Initializing model optimizer")
        assert params.gradient_accumulation_steps >= 1
        self.num_steps_epoch = len(self.dataloader)
        num_train_optimization_steps = (
            int(self.num_steps_epoch / params.gradient_accumulation_steps * params.n_epoch) + 1
        )

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in student.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": params.weight_decay,
            },
            {
                "params": [
                    p for n, p in student.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        logger.info(
            "------ Number of trainable parameters (student): %i"
            % sum([p.numel() for p in self.student.parameters() if p.requires_grad])
        )
        logger.info("------ Number of parameters (student): %i" % sum([p.numel() for p in self.student.parameters()]))
        self.optimizer = AdamW(
            optimizer_grouped_parameters, lr=params.learning_rate, eps=params.adam_epsilon, betas=(0.9, 0.98)
        )

        warmup_steps = math.ceil(num_train_optimization_steps * params.warmup_prop)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps
        )

        if self.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            logger.info(f"Using fp16 training: {self.params.fp16_opt_level} level")
            self.student, self.optimizer = amp.initialize(
                self.student, self.optimizer, opt_level=self.params.fp16_opt_level
            )
            self.teacher = self.teacher.half()

        if self.multi_gpu:
            if self.fp16:
                from apex.parallel import DistributedDataParallel

                logger.info("Using apex.parallel.DistributedDataParallel for distributed training.")
                self.student = DistributedDataParallel(self.student)
            else:
       
                if params.local_rank == -1:
                    logger.info("Using nn.DataParallel for the teacher model.")
                    self.teacher = torch.nn.DataParallel(self.teacher)
                    self.teacher.to(torch.device("cuda")) 

                    logger.info("Using nn.DataParallel for the student model.")
                    self.student = torch.nn.DataParallel(self.student)
                    self.student.to(torch.device("cuda"))
                else:
                
                    from torch.nn.parallel import DistributedDataParallel

                    logger.info("Using nn.parallel.DistributedDataParallel for distributed training.")
                    self.student = DistributedDataParallel(
                        self.student,
                        device_ids=[params.local_rank],
                        output_device=params.local_rank,
                        find_unused_parameters=True,
                    )
        else:
            # for signle gpu usage
            self.teacher.to(torch.device("cuda"))
            self.student.to(torch.device("cuda"))
        self.is_master = params.is_master
        
        
    def prepare_batch_mlm(self, batch):
        """
        Prepare the batch: from the token_ids and the lengths, compute the attention mask and the masked label for MLM.

        Input:
        ------
            batch: `Tuple`
                token_ids: `torch.tensor(bs, seq_length)` - The token ids for each of the sequence. It is padded.
                lengths: `torch.tensor(bs)` - The lengths of each of the sequences in the batch.

        Output:
        -------
            token_ids: `torch.tensor(bs, seq_length)` - The token ids after the modifications for MLM.
            attn_mask: `torch.tensor(bs, seq_length)` - The attention mask for the self-attention.
            mlm_labels: `torch.tensor(bs, seq_length)` - The masked language modeling labels. There is a -100 where there is nothing to predict.
        """
        token_ids, lengths = batch
        token_ids, lengths = self.round_batch(x=token_ids, lengths=lengths)
        assert token_ids.size(0) == lengths.size(0)

        attn_mask = torch.arange(token_ids.size(1), dtype=torch.long, device=lengths.device) < lengths[:, None]

        bs, max_seq_len = token_ids.size()
        mlm_labels = token_ids.new(token_ids.size()).copy_(token_ids)

        x_prob = self.token_probs[token_ids.flatten()]
        n_tgt = math.ceil(self.mlm_mask_prop * lengths.sum().item())
        tgt_ids = torch.multinomial(x_prob / x_prob.sum(), n_tgt, replacement=False)
        pred_mask = torch.zeros(
            bs * max_seq_len, dtype=torch.bool, device=token_ids.device
        )  # previously `dtype=torch.uint8`, cf pytorch 1.2.0 compatibility
        pred_mask[tgt_ids] = 1
        pred_mask = pred_mask.view(bs, max_seq_len)

        pred_mask[token_ids == self.params.special_tok_ids["pad_token"]] = 0

        # mask a number of words == 0 [8] (faster with fp16)
        if self.fp16:
            n1 = pred_mask.sum().item()
            if n1 > 8:
                pred_mask = pred_mask.view(-1)
                n2 = max(n1 % 8, 8 * (n1 // 8))
                if n2 != n1:
                    pred_mask[torch.nonzero(pred_mask).view(-1)[: n1 - n2]] = 0
                pred_mask = pred_mask.view(bs, max_seq_len)
                assert pred_mask.sum().item() % 8 == 0, pred_mask.sum().item()

        _token_ids_real = token_ids[pred_mask]
        _token_ids_rand = _token_ids_real.clone().random_(self.vocab_size)
        _token_ids_mask = _token_ids_real.clone().fill_(self.params.special_tok_ids["mask_token"])
        probs = torch.multinomial(self.pred_probs, len(_token_ids_real), replacement=True).to(_token_ids_real.device, non_blocking=True)
        _token_ids = (
            _token_ids_mask * (probs == 0).long()
            + _token_ids_real * (probs == 1).long()
            + _token_ids_rand * (probs == 2).long()
        )
        token_ids = token_ids.masked_scatter(pred_mask, _token_ids)

        mlm_labels[~pred_mask] = -100  # previously `mlm_labels[1-pred_mask] = -1`, cf pytorch 1.2.0 compatibility

        # sanity checks
        assert 0 <= token_ids.min() <= token_ids.max() < self.vocab_size

        return token_ids, attn_mask, mlm_labels, pred_mask

    def prepare_batch_clm(self, batch):
        """
        Prepare the batch: from the token_ids and the lengths, compute the attention mask and the labels for CLM.

        Input:
        ------
            batch: `Tuple`
                token_ids: `torch.tensor(bs, seq_length)` - The token ids for each of the sequence. It is padded.
                lengths: `torch.tensor(bs)` - The lengths of each of the sequences in the batch.

        Output:
        -------
            token_ids: `torch.tensor(bs, seq_length)` - The token ids after the modifications for MLM.
            attn_mask: `torch.tensor(bs, seq_length)` - The attention mask for the self-attention.
            clm_labels: `torch.tensor(bs, seq_length)` - The causal language modeling labels. There is a -100 where there is nothing to predict.
        """
        token_ids, lengths = batch
        token_ids, lengths = self.round_batch(x=token_ids, lengths=lengths)
        assert token_ids.size(0) == lengths.size(0)

        attn_mask = torch.arange(token_ids.size(1), dtype=torch.long, device=lengths.device) < lengths[:, None]
        clm_labels = token_ids.new(token_ids.size()).copy_(token_ids)
        clm_labels[~attn_mask] = -100  # previously `clm_labels[1-attn_mask] = -1`, cf pytorch 1.2.0 compatibility

        # sanity checks
        assert 0 <= token_ids.min() <= token_ids.max() < self.vocab_size

        return token_ids, attn_mask, clm_labels

    def round_batch(self, x: torch.tensor, lengths: torch.tensor):
        """
        For float16 only.
        Sub-sample sentences in a batch, and add padding, so that each dimension is a multiple of 8.

        Input:
        ------
            x: `torch.tensor(bs, seq_length)` - The token ids.
            lengths: `torch.tensor(bs, seq_length)` - The lengths of each of the sequence in the batch.

        Output:
        -------
            x:  `torch.tensor(new_bs, new_seq_length)` - The updated token ids.
            lengths: `torch.tensor(new_bs, new_seq_length)` - The updated lengths.
        """
        if not self.fp16 or len(lengths) < 8:
            return x, lengths

        # number of sentences == 0 [8]
        bs1 = len(lengths)
        bs2 = 8 * (bs1 // 8)
        assert bs2 > 0 and bs2 % 8 == 0
        if bs1 != bs2:
            idx = torch.randperm(bs1)[:bs2]
            lengths = lengths[idx]
            slen = lengths.max().item()
            x = x[idx, :slen]
        else:
            idx = None

        # sequence length == 0 [8]
        ml1 = x.size(1)
        if ml1 % 8 != 0:
            pad = 8 - (ml1 % 8)
            ml2 = ml1 + pad
            if self.mlm:
                pad_id = self.params.special_tok_ids["pad_token"]
            else:
                pad_id = self.params.special_tok_ids["unk_token"]
            padding_tensor = torch.zeros(bs2, pad, dtype=torch.long, device=x.device).fill_(pad_id)
            x = torch.cat([x, padding_tensor], 1)
            assert x.size() == (bs2, ml2)

        assert x.size(0) % 8 == 0
        assert x.size(1) % 8 == 0
        return x, lengths

    # select 30% randomly of neurons to interchange 
    def sample_interchange(
        self,
        lengths, dual_lengths,
        pred_mask, dual_pred_mask,
    ):        
        # create mask for interchanged neurons
        interchange_mask = torch.zeros_like(pred_mask, dtype=torch.bool)
        dual_interchange_mask = torch.zeros_like(dual_pred_mask, dtype=torch.bool)

        # sequential interchanging, sample 30% of neurons
        for i in range(0, self.params.batch_size):
            num_interchange_1 = int(lengths[i] * 0.3)
            num_interchange_2 = int(dual_lengths[i] * 0.3) 

            num_interchange = min(num_interchange_1, num_interchange_2)

            start_1 = random.randint(0, lengths[i].tolist() - num_interchange)
            end_1 = start_1 + num_interchange

            start_2 = random.randint(0, lengths[i].tolist() - num_interchange)
            end_2 = start_2 + num_interchange

            for j in range(start_1, end_1):
                interchange_mask[i][j] = 1 
            
            for j in range(start_2, end_2):
                dual_interchange_mask[i][j] = 1

        return interchange_mask, dual_interchange_mask

    
    def train(self):
        """
        The real training loop.
        """
        if self.is_master:
            logger.info("Starting training")
        self.last_log = time.time()
        self.student.train()
        self.teacher.eval()

        for _ in range(self.params.n_epoch):
            if self.is_master:
                logger.info(f"--- Starting epoch {self.epoch}/{self.params.n_epoch-1}")

            iter_bar = tqdm(self.dataloader, desc="-Iter", disable=self.params.local_rank not in [-1, 0])
            for batch in iter_bar:
                # dual_token ids, dual_lengths: x2, y2
                if self.params.n_gpu > 0:
                    batch = tuple(t.to(torch.device("cuda"), non_blocking=True) for t in batch)

                token_ids, lengths, dual_token_ids, dual_lengths = batch

                if self.mlm:
                    token_ids, attn_mask, lm_labels, pred_mask = self.prepare_batch_mlm(
                        batch=(token_ids, lengths)
                    )

                    # prepare mlm mask for x2, y2
                    dual_token_ids, dual_attn_mask, dual_lm_labels, dual_pred_mask = self.prepare_batch_mlm(
                        batch=(dual_token_ids, dual_lengths)
                    )
                else:
                    token_ids, attn_mask, lm_labels = self.prepare_batch_clm(batch=(token_ids, lengths))
                    
                    # prepare clm mask for x2, y2
                    dual_token_ids, dual_attn_mask, dual_lm_labels = self.prepare_batch_clm(
                        batch=(dual_token_ids, dual_lengths)
                    )
                    
                interchange_mask, dual_interchange_mask = self.sample_interchange(
                    lengths, dual_lengths,
                    pred_mask, dual_pred_mask,
                )

                self.step(
                    input_ids=token_ids, 
                    attention_mask=attn_mask, 
                    lm_labels=lm_labels,
                    dual_input_ids=dual_token_ids, 
                    dual_attention_mask=dual_attn_mask, 
                    interchange_mask=interchange_mask, 
                    dual_interchange_mask=dual_interchange_mask,
                )
                iter_bar.update()
                iter_bar.set_postfix(
                    {
                        "Last_loss": f"{self.last_loss:.2f}", 
                         "Avg_cum_loss": f"{self.total_loss_epoch/self.n_iter:.2f}", 
                         "Last_cf_loss": f"{self.last_loss_causal_ce:.2f}", 
                    }
                )
            iter_bar.close()

            if self.is_master:
                logger.info(f"--- Ending epoch {self.epoch}/{self.params.n_epoch-1}")
            self.end_epoch()

        if self.is_master:
            logger.info("Save very last checkpoint as `pytorch_model.bin`.")
            self.save_checkpoint(checkpoint_name="pytorch_model.bin")
            logger.info("Training is finished")

    def get_activations(
        self, model, input_ids, attention_mask, 
        variable_names
    ):
        # we don't want embedding activations
        if variable_names == "embeddings":
            return None

        # get outputs from model
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        total_activations = []

        # get head_dimension: 
        # should = 64 for default bert
        head_dim = 64

        for v in variable_names:
            # hidden sate format: n tuple with embeddings + layer, each layer is another
            # tuple of format ((batch_size, sequence_length, hidden_size))
            # in this case, we want to extract the activation weights at each layer
            layer_index, head_index, activation_locations = v
            
            hidden_states = outputs["hidden_states"]
            layer = hidden_states[layer_index]
            head_hidden_states = layer[:,:, (head_index * head_dim):((head_index+1) * head_dim)]

            # 12 attention heads, each attention head is 64 nodes wide
            total_activations.append(head_hidden_states[:,:,activation_locations])

        return total_activations

    def get_interchanged_variables_mapping(self, variable_names):
        interchanged_variables_mapping = defaultdict(list)
        for i, variable in enumerate(variable_names):
            layer_index, head_index, activation_locations = variable
            interchanged_variables_mapping[layer_index].append((i, head_index, activation_locations))

        return interchanged_variables_mapping

    # code taken from huggingface Distillation module
    def calculate_loss(self, 
        student_outputs, 
        t_hidden_states, 
        t_logits, 
        lm_labels,
        attention_mask):
        if t_logits is not None:
            # regular loss
            s_logits, s_hidden_states = student_outputs["logits"], student_outputs["hidden_states"]
            assert s_logits.size() == t_logits.size()
            # https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # https://github.com/peterliht/knowledge-distillation-pytorch/issues/2
            if self.params.restrict_ce_to_mask:
                mask = (lm_labels > -1).unsqueeze(-1).expand_as(s_logits)  # (bs, seq_length, voc_size)
            else:
                mask = attention_mask.unsqueeze(-1).expand_as(s_logits)  # (bs, seq_length, voc_size)
            s_logits_slct = torch.masked_select(s_logits, mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
            s_logits_slct = s_logits_slct.view(-1, s_logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
            t_logits_slct = torch.masked_select(t_logits, mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
            t_logits_slct = t_logits_slct.view(-1, s_logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
            assert t_logits_slct.size() == s_logits_slct.size()

            loss_ce = (
                self.ce_loss_fct(
                    nn.functional.log_softmax(s_logits_slct / self.temperature, dim=-1),
                    nn.functional.softmax(t_logits_slct / self.temperature, dim=-1),
                )
                * (self.temperature) ** 2
            )
            student_outputs["loss_ce"] = loss_ce

            # other distillation loss.
            if self.alpha_mlm > 0.0:
                loss_mlm = self.lm_loss_fct(s_logits.view(-1, s_logits.size(-1)), lm_labels.view(-1))
                student_outputs["loss_mlm"] = loss_mlm
            if self.alpha_clm > 0.0:
                shift_logits = s_logits[..., :-1, :].contiguous()
                shift_labels = lm_labels[..., 1:].contiguous()
                loss_clm = self.lm_loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                student_outputs["loss_mlm"] = loss_clm
            if self.alpha_mse > 0.0:
                loss_mse = self.mse_loss_fct(s_logits_slct, t_logits_slct) / s_logits_slct.size(
                    0
                )  # Reproducing batchmean reduction
                student_outputs["loss_mse"] = loss_mse
            if self.alpha_cos > 0.0:
                s_hidden_states = s_hidden_states[-1]  # (bs, seq_length, dim)
                t_hidden_states = t_hidden_states[-1]  # (bs, seq_length, dim)
                mask = attention_mask.unsqueeze(-1).expand_as(s_hidden_states)  # (bs, seq_length, dim)
                assert s_hidden_states.size() == t_hidden_states.size()
                dim = s_hidden_states.size(-1)

                s_hidden_states_slct = torch.masked_select(s_hidden_states, mask)  # (bs * seq_length * dim)
                s_hidden_states_slct = s_hidden_states_slct.view(-1, dim)  # (bs * seq_length, dim)
                t_hidden_states_slct = torch.masked_select(t_hidden_states, mask)  # (bs * seq_length * dim)
                t_hidden_states_slct = t_hidden_states_slct.view(-1, dim)  # (bs * seq_length, dim)

                target = s_hidden_states_slct.new(s_hidden_states_slct.size(0)).fill_(1)  # (bs * seq_length,)
                loss_cos = self.cosine_loss_fct(s_hidden_states_slct, t_hidden_states_slct, target)
                student_outputs["loss_cos"] = loss_cos
            
            return student_outputs
    
    # taken from Huggingface distillation module - calculating L_CE
    def calculate_causal_loss(self, 
        student_outputs,
        lm_labels,
        attention_mask,
        causal_t_logits,
        ):

        causal_s_logits = student_outputs["logits"]
        assert causal_s_logits.size() == causal_t_logits.size()
        # https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
        # https://github.com/peterliht/knowledge-distillation-pytorch/issues/2
        if self.params.restrict_ce_to_mask:
            causal_mask = (lm_labels > -1).unsqueeze(-1).expand_as(causal_s_logits)  # (bs, seq_length, voc_size)
        else:
            causal_mask = attention_mask.unsqueeze(-1).expand_as(causal_s_logits)  # (bs, seq_length, voc_size)
        causal_s_logits_slct = torch.masked_select(causal_s_logits, causal_mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        causal_s_logits_slct = causal_s_logits_slct.view(-1, causal_s_logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
        causal_t_logits_slct = torch.masked_select(causal_t_logits, causal_mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        causal_t_logits_slct = causal_t_logits_slct.view(-1, causal_s_logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
        assert causal_t_logits_slct.size() == causal_s_logits_slct.size()
        
        causal_loss_ce = (
            self.ce_loss_fct(
                nn.functional.log_softmax(causal_s_logits_slct / self.temperature, dim=-1),
                nn.functional.softmax(causal_t_logits_slct / self.temperature, dim=-1),
            )
            * (self.temperature) ** 2
        )
        student_outputs["causal_loss_ce"] = causal_loss_ce
        
        return student_outputs

    def step(
        self, input_ids: torch.tensor, 
        attention_mask: torch.tensor, 
        lm_labels: torch.tensor,
        dual_input_ids: torch.tensor, 
        dual_attention_mask: torch.tensor, 
        interchange_mask: torch.tensor,
        dual_interchange_mask: torch.tensor,
        skip_update_iter=False,
    ):
        """
        One optimization step: forward of student AND teacher, backward on the loss (for gradient accumulation),
        and possibly a parameter update (depending on the gradient accumulation).

        Input:
        ------
        input_ids/dual_input_ids: `torch.tensor(bs, seq_length)` - The token ids.
        attention_mask/dual_attention_mask: `torch.tensor(bs, seq_length)` - The attention mask for self attention.
        lm_labels/dual_lm_labels: `torch.tensor(bs, seq_length)` - The language modeling labels (mlm labels for MLM and clm labels for CLM).
        """

        # select random variable names for interchange        
        teacher_variable_names = random.choice(self.deserialized_variable_mappings["teacher"])
        student_variable_names = random.choice(self.deserialized_variable_mappings["student"])

        # get variables to interchange
        teacher_interchanged_variables_mapping = self.get_interchanged_variables_mapping(teacher_variable_names)
        student_interchanged_variables_mapping = self.get_interchanged_variables_mapping(student_variable_names)        

        # counterfactual becomes x1 
        counterfactual_input_ids = input_ids
        
        # input x2
        dual_inputs = {
            'input_ids': dual_input_ids,
            'attention_mask': dual_attention_mask,
        }
        
        if self.mlm:
            with torch.no_grad():
                """
                    Within a training step: 
                        given a sample x1, we pass that through the teacher 
                        After a single pass we gather the activations for a second input x2 (no grad here) 

                        We then replace the values of the neurons
                        from the teacher model with the values after inputting x1 with the activations
                         after inputting x2

                        we do the same with the student model 

                        using these new interchanged models, we calculate the loss using the output 
                        of x2 between both the teacher and the student model

                        in the end, an intervention operation is the output state we get from a model
                        for an input x2 but with the neurons N set to the values
                        obtained when processing input x1

                    Here the paper states that: 
                    GETVAL(M, x, N) = gets the set of values that N takes when 
                        processing x --> getting the activations of the neurons N 
                        - in the case of an output, then GETVALS is the output of model M 
                    
                    SETVAL(M, N, v) = returns a new neural model where neurons N are 
                            set to constant value v in the model M

                        T_a = SETVAL(T, N_t, GETVAL(T, x_1, N_t))
                            where N_t are the neurons of T,
                            x_1 is an input, 
                        
                        The operation here basically is saying:
                            - get the activations of the taecher model at an input x_1,
                              and set the neurons of the taecher model to the activations
                    
                    an INTINV operation returns the output state from a model 
                    for an input x2, but wiht neurons set to the values obtained when 
                    processing input x1

                """
                # perform a pass of the teacher on input x1
                teacher_outputs = self.teacher(input_ids=input_ids, attention_mask=attention_mask)

                # gather activations for another input x2
                dual_counterfactual_activations_teacher = self.get_activations(
                    self.teacher,
                    **dual_inputs,
                    variable_names=teacher_variable_names
                )
                
                counterfactual_inputs = {
                    'input_ids': counterfactual_input_ids,
                    'attention_mask': attention_mask,
                    'interchanged_variables': dual_counterfactual_activations_teacher,
                    'variable_names': teacher_interchanged_variables_mapping,
                    'interchange_mask': interchange_mask,
                    'dual_interchange_mask': dual_interchange_mask,
                }
                # perform another pass with second input with interchanged nodes
                counterfactual_outputs_teacher = self.teacher(**counterfactual_inputs)

            # logits from the teacher are then used to guide loss for the student
            t_logits, t_hidden_states = teacher_outputs["logits"], teacher_outputs["hidden_states"]
            student_inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            }
            student_outputs = self.student(**student_inputs)   # (bs, seq_length, voc_size)
            causal_t_logits = counterfactual_outputs_teacher["logits"]
        else:
            assert False

        # calculate initial loss for student
        student_outputs = self.calculate_loss(student_outputs, t_hidden_states, t_logits, lm_labels, attention_mask)
        loss_ce = student_outputs["loss_ce"].mean() if self.multi_gpu else student_outputs["loss_ce"]
        loss = self.alpha_ce * loss_ce

        def update_loss(loss_name, loss_value, loss, outputs):
            loss_ = 0
            if loss_value > 0.0:
                loss_ = outputs[loss_name].mean() if self.multi_gpu else outputs[loss_name]
                loss += loss_value * loss_ 
            return loss, loss_

        loss, loss_mlm = update_loss("loss_mlm", self.alpha_mlm, loss, student_outputs)
        loss, loss_clm = update_loss("loss_clm", self.alpha_clm, loss, student_outputs)
        loss, loss_mse = update_loss("loss_mse", self.alpha_mse, loss, student_outputs)
        loss, loss_cos = update_loss("loss_cos", self.alpha_cos, loss, student_outputs)

        # get activations with second input, x2
        dual_counterfactual_activations_student = self.get_activations(
            self.student,
            **dual_inputs,
            variable_names=student_variable_names
        )

        interchange_inputs = {
            "interchanged_variables": dual_counterfactual_activations_student,
            "variable_names": student_interchanged_variables_mapping,
            "interchange_mask": interchange_mask,
            "dual_interchange_mask": dual_interchange_mask
        }
        # perform a pass through of the original input ids with new activations
        counterfactual_outputs_student = self.student(
            input_ids=counterfactual_input_ids,
            attention_mask=attention_mask,
            **interchange_inputs
        )

        # calculate causal loss - corresponds to L_ce in the paper,
        # essentially the same as the original task in original distillation paper
        # except model outputs are now from the interchanged models
        counterfactual_outputs_student = self.calculate_causal_loss(
            counterfactual_outputs_student, lm_labels, attention_mask, causal_t_logits)
        loss, causal_loss_ce = update_loss("causal_loss_ce", self.alpha_causal_ce, loss, counterfactual_outputs_student)
                
        self.total_loss_epoch += loss.item()
        self.last_loss = loss.item()
        self.last_loss_ce = loss_ce.item()
        if self.alpha_mlm > 0.0:
            self.last_loss_mlm = loss_mlm.item()
        if self.alpha_clm > 0.0:
            self.last_loss_clm = loss_clm.item()
        if self.alpha_mse > 0.0:
            self.last_loss_mse = loss_mse.item()
        if self.alpha_cos > 0.0:
            self.last_loss_cos = loss_cos.item()

        # for logging purposes
        self.last_loss_causal_ce = causal_loss_ce.item()
            
        self.optimize(loss, skip_update_iter=skip_update_iter)

        self.n_sequences_epoch += input_ids.size(0)

    def optimize(self, loss, skip_update_iter=False):
        """
        Normalization on the loss (gradient accumulation or distributed training), followed by
        backward pass on the loss, possibly followed by a parameter update (depending on the gradient accumulation).
        Also update the metrics for tensorboard.
        """
        # Check for NaN
        if (loss != loss).data.any():
            logger.error("NaN detected")
            exit()

        if self.multi_gpu:
            loss = loss.mean()
        if self.params.gradient_accumulation_steps > 1:
            loss = loss / self.params.gradient_accumulation_steps

        if self.fp16:
            from apex import amp

            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        
        """
        In case where we want to do two mini-steps for dual on main interchange,
        and main on dual interchange (including normal objectives), we want to
        skip the iter update, so the gradients are accumulated within the step
        which includes gradients from two mini-steps.
        """
        self.iter(skip_update_iter=skip_update_iter)

        if self.n_iter % self.params.gradient_accumulation_steps == 0:
            if self.fp16:
                nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.params.max_grad_norm)
            else:
                nn.utils.clip_grad_norm_(self.student.parameters(), self.params.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

    def iter(self, skip_update_iter=False):
        """
        Update global counts, write to tensorboard and save checkpoint.
        """
        
        if not skip_update_iter:
            self.n_iter += 1
            self.n_total_iter += 1
            if self.n_total_iter % self.params.checkpoint_interval == 0:
                self.save_checkpoint()
        
        """
        Logging is not affected by the flag skip_update_iter.
        We want to log crossway effects, and losses should be
        in the same magnitude.
        """
        if self.n_total_iter % self.params.log_interval == 0:
            # self.log_tensorboard()
            self.last_log = time.time()

    def end_epoch(self):
        """
        Finally arrived at the end of epoch (full pass on dataset).
        Do some tensorboard logging and checkpoint saving.
        """
        logger.info(f"{self.n_sequences_epoch} sequences have been trained during this epoch.")

        if self.is_master:
            self.save_checkpoint(checkpoint_name=f"model_epoch_{self.epoch}.pth")
            # if self.is_wandb:
            #     wandb.log(
            #         {
            #             "epoch/loss": self.total_loss_epoch / self.n_iter, 
            #             'epoch': self.epoch
            #         }
            #     )

        self.epoch += 1
        self.n_sequences_epoch = 0
        self.n_iter = 0
        self.total_loss_epoch = 0

    def save_checkpoint(self, checkpoint_name: str = "checkpoint.pth"):
        """
        Save the current state. Only by the master process.
        """
        if not self.is_master:
            return
        mdl_to_save = self.student.module if hasattr(self.student, "module") else self.student
        mdl_to_save.config.save_pretrained(self.dump_path)
        state_dict = mdl_to_save.state_dict()
        torch.save(state_dict, os.path.join(self.dump_path, checkpoint_name))