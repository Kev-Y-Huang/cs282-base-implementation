# CS 282 Baseline Implementation

Checkpoint 2 for Harvard's CS282BR: Interpretability and Explainability

Project Members: Chelsea (Zixi) Chen, Kevin Huang, Steve Li

## How to Run

This procedure requires some preprocessing of data. The original paper cites the original HuggingFace implementation of preprocessing/tokenizing data to be run for distillation. The procedure goes as follows:

### 1. Tokenize Data

```
python scripts/binarized_data.py \
--dataset_name wikitext \
--split train \
--field_name text \
--tokenizer_type bert \
--tokenizer_name bert-base-uncased \
--dump_file bookcorpus-dataset/binarized_text \
--cache_dir ./distill_cache/
```

### 2. Calculate Token Counts

```
python scripts/token_counts.py \
--data_file data/binarized_text.train.bert-base-uncased.pickle \
--token_counts_dump data/binarized_text.train.token_counts.bert-base-uncased.pickle \
--vocab_size 30522
```

### 3. Initialize student model weights for distillation

```
python scripts/extract_distilbert.py \
--model_type bert \
--model_name bert-base-uncased \
--dump_checkpoint ./distillation_checkpoints/bert-base-uncased_num_layer_3.pth \
--num_layers 3
```

After preprocessing all the data, the distillation process can be run using this example command:

```
CUDA_VISIBLE_DEVICES=0 python causal_train.py \
--force \
--n_gpu 1 \
--log_interval 10 \
--student_type distilbert \
--student_config ./training_configs/distilbert-base-uncased-large.json \
--student_pretrained_weights ./distillation_checkpoints/bert-base-uncased_num_layer_3.pth \
--teacher_type bert \
--teacher_name bert-base-uncased \
--neuron_mapping ./training_configs/single_middle_layer_6.nm \
--mlm --alpha_ce 0.25 --alpha_mlm 0.25 --alpha_cos 0.25 --alpha_clm 0.0 --alpha_causal_ce 0.25  \
--freeze_pos_embs \
--dump_path ./results/ \
--data_file ./scripts/wikitext_dataset/binarized_text.train.bert-base-uncased.pickle \
--token_counts ./scripts/wikitext_dataset/token_counts.bert-base-uncased.pickle \
--seed 42 \
--n_epoch 3 \
--gradient_accumulation_steps 6 \
--batch_size 1
```

We highly recommend using multiple GPUs for training. Multi-GPU support is done using the `torch.distributed` module. More documentation can be found here: https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization.

Essentially, change the variable in `CUDA_VISIBLE_DEVICES` to a comma separate list of how many GPUs you have on a single node, and change the `--n_gpu` argument to match that number. For example, if I had a node with 4 GPUs my command would look like the following:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python...
...
--n_gpu 4
```

Additionally, we have to set global environment variables, namely:

```
export NODE_RANK=0
export N_NODES=1
export N_GPU_NODE=4
export WORLD_SIZE=4
export MASTER_PORT=12355
export MASTER_ADDR=localhost
```

Evaluation of the model can be done with the following script:

```
python bert_experiments.py --task <GLUE task name>
```

Make sure to change the model path in `bert_experiments.py` as well.
