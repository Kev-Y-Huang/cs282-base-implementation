# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team.
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
"""
Preprocessing script before distillation.
"""

# modified from original binarized_data.py script
import argparse
import logging
import pickle
import random
import time

import numpy as np

from transformers import BertTokenizer

from datasets import load_dataset, load_metric
from datasets import Dataset
from datasets import DatasetDict

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)

def tokenize_function(tokenizer, examples):
  token_ids = tokenizer(
    examples["text"], 
    return_token_type_ids=False,
    return_attention_mask=False,
    return_overflowing_tokens=False,
    return_special_tokens_mask=False,
    return_offsets_mapping=False,
    return_length=False,
  )
  return token_ids


def main():
  parser = argparse.ArgumentParser(
    description="Preprocess the data to avoid re-doing it several times by (tokenization + token_to_ids)."
  )
  parser.add_argument("--dataset_name", type=str, required=True, help="The path to the data.")
  parser.add_argument("--cache_dir", type=str, default="distill_cache/", help="The path to the data.")
  parser.add_argument("--split", type=str, default="train", help="The split to parse.")
  parser.add_argument("--tokenizer_name", type=str, default="bert-base-uncased", help="The tokenizer to use.")
  parser.add_argument("--dump_file", type=str, default="data/dump", help="The dump file prefix.")
  parser.add_argument("--preprocessing_num_workers", type=int, default=10, help="Number of process to preprocess the dataset")

  args = parser.parse_args()
  tokenizer = BertTokenizer.from_pretrained(
    args.tokenizer_name,
    cache_dir=args.cache_dir
  )

  bos = tokenizer.special_tokens_map["cls_token"]  # `[CLS]`
  sep = tokenizer.special_tokens_map["sep_token"]  # `[SEP]`

  all_datasets = []
  dataset_names = []
  for dataset_n in args.dataset_name.split("+"):
    dataset_names += [dataset_n]
    # logger.info(f"Loading text from {dataset_n}")
    if dataset_n == "wikitext":
        dataset = load_dataset(
            "wikitext", "wikitext-103-v1",
            cache_dir=args.cache_dir
        )
    else:
        logger.info(f"loading {dataset_n}...")
        dataset = load_dataset(
            dataset_n,
            cache_dir=args.cache_dir
        )
    all_datasets += [dataset]
  logger.info("Finished loading datasets!")
  # encoding 

  rslt = [] 
  iter = 0 
  interval = 10000
  start = time.time() 

  for i, dataset in enumerate(all_datasets):
    logger.info(f"Multiprocessing dataset = {dataset_names[i]}.")
    tokenize_dataset = dataset[args.split].map(
        lambda e: tokenize_function(tokenizer, e),
        batched=True,
        num_proc=args.preprocessing_num_workers
     )
    iter = 0 
    for example in tokenize_dataset:
      input_ids= example["input_ids"]
      rslt.append(input_ids)
      iter += 1 
      if iter % interval == 0:
          end = time.time()
          logger.info(f"{iter} examples processed. - {(end-start):.2f}s/{interval}expl")
          start = time.time()

  logger.info("Finished binarization")
  logger.info(f"{len(rslt)} examples processed.")

  dp_file = f"{args.dump_file}.{args.split}.{args.tokenizer_name}.pickle"
  vocab_size = tokenizer.vocab_size
  if vocab_size < (1 << 16):
        rslt_ = [np.uint16(d) for d in rslt]
  else:
      rslt_ = [np.int32(d) for d in rslt]
  random.shuffle(rslt_)
  logger.info(f"Dump to {dp_file}")
  with open(dp_file, "wb") as handle:
      pickle.dump(rslt_, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
  main()