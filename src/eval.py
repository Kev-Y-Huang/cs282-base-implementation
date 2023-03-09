from copy import deepcopy
from transformers import BertForSequenceClassification, AutoModelForSequenceClassification, AutoTokenizer, AutoModel, DataCollatorWithPadding
from torch import nn
from datasets import load_dataset
from collections import OrderedDict
from transformers import AdamW, get_scheduler
import torch 
from torch.utils.data import (
    Dataset,
    DataLoader,
)
from torch.utils.data import Subset
from tqdm import tqdm, trange
import evaluate as e
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac

from load_glue import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, dataloader, tokenizer):
    model.eval()
    val_acc = 0

    # original model performance
    metric1 = e.load("glue", "qqp")
    for n, inputs in enumerate(tqdm(dataloader)):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric1.add_batch(predictions=predictions, references=inputs["labels"])
    original_scores = metric1.compute()

    # robustness - replace characters
    metric2 = e.load("glue", "qqp")
    aug_charsub = nac.RandomCharAug(action='substitute', aug_char_min=5, aug_char_max=5)
    for n, inputs in enumerate(tqdm(dataloader)):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        augmented_inputs = inputs.copy()
        augmented_inputs["input_ids"], augmented_inputs["attention_mask"] = [], []
        for i in range(len(inputs["input_ids"])):
            text = tokenizer.decode(inputs["input_ids"][i], skip_special_tokens=True)
            augmented_text = aug_charsub.augment(text)
            augmented_input = tokenizer(augmented_text, max_length=len(inputs["input_ids"][i]),
                                        padding='max_length', truncation=True)
            print(sum([x!=inputs["input_ids"][i][j] for j, x in enumerate(augmented_input["input_ids"][0])]))
            print(sum([x!=inputs["attention_mask"][i][j] for j, x in enumerate(augmented_input["attention_mask"][0])]))
            augmented_inputs["input_ids"] += augmented_input["input_ids"]
            augmented_inputs["attention_mask"] += augmented_input["attention_mask"]
        augmented_inputs["input_ids"] = torch.tensor(augmented_inputs["input_ids"]).to(device)
        augmented_inputs["attention_mask"] = torch.tensor(augmented_inputs["attention_mask"]).to(device)
        with torch.no_grad():
            outputs = model(**augmented_inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric2.add_batch(predictions=predictions, references=augmented_inputs["labels"])
    charsub_scores = metric2.compute()
    
    # robustness - swap adjacent word
    metric3 = e.load("glue", "qqp")
    aug_wordswap = naw.RandomWordAug(action="swap", aug_p=0.4, aug_min=5, aug_max=5)
    for n, inputs in enumerate(tqdm(dataloader)):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        augmented_inputs = inputs.copy()
        augmented_inputs["input_ids"], augmented_inputs["attention_mask"] = [], []
        for i in range(len(inputs["input_ids"])):
            text = tokenizer.decode(inputs["input_ids"][i], skip_special_tokens=True)
            augmented_text = aug_wordswap.augment(text)
            augmented_input = tokenizer(augmented_text, max_length=len(inputs["input_ids"][i]),
                                        padding='max_length', truncation=True)
            augmented_inputs["input_ids"] += augmented_input["input_ids"]
            augmented_inputs["attention_mask"] += augmented_input["attention_mask"]
        augmented_inputs["input_ids"] = torch.tensor(augmented_inputs["input_ids"]).to(device)
        augmented_inputs["attention_mask"] = torch.tensor(augmented_inputs["attention_mask"]).to(device)
        with torch.no_grad():
            outputs = model(**augmented_inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric3.add_batch(predictions=predictions, references=augmented_inputs["labels"])
    wordswap_scores = metric3.compute()

    return original_scores, charsub_scores, wordswap_scores


def main():

    dataset = load_dataset("glue", "qqp")
    config = GLUE_CONFIGS["qqp"]
    print(dataset)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    train, val, test = tokenize_pair(tokenizer, dataset, **config)
    val.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    #val_dataloader = DataLoader(val, shuffle=True, batch_size=4, collate_fn=data_collator)
    val_subset = Subset(val, range(100))
    val_subset_dataloader = DataLoader(val_subset, shuffle=True, batch_size=4, collate_fn=data_collator)

    evaluate(model, val_subset_dataloader, tokenizer)


if __name__ == "__main__":
    main()