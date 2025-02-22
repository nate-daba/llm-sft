#!/usr/bin/env python3
import os
project_name = 'llm-fine-tuning'
os.environ["WANDB_PROJECT"] = project_name
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3,4"

import torch
import wandb
import numpy as np
from datetime import datetime
from transformers import AutoModelForSequenceClassification

from load_data import load_and_preprocess
from transformers import DataCollatorWithPadding, TrainingArguments, Trainer
import evaluate 

# model_name = 'gpt2'
model_name = 'EleutherAI/gpt-neo-125M'
dataset_name = 'fancyzhx/ag_news' # 
dataset_name = 'imdb'
max_length = 1024
task = 'text-classification'

tokenizer, tokenized_datasets = load_and_preprocess(dataset_name=dataset_name,
                                     model_checkpoint=model_name,
                                     text_column='text',
                                     max_length=max_length)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
accuracy = evaluate.load("accuracy")

now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
ckpt_dir = f'../checkpoints/{now}-{task}-{dataset_name.split("/")[-1]}'

# id2label = {0: "NEGATIVE", 1: "POSITIVE"}
# label2id = {"NEGATIVE": 0, "POSITIVE": 1}
id2label = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
label2id = {"World": 0, "Sports": 1, "Business": 2, "Sci/Tech": 3}
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4, id2label=id2label, label2id=label2id)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = model.config.eos_token_id
model.train()

print(model)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir=ckpt_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=24,
    per_device_eval_batch_size=24,
    num_train_epochs=15,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="wandb",
    run_name=f'run-{now}-{task}-{dataset_name.split("/")[-1]}'
)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

wandb.finish()

