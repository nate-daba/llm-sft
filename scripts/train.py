#!/usr/bin/env python3
import os
# Rest of your training script
project_name = 'llm-fine-tuning'
os.environ["WANDB_PROJECT"] = project_name
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

import torch
import wandb
import numpy as np
from datetime import datetime
from transformers import AutoModelForSequenceClassification

from load_data import load_and_preprocess
from transformers import DataCollatorWithPadding, TrainingArguments, Trainer, pipeline
from datasets import load_dataset
from random import randrange 
import evaluate 


# model_name = 'EleutherAI/gpt-neo-125M'
# model_name = '../checkpoints/2025-02-20_16-54-07-text-classification-ag_news/checkpoint-13500/'
# dataset_name = 'fancyzhx/ag_news' # 
# model_name = '../checkpoints/2025-02-23_00-19-57-text-classification-imdb/checkpoint-5280/'
# model_name = '../checkpoints/2025-02-23_00-49-20-text-classification-ag_news/checkpoint-20256/'
# model_name = '../checkpoints/2025-02-23_00-56-11-text-classification-imdb/checkpoint-2814/'
# model_name = '../checkpoints/2025-02-23_01-02-47-text-classification-ag_news/checkpoint-33750/'
model_name = '../checkpoints/2025-02-24_05-07-33-text-classification-ag_news/checkpoint-2250/'
dataset_name = 'fancyzhx/ag_news'
max_length = 1024
task = 'text-classification'

tokenizer, tokenized_datasets = load_and_preprocess(dataset_name=dataset_name,
                                     model_checkpoint=model_name,
                                     text_column='text',
                                     max_length=max_length)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
ckpt_dir = f'../checkpoints/{now}-{task}-{dataset_name.split("/")[-1]}'

# id2label = {0: "NEGATIVE", 1: "POSITIVE"} # uncomment this for imdb dataset
# label2id = {"NEGATIVE": 0, "POSITIVE": 1} # uncomment this for imdb dataset
id2label = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"} # uncomment this for ag_news dataset
label2id = {"World": 0, "Sports": 1, "Business": 2, "Sci/Tech": 3} # uncomment this for ag_news dataset
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4, id2label=id2label, label2id=label2id)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = model.config.eos_token_id

print(model)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    # Compute accuracy
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    # Compute F1-score with macro averaging
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")["f1"]
    
    return {"accuracy": accuracy, "f1": f1}

training_args = TrainingArguments(
    output_dir=ckpt_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=24,
    per_device_eval_batch_size=24,
    num_train_epochs=50,
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
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print(tokenized_datasets)
print(model)

# trainer.train()

test_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
print(test_results)


# =======================================================
# SENTIMENT CLASSIFICATION (GPT-2 on IMDB dataset)
# =======================================================

# dataset = load_dataset('imdb')

# baseline_classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer) 
# finetuned_model_name = '../checkpoints/2025-02-23_00-19-57-text-classification-imdb/checkpoint-5280/'
# finetuned_classifier = pipeline("sentiment-analysis", model=finetuned_model_name, tokenizer=tokenizer)

# # select a random test sample
# sample = dataset['test'][6]
# print(f"Review: \n{sample['text']}\n---------------")

# baseline_prediction = baseline_classifier(sample['text'])
# finetuned_prediction = finetuned_classifier(sample['text'])

# print('Baseline prediction:', baseline_prediction)
# print('Fine-tuned prediction:', finetuned_prediction)
# print('Ground truth:', id2label[sample['label']])
# =======================================================

# =======================================================
# SENTIMENT CLASSIFICATION (GPT-Neo on IMDB dataset)
# =======================================================

# dataset = load_dataset('imdb')

# baseline_classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer) 
# finetuned_model_name = '../checkpoints/2025-02-23_00-56-11-text-classification-imdb/checkpoint-2814/'
# finetuned_classifier = pipeline("sentiment-analysis", model=finetuned_model_name, tokenizer=tokenizer)

# # select a random test sample
# sample = dataset['test'][6]
# print(f"Review: \n{sample['text']}\n---------------")

# baseline_prediction = baseline_classifier(sample['text'])
# finetuned_prediction = finetuned_classifier(sample['text'])

# print('Baseline prediction:', baseline_prediction)
# print('Fine-tuned prediction:', finetuned_prediction)
# print('Ground truth:', id2label[sample['label']])

# =======================================================

# =======================================================
# TOPIC CLASSIFICATION (GPT-2 on AG_News)
# =======================================================
# dataset = load_dataset("fancyzhx/ag_news")

# baseline_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
# finetuned_model_name = '../checkpoints/2025-02-23_00-49-20-text-classification-ag_news/checkpoint-20256/'
# finetuned_classifier = pipeline("text-classification", model=finetuned_model_name, tokenizer=tokenizer)

# # select a random test sample
# sample = dataset['test'][6]
# print(f"News: \n{sample['text']}\n---------------")

# baseline_prediction = baseline_classifier(sample['text'])
# finetuned_prediction = finetuned_classifier(sample['text'])

# print('Baseline prediction:', baseline_prediction)
# print('Fine-tuned prediction:', finetuned_prediction)
# print('Ground truth:', id2label[sample['label']])

# =======================================================
# TOPIC CLASSIFICATION (GPT-Neo on AG_News)
# =======================================================
# dataset = load_dataset("fancyzhx/ag_news")

# baseline_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
# finetuned_model_name = '../checkpoints/2025-02-23_01-02-47-text-classification-ag_news/checkpoint-33750/'
# finetuned_classifier = pipeline("text-classification", model=finetuned_model_name, tokenizer=tokenizer)

# # select a random test sample
# sample = dataset['test'][6]
# print(f"News: \n{sample['text']}\n---------------")

# baseline_prediction = baseline_classifier(sample['text'])
# finetuned_prediction = finetuned_classifier(sample['text'])

# print('Baseline prediction:', baseline_prediction)
# print('Fine-tuned prediction:', finetuned_prediction)
# print('Ground truth:', id2label[sample['label']])

wandb.finish()

