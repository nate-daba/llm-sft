import os
project_name = 'llm-fine-tuning'
os.environ["WANDB_PROJECT"] = project_name
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3,4"

from datetime import datetime
import torch
import wandb
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
# nltk.download("punkt")
# nltk.download("punkt_tab")

from load_data import load_and_preprocess
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, pipeline
from datasets import load_dataset
from random import randrange        

import evaluate 
from huggingface_hub import HfFolder

model_name = 'google/flan-t5-base'
# model_name = '../checkpoints/2025-02-23_11-24-19-text-summarization-samsum/checkpoint-1392/'
dataset_name = "knkarthick/samsum"
max_length = 1024
task = 'text-summarization'

base_model_name = "google/flan-t5-base"  # or your original model path

# Load and preprocess the dataset
tokenizer, tokenized_dataset = load_and_preprocess(
    dataset_name=dataset_name,
    model_checkpoint=base_model_name,
    text_column='dialogue',  # Input column in the original dataset
    max_length=max_length,
    task='text-summarization'
)

# Load the model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
ckpt_dir = f'../checkpoints/{now}-{task}-{dataset_name.split("/")[-1]}'

# Metric
metric = evaluate.load("rouge")

# helper function to postprocess text
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result

# we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100
# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)

# Define training args
training_args = Seq2SeqTrainingArguments(
    output_dir=ckpt_dir,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    predict_with_generate=True,
    fp16=False, # Overflows with fp16
    learning_rate=5e-5,
    num_train_epochs=40,
    # logging & evaluation strategies
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="wandb"
)

# Create Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics=compute_metrics,
)

print(tokenized_dataset)
print(model)

# trainer.train()

# test_results = trainer.evaluate(eval_dataset=tokenized_dataset["test"])
# print(test_results)

dataset_id = "samsum"
dataset = load_dataset(dataset_id)

# load model and tokenizer from huggingface hub with pipeline
summarizer = pipeline("summarization", 
                    #   model="../checkpoints/2025-02-23_11-24-19-text-summarization-samsum/checkpoint-1392/", 
                        model="google/flan-t5-base",
                        tokenizer=base_model_name, 
                        device=0)

# select a random test sample
sample = dataset['test'][randrange(len(dataset["test"]))]
print(f"dialogue: \n{sample['dialogue']}\n---------------")

# summarize dialogue
res = summarizer(sample["dialogue"])

print(f"flan-t5-base summary:\n{res[0]['summary_text']}")

wandb.finish()