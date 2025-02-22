from load_data import load_and_preprocess  # Importing the existing data loader
from transformers import AutoModelForSeq2SeqLM
from evaluate import evaluator

# Initialize the summarization evaluator
task_evaluator = evaluator("summarization")

# Model and dataset details
model_name = 'google-t5/t5-small'
dataset_name = "knkarthick/samsum"
max_length = 1024

# Load and preprocess the dataset
tokenizer, tokenized_data = load_and_preprocess(
    dataset_name=dataset_name,
    model_checkpoint=model_name,
    text_column='dialogue',  # Input column in the original dataset
    max_length=max_length,
    task='text-summarization'
)

# Load the model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = model.config.eos_token_id

# Evaluate the model
results = task_evaluator.compute(
    model_or_pipeline=model,
    data=tokenized_data['test'],
    metric="rouge",
    strategy="bootstrap",
    n_resamples=10,
    random_state=0,
    tokenizer=tokenizer,
    input_column="dialogue",  # Specify the input column name
    label_column="summary"   # Specify the label column name
)

print(results)


