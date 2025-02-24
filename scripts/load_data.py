import sys
from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer

def load_and_preprocess(dataset_name, model_checkpoint='gpt2', text_column='text', max_length=1024, task="text-classification"):
    """
    Load, preprocess, and tokenize the specified dataset.

    Args:
        dataset_name (str): The name of the dataset to load.
        model_checkpoint (str): The model checkpoint for the tokenizer.
        text_column (str, optional): The name of the text column in the dataset. Defaults to 'text'.
        max_length (int, optional): The maximum sequence length for tokenization. Defaults to 1024.

    Returns:
        DatasetDict: A dictionary with tokenized 'train', 'validation', and 'test' splits.
    """
    # Load the dataset
    dataset = load_dataset(dataset_name, cache_dir='/workspace/ndaba/research/data/llm_data')
    
    # Remove examples with None or empty strings in the text column
    dataset = dataset.filter(lambda x: x[text_column] is not None and x[text_column].strip() != '')
    
    # Check for existing 'train', 'validation', and 'test' splits
    if all(split in dataset for split in ['train', 'validation', 'test']):
        train_dataset = dataset['train']
        validation_dataset = dataset['validation']
        test_dataset = dataset['test']
    else:
        # Split the dataset into train, validation, and test sets
        if 'train' not in dataset or 'test' not in dataset:
            raise ValueError("Dataset must have 'train' and 'test' splits.")

        if 'validation' not in dataset:
            dataset['train'] = dataset['train'].train_test_split(test_size=0.1, seed=42)
            dataset['validation'] = dataset['train']['test']
            dataset['train'] = dataset['train']['train']

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # Assign the end-of-sequence token as the padding token
    # add pad token if none exists
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    if task == "text-classification":
        # Tokenize the dataset
        def tokenize_function(examples):
            return tokenizer(
                examples[text_column],
                padding='max_length',
                truncation=True,
                max_length=max_length
            )
        tokenized_datasets = dataset.map(tokenize_function, batched=True)
    elif task == "text-summarization":
        # Borrowed from https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/flan-t5-samsum-summarization.ipynb
        # The maximum total input sequence length after tokenization. 
        # Sequences longer than this will be truncated, sequences shorter will be padded.
        tokenized_inputs = concatenate_datasets([dataset["train"], 
                                                 dataset["test"]]).map(lambda x: 
                                                     tokenizer(x["dialogue"], 
                                                               truncation=True), 
                                                     batched=True, 
                                                     remove_columns=["dialogue", 
                                                                     "summary"])
        max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
        print(f"Max source length: {max_source_length}")

        # The maximum total sequence length for target text after tokenization. 
        # Sequences longer than this will be truncated, sequences shorter will be padded."
        tokenized_targets = concatenate_datasets([dataset["train"], 
                                                  dataset["test"]]).map(lambda x: 
                                                      tokenizer(x["summary"], 
                                                                truncation=True), 
                                                      batched=True, 
                                                      remove_columns=["dialogue", 
                                                                      "summary"])
        max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])
        print(f"Max target length: {max_target_length}")
        
        def tokenize_function(sample,padding="max_length"):
            # add prefix to the input for t5
            inputs = ["summarize: " + item for item in sample["dialogue"]]

            # tokenize inputs
            model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

            # Tokenize targets with the `text_target` keyword argument
            labels = tokenizer(text_target=sample["summary"], max_length=max_target_length, padding=padding, truncation=True)

            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
            # padding in the loss.
            if padding == "max_length":
                labels["input_ids"] = [
                    [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["dialogue", "summary", "id"])
        print(f"Keys of tokenized dataset: {list(tokenized_datasets['train'].features)}")

    else:
        print('Unrecognized task:', task)
        sys.exit()

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    return tokenizer, tokenized_datasets