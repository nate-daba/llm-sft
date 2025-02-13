from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

def load_and_preprocess(dataset_name, model_checkpoint='gpt2', text_column='text', max_length=1024):
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
    dataset = load_dataset(dataset_name)

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
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples[text_column],
            padding='max_length',
            truncation=True,
            max_length=max_length
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    return tokenized_datasets