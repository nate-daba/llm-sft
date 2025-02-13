from datasets import load_dataset

def load_and_preprocess(dataset_name):
    """
    Load and preprocess the specified dataset.

    Args:
        dataset_name (str): The name of the dataset to load.

    Returns:
        DatasetDict: A dictionary with 'train', 'validation', and 'test' splits.
    """
    # Load the dataset
    dataset = load_dataset(dataset_name)

    # Split the dataset into train, validation, and test sets
    if 'train' not in dataset or 'test' not in dataset:
        raise ValueError("Dataset must have 'train' and 'test' splits.")

    # Create a validation split from the training data if not present
    if 'validation' not in dataset:
        dataset['train'] = dataset['train'].train_test_split(test_size=0.1, seed=42)
        dataset['validation'] = dataset['train']['test']
        dataset['train'] = dataset['train']['train']

    return dataset