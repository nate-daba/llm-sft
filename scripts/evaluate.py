#!/usr/bin/env python3

import argparse
import torch
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score
from load_data import load_and_preprocess  # Importing the existing data loader

class Evaluator:
    def __init__(self, model_name, dataset_name, task='classification', split='test', batch_size=16, max_length=1024):
        """
        Initialize the Evaluator with model and dataset details.

        Args:
            model_name (str): The name or path of the model to evaluate.
            dataset_name (str): The name of the dataset to load.
            task (str, optional): The task type ('classification'). Defaults to 'classification'.
            split (str, optional): The dataset split to use for evaluation. Defaults to 'test'.
            batch_size (int, optional): The batch size for evaluation. Defaults to 16.
            max_length (int, optional): The maximum sequence length for tokenization. Defaults to 512.
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.task = task
        self.split = split
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the model based on the task
        if self.task == 'classification':
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        else:
            raise ValueError(f"Unsupported task type: {self.task}")

        self.model.to(self.device)
        self.model.eval()

    def evaluate_classification(self, dataset):
        """
        Evaluate the model on a classification task.

        Args:
            dataset (Dataset): The dataset to evaluate on.

        Returns:
            dict: A dictionary containing accuracy and F1-score.
        """
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)

        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')

        return {'accuracy': accuracy, 'f1_score': f1}

    def evaluate(self):
        """
        Run evaluation based on the specified task.
        """
        # Load and preprocess the dataset using the existing function
        tokenized_datasets = load_and_preprocess(self.dataset_name, self.model_name, max_length=self.max_length)
        dataset = tokenized_datasets[self.split]

        if self.task == 'classification':
            return self.evaluate_classification(dataset)
        else:
            raise ValueError(f"Unsupported task type: {self.task}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate a language model on a specified task.")
    parser.add_argument('--model_name', type=str, required=True, help="The name or path of the model to evaluate.")
    parser.add_argument('--dataset_name', type=str, required=True, help="The name of the dataset to load.")
    parser.add_argument('--task', type=str, default='classification', choices=['classification'], help="The task type to evaluate.")
    parser.add_argument('--split', type=str, default='test', help="The dataset split to use for evaluation.")
    parser.add_argument('--batch_size', type=int, default=16, help="The batch size for evaluation.")
    parser.add_argument('--max_length', type=int, default=1024, help="The maximum sequence length for tokenization.")

    args = parser.parse_args()

    evaluator = Evaluator(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        task=args.task,
        split=args.split,
        batch_size=args.batch_size,
        max_length=args.max_length
    )

    results = evaluator.evaluate()
    print(f"Evaluation results: {results}")

if __name__ == "__main__":
    main()