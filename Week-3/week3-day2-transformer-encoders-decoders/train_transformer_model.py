import argparse
import csv
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_DATA = [
    ("I loved the product and would buy it again.", 1),
    ("This is one of the best experiences I have had.", 1),
    ("Everything worked perfectly and the service was great.", 1),
    ("The movie was fun, heartwarming, and well acted.", 1),
    ("I am very happy with the final result.", 1),
    ("The support team fixed my problem quickly.", 1),
    ("The app keeps crashing and is very frustrating.", 0),
    ("I regret buying this because it does not work.", 0),
    ("The experience was disappointing and a waste of time.", 0),
    ("This was poorly made and broke immediately.", 0),
    ("I am unhappy with the quality of the service.", 0),
    ("The update made the software worse.", 0),
]


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoding = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {key: value.squeeze(0) for key, value in encoding.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def load_csv_dataset(csv_path):
    texts = []
    labels = []

    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)

        if "text" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError("CSV file must contain 'text' and 'label' columns.")

        for row in reader:
            texts.append(row["text"])
            labels.append(int(row["label"]))

    if not texts:
        raise ValueError("CSV file is empty.")

    unique_labels = sorted(set(labels))
    if unique_labels != list(range(len(unique_labels))):
        raise ValueError(
            "Labels must be zero-based contiguous integers like 0,1 or 0,1,2."
        )

    return texts, labels


def load_training_data(csv_path):
    if csv_path:
        return load_csv_dataset(csv_path)

    texts = [text for text, _ in DEFAULT_DATA]
    labels = [label for _, label in DEFAULT_DATA]
    return texts, labels


def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_examples = 0

    with torch.no_grad():
        for batch in data_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)

            total_loss += outputs.loss.item()
            predictions = torch.argmax(outputs.logits, dim=1)
            correct_predictions += (predictions == batch["labels"]).sum().item()
            total_examples += batch["labels"].size(0)

    average_loss = total_loss / max(len(data_loader), 1)
    accuracy = correct_predictions / max(total_examples, 1)
    return average_loss, accuracy


def train(args):
    torch.manual_seed(args.seed)

    texts, labels = load_training_data(args.csv_path)
    num_labels = len(set(labels))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
    )

    dataset = TextClassificationDataset(texts, labels, tokenizer, args.max_length)

    validation_size = max(1, int(len(dataset) * args.validation_split))
    train_size = len(dataset) - validation_size

    if train_size < 1:
        raise ValueError("Not enough examples for the requested validation split.")

    train_dataset, validation_dataset = random_split(
        dataset,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        average_train_loss = total_train_loss / max(len(train_loader), 1)
        validation_loss, validation_accuracy = evaluate(
            model, validation_loader, device
        )

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {average_train_loss:.4f} | "
            f"Val Loss: {validation_loss:.4f} | "
            f"Val Accuracy: {validation_accuracy:.2%}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\nSaved trained model to: {output_dir.resolve()}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train a Transformer text-classification model."
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="Optional CSV with 'text' and 'label' columns.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="distilbert-base-uncased",
        help="Hugging Face model checkpoint to fine-tune.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="trained_transformer_model",
        help="Directory where the trained model will be saved.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    train(parser.parse_args())
