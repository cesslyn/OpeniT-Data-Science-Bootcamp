import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score


# Load dataset
df = pd.read_csv("transformer_workbook_dataset.csv.csv")

# Keep only encoder classification rows
df = df[df["task_type"] == "encoder_classification"]

# Split dataset
train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "validation"].copy()
test_df = df[df["split"] == "test"].copy()


# Convert labels to integers
label_map = {
    label: idx
    for idx, label in enumerate(sorted(df["label"].unique()))
}

df["label_id"] = df["label"].map(label_map)

train_df["label_id"] = train_df["label"].map(label_map)
val_df["label_id"] = val_df["label"].map(label_map)
test_df["label_id"] = test_df["label"].map(label_map)

reverse_label_map = {
    v: k for k, v in label_map.items()
}

print("Label Map:", label_map)

# Dataset class
class EncoderDataset(Dataset):

    def __init__(
        self,
        texts,
        labels,
        tokenizer,
        max_length=128,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            key: value.squeeze(0)
            for key, value in encoding.items()
        }

        item["labels"] = torch.tensor(
            self.labels[idx],
            dtype=torch.long,
        )

        return item


# Tokenizer and model

model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label_map),
)


# Create datasets

train_dataset = EncoderDataset(
    train_df["input_text"].tolist(),
    train_df["label_id"].tolist(),
    tokenizer,
)

val_dataset = EncoderDataset(
    val_df["input_text"].tolist(),
    val_df["label_id"].tolist(),
    tokenizer,
)

test_dataset = EncoderDataset(
    test_df["input_text"].tolist(),
    test_df["label_id"].tolist(),
    tokenizer,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=2,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=2,
)


#  Device

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

model.to(device)


#  Optimizer
optimizer = AdamW(
    model.parameters(),
    lr=2e-5,
)

# Train model
epochs = 5

best_val_accuracy = 0

for epoch in range(epochs):

    # TRAINING
    model.train()

    total_loss = 0

    train_predictions = []
    train_targets = []

    for batch in train_loader:

        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }

        outputs = model(**batch)

        loss = outputs.loss

        logits = outputs.logits

        preds = torch.argmax(
            logits,
            dim=1,
        )

        train_predictions.extend(
            preds.cpu().numpy()
        )

        train_targets.extend(
            batch["labels"].cpu().numpy()
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    train_accuracy = accuracy_score(
        train_targets,
        train_predictions,
    )


    # VALIDATION
    model.eval()

    val_predictions = []
    val_targets = []

    with torch.no_grad():

        for batch in val_loader:

            labels = batch["labels"]

            batch = {
                k: v.to(device)
                for k, v in batch.items()
            }

            outputs = model(**batch)

            preds = torch.argmax(
                outputs.logits,
                dim=1,
            )

            val_predictions.extend(
                preds.cpu().numpy()
            )

            val_targets.extend(
                labels.numpy()
            )

    val_accuracy = accuracy_score(
        val_targets,
        val_predictions,
    )

    # Save best model

    if val_accuracy >= best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "best_model.pt",
        )

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Loss: {total_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.4f} | "
        f"Validation Accuracy: {val_accuracy:.4f}"
    )

# Load best model
model.load_state_dict(
    torch.load("best_model.pt")
)


# Evaluate on test set
model.eval()

test_predictions = []
test_targets = []

with torch.no_grad():

    for batch in test_loader:

        labels = batch["labels"]

        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }

        outputs = model(**batch)

        preds = torch.argmax(
            outputs.logits,
            dim=1,
        )

        test_predictions.extend(
            preds.cpu().numpy()
        )

        test_targets.extend(
            labels.numpy()
        )

test_accuracy = accuracy_score(
    test_targets,
    test_predictions,
)

print("\nTest Accuracy:", test_accuracy)


# Inference
examples = [
    "The model produced useful summaries.",
    "The output was confusing and inaccurate.",
    "The explanations were detailed and clear.",
]

print("\n--- Predictions ---")

for text in examples:

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    with torch.no_grad():

        outputs = model(**inputs)

    prediction = torch.argmax(
        outputs.logits,
        dim=1,
    ).item()

    predicted_label = reverse_label_map[prediction]

    print("\nInput:", text)
    print("Prediction:", predicted_label)

