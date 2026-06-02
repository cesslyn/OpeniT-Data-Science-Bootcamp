# Install first:
# pip install transformers torch

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load a Transformer model and its BPE-based tokenizer
model_name = "trained_transformer_model"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Input sentence
sentence = input("Enter a sentence: ")

# Tokenize the sentence using the tokenizer's subword/BPE-style encoding
inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)

# Make prediction
with torch.no_grad():
    outputs = model(**inputs)

# Get predicted class
logits = outputs.logits
predicted_class_id = torch.argmax(logits, dim=1).item()

# Convert class ID to label
label = model.config.id2label[predicted_class_id]

print("Input sentence:", sentence)
print("Prediction:", label)
