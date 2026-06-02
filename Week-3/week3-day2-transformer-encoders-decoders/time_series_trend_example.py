# Install:
# pip install torch numpy matplotlib

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt   # <-- Added for plotting

# -----------------------------
# 1. Create example time-series data
# -----------------------------

np.random.seed(42)

time = np.arange(1000)
series = 0.05 * time + 10 * np.sin(time * 0.05) - np.random.normal(0, 2, size=len(time))

mean = series.mean()
std = series.std()
series = (series - mean) / std

# Plot raw normalized series
plt.figure(figsize=(12, 4))
plt.plot(time, series, label="Normalized Series")
plt.title("Synthetic Time-Series Data")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.show()

# -----------------------------
# 2. Convert series into trend labels
# -----------------------------

def get_trend_label(current_value, future_value, threshold=0.05):
    change = future_value - current_value
    if change > threshold:
        return 2  # UP
    elif change < -threshold:
        return 0  # DOWN
    else:
        return 1  # STABLE

sequence_length = 20
future_steps = 5

X, y = [], []
for i in range(len(series) - sequence_length - future_steps):
    past_window = series[i : i + sequence_length]
    current_value = series[i + sequence_length - 1]
    future_value = series[i + sequence_length + future_steps - 1]
    trend = get_trend_label(current_value, future_value)
    X.append(past_window)
    y.append(trend)

X = np.array(X)
y = np.array(y)

# -----------------------------
# 3. Dataset
# -----------------------------

class TrendDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index].unsqueeze(-1), self.y[index]

dataset = TrendDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# -----------------------------
# 4. Transformer model
# -----------------------------

class TrendTransformer(nn.Module):
    def __init__(self, input_dim=1, embed_dim=64, num_heads=4,
                 hidden_dim=128, num_layers=2, num_classes=3, max_len=100):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=hidden_dim, dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.input_projection(x) + self.position_embedding(positions)
        encoded = self.transformer_encoder(x)
        final_output = encoded[:, -1, :]
        return self.classifier(final_output)

model = TrendTransformer(max_len=sequence_length)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

# -----------------------------
# 5. Train model
# -----------------------------

epochs = 20
losses = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in loader:
        logits = model(batch_X)
        loss = criterion(logits, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    losses.append(total_loss)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

# Plot training loss
plt.figure(figsize=(8, 4))
plt.plot(range(1, epochs + 1), losses, marker="o")
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# -----------------------------
# 6. Predict trend
# -----------------------------

def predict_trend(values):
    model.eval()
    values = np.array(values)
    values = (values - mean) / std
    x = torch.tensor(values, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    with torch.no_grad():
        logits = model(x)
        prediction = torch.argmax(logits, dim=1).item()
    labels = {0: "DOWN", 1: "STABLE", 2: "UP"}
    return labels[prediction]

recent_values = series[-sequence_length:] * std + mean
print("Recent values:", recent_values)
print("Predicted trend:", predict_trend(recent_values))

# Plot recent values with predicted trend
plt.figure(figsize=(10, 4))
plt.plot(range(sequence_length), recent_values, marker="o", label="Recent Values")
plt.title(f"Recent Window - Predicted Trend: {predict_trend(recent_values)}")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.legend()
plt.show()
