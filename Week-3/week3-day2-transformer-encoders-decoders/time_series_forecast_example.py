# Install:
# pip install torch numpy matplotlib

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# -----------------------------
# 1. Create sample time-series data
# -----------------------------

np.random.seed(42)

time = np.arange(1200)

series = (
    0.03 * time + 10 * np.sin(time * 0.05) + np.random.normal(0, 1.5, size=len(time))
)

# Normalize data
mean = series.mean()
std = series.std()
series_norm = (series - mean) / std


# -----------------------------
# 2. Create input/output windows
# -----------------------------

input_length = 30  # past 30 time steps
forecast_length = 5  # predict next 5 time steps

X = []
y = []

for i in range(len(series_norm) - input_length - forecast_length):
    past_values = series_norm[i : i + input_length]
    future_values = series_norm[i + input_length : i + input_length + forecast_length]

    X.append(past_values)
    y.append(future_values)

X = np.array(X)
y = np.array(y)


# -----------------------------
# 3. Dataset
# -----------------------------


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # X shape: [input_length, 1]
        # y shape: [forecast_length]
        return self.X[index].unsqueeze(-1), self.y[index]


dataset = TimeSeriesDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)


# -----------------------------
# 4. Transformer forecasting model
# -----------------------------


class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        input_dim=1,
        embed_dim=64,
        num_heads=4,
        hidden_dim=128,
        num_layers=2,
        forecast_length=5,
        max_len=100,
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.forecast_head = nn.Linear(embed_dim, forecast_length)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        positions = torch.arange(seq_len, device=x.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)

        x = self.input_projection(x)
        x = x + self.position_embedding(positions)

        encoded = self.transformer_encoder(x)

        # Use the final time step representation
        last_hidden_state = encoded[:, -1, :]

        forecast = self.forecast_head(last_hidden_state)

        return forecast


model = TimeSeriesTransformer(forecast_length=forecast_length, max_len=input_length)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)


# -----------------------------
# 5. Train the model
# -----------------------------

epochs = 30

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch_X, batch_y in loader:
        prediction = model(batch_X)
        loss = criterion(prediction, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")


# -----------------------------
# 6. Predict future values
# -----------------------------


def predict_future(recent_values):
    """
    recent_values should contain exactly input_length values.
    Example: [101.2, 102.5, 103.1, ...]
    """

    model.eval()

    recent_values = np.array(recent_values)

    # Normalize using training statistics
    recent_values_norm = (recent_values - mean) / std

    x = torch.tensor(recent_values_norm, dtype=torch.float32)
    x = x.unsqueeze(0).unsqueeze(-1)

    with torch.no_grad():
        prediction_norm = model(x)

    # Convert prediction back to original scale
    prediction = prediction_norm.numpy()[0] * std + mean

    return prediction


recent_values = series[-input_length:]

future_prediction = predict_future(recent_values)

print("Recent values:")
print(recent_values)

print("\nPredicted future values:")
print(future_prediction)


# -----------------------------
# 7. Plot prediction
# -----------------------------

plt.figure(figsize=(10, 5))

past_x = np.arange(input_length)
future_x = np.arange(input_length, input_length + forecast_length)

plt.plot(past_x, recent_values, label="Past values")
plt.plot(future_x, future_prediction, marker="o", label="Predicted future")

plt.xlabel("Time step")
plt.ylabel("Value")
plt.title("Transformer Time-Series Forecast")
plt.legend()
plt.show()
