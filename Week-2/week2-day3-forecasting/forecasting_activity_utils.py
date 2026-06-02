from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from torch.utils.data import DataLoader, TensorDataset


def make_loader(X, y, batch_size=32, shuffle=False):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def rmse_mae(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    return {"rmse": rmse, "mae": mae}


def predict_model(model, loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            batch_predictions = model(X_batch).detach().numpy().reshape(-1)
            predictions.extend(batch_predictions)
            actuals.extend(y_batch.numpy().reshape(-1))
    return np.array(actuals), np.array(predictions)


def plot_forecast(dates, actual, forecast, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=actual, mode="lines+markers", name="Actual"))
    fig.add_trace(go.Scatter(x=dates, y=forecast, mode="lines+markers", name="Forecast"))
    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="Demand score",
    )
    fig.show()


def make_sequence_arrays(frame: pd.DataFrame, feature_columns, target_column, window_size):
    """Create rolling sequence windows.

    Sequence input shape is [samples, timesteps, features]. Each sample uses
    prior rows to forecast the target at the current row.
    """
    feature_values = frame[feature_columns].to_numpy(dtype=float)
    target_values = frame[target_column].to_numpy(dtype=float)
    dates = frame["date"].to_numpy()
    day_indices = frame["day_index"].to_numpy(dtype=int)

    X, y, target_dates, target_day_indices = [], [], [], []
    for end_index in range(window_size, len(frame)):
        start_index = end_index - window_size
        X.append(feature_values[start_index:end_index])
        y.append(target_values[end_index])
        target_dates.append(pd.to_datetime(dates[end_index]) + pd.Timedelta(days=1))
        target_day_indices.append(day_indices[end_index] + 1)

    return np.array(X), np.array(y), np.array(target_dates), np.array(target_day_indices)


def _apply_standardizer(values: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (np.asarray(values, dtype=float) - mean) / std


def _inverse_standardizer(values: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (np.asarray(values, dtype=float) * np.asarray(std).reshape(-1)[0]) + np.asarray(mean).reshape(-1)[0]


def make_future_feature_frame(history_df, horizon, lookback_days=14):
    """Create future feature assumptions from recent history."""
    history = history_df.copy()
    history["date"] = pd.to_datetime(history["date"])
    recent = history.tail(lookback_days).reset_index(drop=True)

    last_date = history["date"].max()
    last_day_index = int(history["day_index"].max())

    if len(recent) > 1:
        active_daily_change = (
            recent["active_accounts"].iloc[-1] - recent["active_accounts"].iloc[0]
        ) / (len(recent) - 1)
    else:
        active_daily_change = 0

    rows = []
    for step in range(1, horizon + 1):
        pattern_row = recent.iloc[(step - 1) % len(recent)]
        rows.append({
            "date": last_date + pd.Timedelta(days=step),
            "day_index": last_day_index + step,
            "active_accounts": history["active_accounts"].iloc[-1] + active_daily_change * step,
            "usage_hours": float(pattern_row["usage_hours"]),
            "support_tickets": float(pattern_row["support_tickets"]),
            "release_flag": 0,
            "incident_flag": 0,
            "promo_flag": 0,
            "demand_score": np.nan,
        })

    return pd.DataFrame(rows)


def build_dense_feature_row(working_history, feature_columns, lag_days):
    """Build one Dense NN feature row from the latest available history."""
    row = {}
    for lag in range(1, lag_days + 1):
        row[f"demand_lag_{lag}"] = working_history["demand_score"].iloc[-1 - lag]
        row[f"usage_lag_{lag}"] = working_history["usage_hours"].iloc[-1 - lag]

    for column in ["release_flag", "incident_flag", "promo_flag", "support_tickets"]:
        row[column] = working_history[column].iloc[-1]

    return np.array([[row[column] for column in feature_columns]], dtype=float)


def forecast_future_dense(
    model,
    history_df,
    feature_columns,
    lag_days,
    horizon,
    feature_mean,
    feature_std,
    target_mean,
    target_std,
):
    """Forecast future demand recursively with the trained Dense NN."""
    working_history = history_df.copy()
    working_history["date"] = pd.to_datetime(working_history["date"])
    future_frame = make_future_feature_frame(working_history, horizon)
    forecasts = []

    model.eval()
    for step in range(horizon):
        X_raw = build_dense_feature_row(working_history, feature_columns, lag_days)
        X_scaled = _apply_standardizer(X_raw, feature_mean, feature_std)
        X_future = torch.tensor(X_scaled, dtype=torch.float32)

        with torch.no_grad():
            scaled_prediction = model(X_future).detach().numpy().reshape(-1)

        demand_prediction = float(_inverse_standardizer(scaled_prediction, target_mean, target_std)[0])
        next_row = future_frame.iloc[step].copy()
        next_row["demand_score"] = demand_prediction

        forecasts.append({
            "date": next_row["date"],
            "forecast_demand_score": demand_prediction,
            "horizon_day": step + 1,
        })
        working_history = pd.concat([working_history, pd.DataFrame([next_row])], ignore_index=True)

    return pd.DataFrame(forecasts)


def forecast_future_sequence(
    model,
    history_df,
    feature_columns,
    horizon,
    window_size,
    feature_mean,
    feature_std,
    target_mean,
    target_std,
):
    """Forecast future demand recursively with a trained RNN or LSTM."""
    working_history = history_df.copy()
    working_history["date"] = pd.to_datetime(working_history["date"])
    future_frame = make_future_feature_frame(working_history, horizon)
    forecasts = []

    model.eval()
    for step in range(horizon):
        model_window = working_history.tail(window_size).copy()
        scaled_window = _apply_standardizer(
            model_window[feature_columns].to_numpy(dtype=float),
            feature_mean,
            feature_std,
        )
        X_future = torch.tensor(
            scaled_window.reshape(1, window_size, len(feature_columns)),
            dtype=torch.float32,
        )

        with torch.no_grad():
            scaled_prediction = model(X_future).detach().numpy().reshape(-1)

        demand_prediction = float(_inverse_standardizer(scaled_prediction, target_mean, target_std)[0])
        next_row = future_frame.iloc[step].copy()
        next_row["demand_score"] = demand_prediction

        forecasts.append({
            "date": next_row["date"],
            "forecast_demand_score": demand_prediction,
            "horizon_day": step + 1,
        })
        working_history = pd.concat([working_history, pd.DataFrame([next_row])], ignore_index=True)

    return pd.DataFrame(forecasts)


def plot_forward_forecast(history_df, future_forecast, title, history_days=45):
    recent_history = history_df.copy().tail(history_days)
    recent_history["date"] = pd.to_datetime(recent_history["date"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recent_history["date"],
        y=recent_history["demand_score"],
        mode="lines+markers",
        name="Recent actual",
    ))
    fig.add_trace(go.Scatter(
        x=future_forecast["date"],
        y=future_forecast["forecast_demand_score"],
        mode="lines+markers",
        name="Forward forecast",
    ))
    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="Demand score",
    )
    fig.show()


def plot_multiple_forward_forecasts(history_df, forecast_frames, title, history_days=45):
    recent_history = history_df.copy().tail(history_days)
    recent_history["date"] = pd.to_datetime(recent_history["date"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recent_history["date"],
        y=recent_history["demand_score"],
        mode="lines+markers",
        name="Recent actual",
    ))

    for model_name, forecast_frame in forecast_frames.items():
        fig.add_trace(go.Scatter(
            x=forecast_frame["date"],
            y=forecast_frame["forecast_demand_score"],
            mode="lines+markers",
            name=f"{model_name} forward forecast",
        ))

    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="Demand score",
    )
    fig.show()
