"""
PrediCO2.py
CO2 Forecasting using Time Series Models and Deep Learning

Models:
- ARIMA
- SARIMA
- LSTM
- Transformer
"""

# =========================
# Imports
# =========================
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input, LayerNormalization, MultiHeadAttention, Dropout

# =========================
# Utility functions
# =========================
def compute_metrics(y_true, y_pred):
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }


def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


# =========================
# Data loading
# =========================
def load_data(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df


# =========================
# Preprocessing
# =========================
def preprocess_data(series):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))
    return scaled, scaler


# =========================
# ARIMA
# =========================
def train_arima(train, test):
    start = time.time()
    model = ARIMA(train, order=(5, 1, 0))
    fitted = model.fit()
    train_time = time.time() - start

    start = time.time()
    preds = fitted.forecast(len(test))
    infer_time = time.time() - start

    metrics = compute_metrics(test, preds)
    params = fitted.params.shape[0]

    return preds, metrics, train_time, infer_time, params


# =========================
# SARIMA
# =========================
def train_sarima(train, test):
    start = time.time()
    model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12))
    fitted = model.fit(disp=False)
    train_time = time.time() - start

    start = time.time()
    preds = fitted.forecast(len(test))
    infer_time = time.time() - start

    metrics = compute_metrics(test, preds)
    params = fitted.params.shape[0]

    return preds, metrics, train_time, infer_time, params


# =========================
# LSTM
# =========================
def train_lstm(train, test, seq_len=12):
    X_train, y_train = create_sequences(train, seq_len)
    X_test, y_test = create_sequences(test, seq_len)

    model = Sequential([
        LSTM(50, activation='tanh', input_shape=(seq_len, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    start = time.time()
    model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)
    train_time = time.time() - start

    start = time.time()
    preds = model.predict(X_test, verbose=0)
    infer_time = time.time() - start

    metrics = compute_metrics(y_test, preds)
    params = model.count_params()

    return preds, metrics, train_time, infer_time, params


# =========================
# Transformer
# =========================
def transformer_model(seq_len):
    inputs = Input(shape=(seq_len, 1))
    x = MultiHeadAttention(num_heads=2, key_dim=16)(inputs, inputs)
    x = LayerNormalization()(x)
    x = Dropout(0.1)(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model


def train_transformer(train, test, seq_len=12):
    X_train, y_train = create_sequences(train, seq_len)
    X_test, y_test = create_sequences(test, seq_len)

    model = transformer_model(seq_len)

    start = time.time()
    model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)
    train_time = time.time() - start

    start = time.time()
    preds = model.predict(X_test, verbose=0)
    infer_time = time.time() - start

    metrics = compute_metrics(y_test, preds)
    params = model.count_params()

    return preds, metrics, train_time, infer_time, params


# =========================
# Main
# =========================
if __name__ == "__main__":
    df = load_data("data/co2_dataset.csv")
    series = df['co2']

    scaled, scaler = preprocess_data(series)

    split = int(len(scaled) * 0.8)
    train, test = scaled[:split], scaled[split:]

    results = {}

    _, results["ARIMA"], *_ = train_arima(series[:split], series[split:])
    _, results["SARIMA"], *_ = train_sarima(series[:split], series[split:])
    _, results["LSTM"], *_ = train_lstm(train, test)
    _, results["Transformer"], *_ = train_transformer(train, test)

    print("\nModel comparison:")
    for model, metrics in results.items():
        print(model, metrics)
