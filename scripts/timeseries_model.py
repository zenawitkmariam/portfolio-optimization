import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Modern TensorFlow and Keras 3 Integration (2026 Standard)
import tensorflow as tf
import keras
from keras import layers, models, Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.optimizers import Adam

def prepare_data(df, asset_name='TSLA'):
    """Filters data and performs a chronological split for time series."""
    df.index = pd.to_datetime(df['Date'])
    asset_df = df[df['Asset'] == asset_name][['Adj Close']].copy()
    
    # Preserving temporal order: Train (2015-2024), Test (2025-2026)
    train = asset_df[:'2024-12-31']
    test = asset_df['2025-01-01':]
    return train, test

def run_arima(train, test):
    """Fits an ARIMA model using automated parameter selection."""
    model_auto = auto_arima(train, seasonal=False, stepwise=True, suppress_warnings=True)
    order = model_auto.order
    
    model = SARIMAX(train, order=order).fit(disp=False)
    forecast = model.get_forecast(steps=len(test))
    return forecast.predicted_mean, order

def run_lstm(train, test, window_size=60):
    """LSTM implementation using Keras 3 multi-backend syntax."""
    # Scale data for deep learning stability
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train)
    scaled_test = scaler.transform(test)
    
    def create_sequences(data, window):
        X, y = [], []
        for i in range(window, len(data)):
            X.append(data[i-window:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(scaled_train, window_size)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    # Architecture for Task 2: Multi-layer LSTM
    model = Sequential([
        Input(shape=(window_size, 1)),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

    # Prepare inputs for forecasting the test period
    inputs = np.concatenate((scaled_train[-window_size:], scaled_test), axis=0)
    X_test, _ = create_sequences(inputs, window_size)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    predictions = model.predict(X_test)
    return scaler.inverse_transform(predictions)

def evaluate_performance(actual, predicted):
    """Calculates MAE, RMSE, and MAPE metrics."""
    # Convert 'actual' to numpy array if it is a Series/DataFrame
    if hasattr(actual, 'values'):
        actual_vals = actual.values.flatten()
    else:
        actual_vals = actual.flatten()
        
    # Convert 'predicted' to numpy array if it is a Series
    # This is where your error is happening
    if hasattr(predicted, 'values'):
        pred_vals = predicted.values.flatten()
    else:
        pred_vals = predicted.flatten()
    
    mae = mean_absolute_error(actual_vals, pred_vals)
    rmse = np.sqrt(mean_squared_error(actual_vals, pred_vals))
    mape = np.mean(np.abs((actual_vals - pred_vals) / actual_vals)) * 100
    
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}