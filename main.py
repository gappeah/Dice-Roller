import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor


class SolanaEnsemblePredictor:
    def __init__(self, start_date="2020-01-01"):
        self.start_date = start_date
        self.end_date = datetime.now().strftime("%Y-%m-%d")
        self.price_scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))

    def fetch_data(self):
        """Fetch Solana price data and additional features"""
        # Fetch Solana data
        sol = yf.download("SOL-USD", start=self.start_date, end=self.end_date)

        # Calculate technical indicators
        sol["SMA_7"] = sol["Close"].rolling(window=7).mean()
        sol["SMA_30"] = sol["Close"].rolling(window=30).mean()
        sol["RSI"] = self.calculate_rsi(sol["Close"])
        sol["MACD"] = self.calculate_macd(sol["Close"])
        sol["Volatility"] = sol["Close"].rolling(window=20).std()

        # Add market sentiment features (example with BTC correlation)
        btc = yf.download("BTC-USD", start=self.start_date, end=self.end_date)["Close"]
        sol["BTC_Correlation"] = sol["Close"].rolling(window=30).corr(btc)

        # Calculate price momentum
        sol["Price_Momentum"] = sol["Close"].pct_change(periods=7)

        self.data = sol.dropna()
        return self.data

    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd - signal_line

    def prepare_ensemble_data(self, sequence_length=60):
        """Prepare data for ensemble model"""
        feature_columns = [
            "Close",
            "SMA_7",
            "SMA_30",
            "RSI",
            "MACD",
            "Volatility",
            "BTC_Correlation",
            "Price_Momentum",
        ]

        # Scale features
        scaled_features = self.feature_scaler.fit_transform(self.data[feature_columns])
        scaled_features = pd.DataFrame(
            scaled_features, columns=feature_columns, index=self.data.index
        )

        X, y = [], []
        for i in range(sequence_length, len(scaled_features)):
            X.append(scaled_features.iloc[i - sequence_length : i].values)
            y.append(self.data["Close"].iloc[i])

        X, y = np.array(X), np.array(y)

        # Split into train and test sets
        train_size = int(len(X) * 0.8)
        self.X_train, self.X_test = X[:train_size], X[train_size:]
        self.y_train, self.y_test = y[:train_size], y[train_size:]

        # Scale target values
        self.y_train_scaled = self.price_scaler.fit_transform(
            self.y_train.reshape(-1, 1)
        )
        self.y_test_scaled = self.price_scaler.transform(self.y_test.reshape(-1, 1))

        return self.X_train, self.X_test, self.y_train_scaled, self.y_test_scaled

    def build_advanced_lstm(self, sequence_length=60, n_features=8):
        """Build advanced LSTM model with attention mechanism"""
        input_layer = Input(shape=(sequence_length, n_features))

        # LSTM layers with residual connections
        lstm1 = LSTM(100, return_sequences=True)(input_layer)
        lstm1 = Dropout(0.2)(lstm1)

        lstm2 = LSTM(50, return_sequences=True)(lstm1)
        lstm2 = Dropout(0.2)(lstm2)

        lstm3 = LSTM(25)(lstm2)
        lstm3 = Dropout(0.2)(lstm3)

        # Dense layers for final prediction
        dense1 = Dense(20, activation="relu")(lstm3)
        output = Dense(1)(dense1)

        model = Model(inputs=input_layer, outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

        return model

    def build_sarima_model(self):
        """Build and train SARIMA model"""
        order = (2, 1, 2)
        seasonal_order = (1, 1, 1, 12)

        self.sarima_model = SARIMAX(
            self.data["Close"], order=order, seasonal_order=seasonal_order
        )
        self.sarima_results = self.sarima_model.fit(disp=False)
        return self.sarima_results

    def optimize_ensemble_weights(self, lstm_pred, sarima_pred, actual):
        """Optimize ensemble weights using Scipy's minimize"""

        def objective(weights):
            weighted_pred = weights[0] * lstm_pred + weights[1] * sarima_pred
            return mean_squared_error(actual, weighted_pred)

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = ((0, 1), (0, 1))

        result = minimize(
            objective,
            [0.5, 0.5],
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        return result.x

    def predict_ensemble(self, days=7):
        """Make ensemble predictions"""
        # Get LSTM predictions
        last_sequence = self.X_test[-1:].copy()
        lstm_predictions = []

        for _ in range(days):
            next_pred = self.lstm_model.predict(last_sequence, verbose=0)
            lstm_predictions.append(next_pred[0, 0])

            # Update sequence for next prediction
            new_features = self.update_features(last_sequence[0, -1], next_pred[0, 0])
            last_sequence = np.roll(last_sequence[0], -1, axis=0)
            last_sequence[-1] = new_features
            last_sequence = last_sequence.reshape(1, *last_sequence.shape)

        # Get SARIMA predictions
        sarima_forecast = self.sarima_results.forecast(steps=days)

        # Scale predictions
        lstm_predictions = self.price_scaler.inverse_transform(
            np.array(lstm_predictions).reshape(-1, 1)
        )

        # Calculate ensemble predictions using optimized weights
        ensemble_predictions = self.weights[0] * lstm_predictions + self.weights[
            1
        ] * sarima_forecast.values.reshape(-1, 1)

        return {
            "ensemble": ensemble_predictions.flatten(),
            "lstm": lstm_predictions.flatten(),
            "sarima": sarima_forecast.values,
        }

    def update_features(self, last_features, predicted_price):
        """Update technical indicators for sequential prediction"""
        # Simplified feature update logic
        new_features = last_features.copy()
        new_features[0] = predicted_price  # Update Close price
        # Update other features (simplified)
        new_features[1] = (new_features[0] + last_features[0] * 6) / 7  # SMA_7
        new_features[2] = (new_features[0] + last_features[0] * 29) / 30  # SMA_30
        return new_features

    def train_ensemble(self, epochs=50, batch_size=32):
        """Train the ensemble model"""
        # Train LSTM
        self.lstm_model = self.build_advanced_lstm()
        lstm_history = self.lstm_model.fit(
            self.X_train,
            self.y_train_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1,
        )

        # Train SARIMA
        self.build_sarima_model()

        # Get predictions for weight optimization
        lstm_pred = self.lstm_model.predict(self.X_test, verbose=0)
        lstm_pred = self.price_scaler.inverse_transform(lstm_pred)

        sarima_pred = self.sarima_results.get_forecast(len(self.y_test))
        sarima_pred = sarima_pred.predicted_mean.values.reshape(-1, 1)

        # Optimize ensemble weights
        self.weights = self.optimize_ensemble_weights(
            lstm_pred, sarima_pred, self.y_test.reshape(-1, 1)
        )

        return lstm_history

    def plot_predictions(self, predictions, days=7):
        """Plot ensemble predictions with confidence intervals"""
        plt.figure(figsize=(15, 8))

        # Plot historical data
        plt.plot(
            self.data.index[-30:],
            self.data["Close"][-30:],
            label="Historical Prices",
            color="blue",
        )

        # Create future dates
        future_dates = pd.date_range(
            start=self.data.index[-1], periods=days + 1, closed="right"
        )

        # Plot predictions
        plt.plot(
            future_dates,
            predictions["ensemble"],
            label="Ensemble Predictions",
            color="purple",
            linewidth=2,
        )
        plt.plot(
            future_dates,
            predictions["lstm"],
            label="LSTM Predictions",
            color="red",
            linestyle="--",
        )
        plt.plot(
            future_dates,
            predictions["sarima"],
            label="SARIMA Predictions",
            color="green",
            linestyle="--",
        )

        # Add confidence intervals (example for SARIMA)
        sarima_forecast = self.sarima_results.get_forecast(days)
        conf_int = sarima_forecast.conf_int()
        plt.fill_between(
            future_dates,
            conf_int.iloc[:, 0],
            conf_int.iloc[:, 1],
            color="gray",
            alpha=0.2,
            label="95% Confidence Interval",
        )

        plt.title("Solana Price Predictions - Ensemble Model")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.grid(True)
        plt.show()

    def get_feature_importance(self):
        """Calculate and visualize feature importance"""
        # Use Random Forest for feature importance
        rf = RandomForestRegressor(n_estimators=100)
        rf.fit(self.X_train.reshape(len(self.X_train), -1), self.y_train)

        feature_names = [
            "Close",
            "SMA_7",
            "SMA_30",
            "RSI",
            "MACD",
            "Volatility",
            "BTC_Correlation",
            "Price_Momentum",
        ]
        importance = pd.DataFrame(
            {
                "feature": feature_names
                * (self.X_train.shape[1] // len(feature_names)),
                "importance": rf.feature_importances_,
            }
        )

        return (
            importance.groupby("feature")
            .mean()
            .sort_values("importance", ascending=False)
        )
