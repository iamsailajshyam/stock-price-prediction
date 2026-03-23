import yfinance as yf # type: ignore
import numpy as np # type: ignore
import os
import joblib # type: ignore
from sklearn.preprocessing import MinMaxScaler # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

TICKERS = ["AAPL", "MSFT", "AMZN", "TSLA", "GOOGL", "META", "NVDA", "NFLX"]
LOOKBACK_STEPS = 60
MODEL_DIR = "models"

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def train_and_save():
    print("Initiating ADVANCED Bidirectional LSTM training pipeline...")
    # Expanded dataset window to 10 years for deeper historical context
    data = yf.download(TICKERS, start="2015-01-01", end="2025-01-01")
    
    for stock in TICKERS:
        print(f"\n[+] Processing {stock}...")
        try:
            close_prices = data['Close'][stock].values.reshape(-1, 1)
            
            # 1. Normalize
            scaler = MinMaxScaler(feature_range=(0,1))
            scaled_data = scaler.fit_transform(close_prices)
            
            # 2. Prepare Windowed Data
            x_train, y_train = [], []
            for i in range(LOOKBACK_STEPS, len(scaled_data)):
                x_train.append(scaled_data[i-LOOKBACK_STEPS:i, 0])
                y_train.append(scaled_data[i, 0])
                
            x_train, y_train = np.array(x_train), np.array(y_train)
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
            
            print(f"    - Sequences built. x_train: {x_train.shape}")
            
            # 3. Build Advanced Bidirectional LSTM Architecture
            model = Sequential()
            # Wrap the first LSTM in a Bidirectional layer to learn forwards & backwards sequences
            model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(x_train.shape[1], 1)))
            model.add(Dropout(0.3))
            # Second layer
            model.add(Bidirectional(LSTM(64, return_sequences=False)))
            model.add(Dropout(0.3))
            
            # Deep dense network for complex feature mapping before the final output
            model.add(Dense(32, activation='relu'))
            model.add(Dense(16, activation='relu'))
            model.add(Dense(1)) # Predict next day close price
            
            model.compile(optimizer="adam", loss="mean_squared_error")
            
            # 4. Train Model with Early Stopping!
            print(f"    - Training Deep Model ({stock})...")
            
            # This hook prevents overfitting by instantly halting training if the validation loss stops dropping
            early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
            
            # Upgrading to 100 Epochs (from 5), using a 10% validation split holding tank
            model.fit(
                x_train, y_train,
                batch_size=64,
                epochs=100,
                validation_split=0.1,
                callbacks=[early_stop],
                verbose=1
            )
            
            # 5. Save Artifacts for FastAPI
            model_path = os.path.join(MODEL_DIR, f"{stock.lower()}_lstm.h5")
            scaler_path = os.path.join(MODEL_DIR, f"{stock.lower()}_scaler.pkl")
            
            model.save(model_path)
            joblib.dump(scaler, scaler_path)
            
            print(f"    - Saved effectively to {model_path} ✅")
            
        except Exception as e:
            print(f"    - Failed to train {stock}: {e}")

if __name__ == "__main__":
    train_and_save()
    print("\nAll models trained and exported! You can now start the FastAPI server.")
