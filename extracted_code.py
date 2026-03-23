
# --- Cell 0 ---
import pandas as pd

import yfinance as yf


data = yf.download(["MSFT","AMZN", "TSLA", "AAPL"], start="2020-08-25", end="2025-08-25")
data

# --- Cell 1 ---
data.head()

# --- Cell 2 ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_datareader as data
import datetime as dt

plt.style.use('fivethirtyeight')
%matplotlib inline

# --- Cell 3 ---
import yfinance as yf

data = yf.download(["MSFT","AMZN", "TSLA", "AAPL"], start="2020-08-25", end="2025-08-25")
data

# --- Cell 4 ---
data.head()

# --- Cell 5 ---
data.tail()

# --- Cell 6 ---
data.shape

# --- Cell 7 ---
data.info()

# --- Cell 8 ---
data.isnull().sum()

# --- Cell 9 ---
data.describe()

# --- Cell 10 ---
data = data.reset_index()

# --- Cell 11 ---
data.head()

# --- Cell 12 ---
df = data.to_csv("stocks_data.csv")

# If you want to download it directly to your computer from Colab
from google.colab import files
files.download("stocks_data.csv")

# --- Cell 13 ---
df = pd.read_csv('stocks_data.csv')

# --- Cell 14 ---
df.head()

# --- Cell 15 ---
#Candlesticks
import plotly.graph_objects as go

fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                                     open=df['Open'],
                                     high=df['High'],
                                     low=df['Low'],
                                     close=df['Close'])])

fig.update_layout(title='Power Grid Stock Price',
                  xaxis_rangeslider_visible=False)
fig.show()

# --- Cell 16 ---
# Plot closing price for each stock
plt.figure(figsize=(12, 6))

for stock in ["MSFT","AMZN", "TSLA", "AAPL"]:
    plt.plot(data.index, data["Close"][stock], label=f'{stock} Closing Price', linewidth=2)

plt.title("Stock Closing Prices (2020–2025)")
plt.xlabel("Date")
plt.ylabel("Closing Price (USD)")
plt.legend()
plt.show()

# --- Cell 17 ---
# Plot opening price for each stock
plt.figure(figsize=(12, 6))

for stock in ["MSFT","AMZN", "TSLA", "AAPL"]:
    plt.plot(data.index, data["Open"][stock], label=f'{stock} Opening Price', linewidth=2)

plt.title("Stock Opening Prices (2020–2025)")
plt.xlabel("Date")
plt.ylabel("Opening Price (USD)")
plt.legend()
plt.show()

# --- Cell 18 ---
# Plot high price for each stock
plt.figure(figsize=(12, 6))

for stock in ["MSFT","AMZN", "TSLA", "AAPL"]:
    plt.plot(data.index, data["High"][stock], label=f'{stock} High Price', linewidth=2)

plt.title("Stock High Prices (2020–2025)")
plt.xlabel("Date")
plt.ylabel("High Price (USD)")
plt.legend()
plt.show()

# --- Cell 19 ---
# Plot trading volume for each stock
plt.figure(figsize=(12, 6))

for stock in ["MSFT","AMZN", "TSLA", "AAPL"]:
    plt.plot(data.index, data["Volume"][stock], label=f'{stock} Volume', linewidth=2)

plt.title("Stock Trading Volume (2020–2025)")
plt.xlabel("Date")
plt.ylabel("Volume")
plt.legend()
plt.show()


# --- Cell 20 ---
# Moving Average
df1 = pd.DataFrame(data)

df1.head()

# --- Cell 21 ---
# Plot moving averages for each stock
for stock in ["MSFT", "AMZN", "TSLA", "AAPL"]:
    close_prices = data['Close'][stock]

    # Calculate moving averages (50-day and 200-day)
    ma50 = close_prices.rolling(window=50).mean()
    ma200 = close_prices.rolling(window=200).mean()

    # Plot
    plt.figure(figsize=(12,6))
    plt.plot(close_prices.index, close_prices, label=f"{stock} Closing Price", linewidth=2)
    plt.plot(ma50.index, ma50, label="50-Day MA", linewidth=2)
    plt.plot(ma200.index, ma200, label="200-Day MA", linewidth=2)

    plt.title(f"{stock} Stock Price with Moving Averages (2020–2025)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.show()

# --- Cell 22 ---

import numpy as np
from sklearn.preprocessing import MinMaxScaler



# Dictionary to hold training datasets for each stock
datasets = {}

# Loop through each stock
for stock in ["MSFT", "AMZN", "TSLA", "AAPL"]:
    print(f"\nProcessing {stock}...")

    # Extract closing prices
    close_prices = data['Close'][stock].values.reshape(-1, 1)

    # Normalize
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    # Training data length (80%)
    training_data_len = int(np.ceil(len(scaled_data) * 0.8))
    train_data = scaled_data[0:training_data_len]

    # Create x_train, y_train
    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    # Convert to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape input to [samples, timesteps, features]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Save results in dictionary
    datasets[stock] = {
        "x_train": x_train,
        "y_train": y_train,
        "scaler": scaler
    }

    print(f"{stock} → x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")

# --- Cell 23 ---

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Dictionaries to hold models, datasets, and histories
models = {}
datasets = {}
histories = {}

# Loop through each stock
for stock in ["MSFT", "AMZN", "TSLA", "AAPL"]:
    print(f"\nPreparing data for {stock}...")

    # Extract closing prices
    close_prices = data['Close'][stock].values.reshape(-1, 1)

    # Normalize
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    # Training data length (80%)
    training_data_len = int(np.ceil(len(scaled_data) * 0.8))
    train_data = scaled_data[0:training_data_len]

    # Create x_train, y_train
    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    # Convert to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape input to [samples, timesteps, features]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Save dataset
    datasets[stock] = {
        "x_train": x_train,
        "y_train": y_train,
        "scaler": scaler
    }

    print(f"{stock} → x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")

    # ------------------ Build LSTM Model ------------------ #
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # ------------------ Train the Model ------------------ #
    print(f"\nTraining model for {stock}...")
    history = model.fit(
        x_train, y_train,
        batch_size=32,
        epochs=10,   # increase if you want better accuracy
        verbose=1
    )

    # Save model and history
    models[stock] = model
    histories[stock] = history

    print(f"Finished training {stock} ✅")

# --- Cell 24 ---
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

models, datasets, histories = {}, {}, {}

for stock in ["MSFT","AMZN","TSLA","AAPL"]:
    print(f"\nPreparing data for {stock}...")

    # Extract closing prices
    close_prices = data['Close'][stock].values.reshape(-1, 1)

    # Normalize
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(close_prices)

    # Training size (80%)
    training_data_len = int(np.ceil(len(scaled_data) * 0.8))
    train_data = scaled_data[:training_data_len]

    # Prepare x_train, y_train
    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    datasets[stock] = {"x_train": x_train, "y_train": y_train, "scaler": scaler}

    print(f"{stock} → x_train: {x_train.shape}, y_train: {y_train.shape}")

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mean_squared_error")

    # Train model
    print(f"Training {stock}...")
    history = model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1)

    models[stock] = model
    histories[stock] = history

# 📉 Plot training loss
plt.figure(figsize=(12,6))
for stock, history in histories.items():
    plt.plot(history.history['loss'], label=f'{stock} Loss')
plt.title("Training Loss per Stock (Price Prediction)")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.legend()
plt.show()

# --- Cell 25 ---
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt


models, datasets, histories = {}, {}, {}

for stock in ["MSFT","AMZN","TSLA","AAPL"]:
    print(f"\nPreparing data for {stock}...")

    close_prices = data['Close'][stock].values.reshape(-1,1)

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(close_prices)

    training_data_len = int(np.ceil(len(scaled_data) * 0.8))
    train_data = scaled_data[:training_data_len]

    # Create features and labels
    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        # Label = 1 if next day > current day, else 0
        y_train.append(1 if train_data[i, 0] > train_data[i-1, 0] else 0)

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    datasets[stock] = {"x_train": x_train, "y_train": y_train, "scaler": scaler}

    print(f"{stock} → x_train: {x_train.shape}, y_train: {y_train.shape} (binary)")

    # Build classification LSTM
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1, activation="sigmoid"))  # sigmoid for binary classification

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    print(f"Training {stock}...")
    history = model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1)

    models[stock] = model
    histories[stock] = history

# 📊 Plot accuracy curves
plt.figure(figsize=(12,6))
for stock, history in histories.items():
    plt.plot(history.history['accuracy'], label=f'{stock} Accuracy')
plt.title("Training Accuracy per Stock (Direction Prediction)")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# --- Cell 26 ---
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Dictionaries
models, datasets, histories = {}, {}, {}

for stock in ["MSFT","AMZN","TSLA","AAPL"]:
    print(f"\nPreparing data for {stock}...")

    # Extract closing prices
    close_prices = data['Close'][stock].values.reshape(-1, 1)

    # Normalize
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(close_prices)

    # Training size (80%)
    training_data_len = int(np.ceil(len(scaled_data) * 0.8))
    train_data = scaled_data[:training_data_len]

    # Prepare x_train, y_train (gain/loss classification)
    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        # gain = 1 if today's price > yesterday's price else 0
        y_train.append(1 if train_data[i, 0] > train_data[i-1, 0] else 0)

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    datasets[stock] = {"x_train": x_train, "y_train": y_train, "scaler": scaler}

    print(f"{stock} → x_train: {x_train.shape}, y_train: {y_train.shape}, Gains: {sum(y_train)}")

    # Build Classification LSTM
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))  # classification output

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Train model
    print(f"Training {stock}...")
    history = model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1)

    models[stock] = model
    histories[stock] = history

# 📉 Plot training accuracy
plt.figure(figsize=(12,6))
for stock, history in histories.items():
    plt.plot(history.history['accuracy'], label=f'{stock} Accuracy')
plt.title("Training Accuracy per Stock (Gain Prediction)")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# --- Cell 27 ---
import math
from sklearn.metrics import mean_squared_error

for stock in ["MSFT","AMZN","TSLA","AAPL"]:
    print(f"\n🔎 Evaluating {stock}...")

    close_prices = data['Close'][stock].values.reshape(-1,1)
    scaler = datasets[stock]["scaler"]

    scaled_data = scaler.transform(close_prices)
    training_data_len = int(np.ceil(len(scaled_data) * 0.8))

    # Test data
    test_data = scaled_data[training_data_len-60: , :]

    x_test, y_test = [], close_prices[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Predictions
    predictions = models[stock].predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # RMSE
    rmse = math.sqrt(mean_squared_error(y_test, predictions))
    print(f"{stock} RMSE: {rmse:.2f}")

    # Plot actual vs predicted
    plt.figure(figsize=(12,6))
    plt.plot(y_test, label="Actual Price")
    plt.plot(predictions, label="Predicted Price")
    plt.title(f"{stock} Actual vs Predicted Prices")
    plt.xlabel("Time")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.show()

# --- Cell 28 ---
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout



# Dictionaries
models, datasets, histories = {}, {}, {}

for stock in ["MSFT","AMZN","TSLA","AAPL"]:
    print(f"\nPreparing data for {stock}...")

    # Extract closing prices
    close_prices = data['Close'][stock].values.reshape(-1, 1)

    # Normalize
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(close_prices)

    # Train-test split (80%-20%)
    training_data_len = int(np.ceil(len(scaled_data) * 0.8))
    train_data = scaled_data[:training_data_len]
    test_data  = scaled_data[training_data_len-60:]  # keep 60 lag for test

    # ----------------------------
    # Training Data
    # ----------------------------
    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(1 if train_data[i, 0] > train_data[i-1, 0] else 0)

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # ----------------------------
    # Testing Data
    # ----------------------------
    x_test, y_test = [], []
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
        y_test.append(1 if test_data[i, 0] > test_data[i-1, 0] else 0)

    x_test, y_test = np.array(x_test), np.array(y_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    datasets[stock] = {
        "x_train": x_train, "y_train": y_train,
        "x_test": x_test, "y_test": y_test,
        "scaler": scaler
    }

    print(f"{stock} → Train: {x_train.shape}, Test: {x_test.shape}, Gains Train: {sum(y_train)}")

    # ----------------------------
    # Build Classification LSTM
    # ----------------------------
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(x_train.shape[1],1)))
    model.add(Dropout(0.3))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))  # binary classification

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Train model with validation split
    print(f"Training {stock}...")
    history = model.fit(
        x_train, y_train,
        batch_size=32,
        epochs=15,
        validation_split=0.1,
        verbose=1
    )

    models[stock] = model
    histories[stock] = history

    # ----------------------------
    # Evaluate
    # ----------------------------
    print(f"\n🔎 Evaluating {stock}...")
    y_pred = (model.predict(x_test) > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=1)

    print(f"{stock} Accuracy: {acc*100:.2f}%")
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)

# ----------------------------
# Plot Training Accuracy
# ----------------------------
plt.figure(figsize=(12,6))
for stock, history in histories.items():
    plt.plot(history.history['accuracy'], label=f'{stock} Train Acc')
    plt.plot(history.history['val_accuracy'], linestyle="--", label=f'{stock} Val Acc')
plt.title("Training vs Validation Accuracy per Stock (Gain Prediction)")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# --- Cell 29 ---
# ----------------------------
# 📌 Prepare Train/Test Datasets for Each Stock
# ----------------------------
datasets = {}

for stock in ["MSFT","AMZN","TSLA","AAPL"]:
    print(f"\nPreparing data for {stock}...")

    # Closing prices
    close_prices = data['Close'][stock].values.reshape(-1, 1)

    # Normalize
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(close_prices)

    # Training length
    training_data_len = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:training_data_len]

    # ----------------------------
    # Training Data
    # ----------------------------
    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(1 if train_data[i,0] > train_data[i-1,0] else 0)  # Gain/Loss classification

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # ----------------------------
    # Test Data
    # ----------------------------
    test_data = scaled_data[training_data_len-60:]
    x_test, y_test = [], []
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
        y_test.append(1 if test_data[i,0] > test_data[i-1,0] else 0)

    x_test, y_test = np.array(x_test), np.array(y_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    datasets[stock] = {
        "x_train": x_train, "y_train": y_train,
        "x_test": x_test, "y_test": y_test,
        "scaler": scaler
    }

    print(f"{stock} → Train: {x_train.shape}, Test: {x_test.shape}")

# --- Cell 30 ---
# 📌 Train, Predict & Plot for Each Stock
# ----------------------------
for stock in ["MSFT","AMZN","TSLA","AAPL"]:
    print(f"\n🔎 Processing {stock}...")

    # Closing prices
    close_prices = data['Close'][stock].values.reshape(-1, 1)

    # Normalize
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(close_prices)

    # Training length
    training_data_len = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:training_data_len]

    # ----------------------------
    # Training Data
    # ----------------------------
    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # ----------------------------
    # Test Data
    # ----------------------------
    test_data = scaled_data[training_data_len-60:]
    x_test, y_test = [], close_prices[training_data_len:]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # ----------------------------
    # LSTM Model
    # ----------------------------
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mean_squared_error")

    # Train model
    model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=1)

    # ----------------------------
    # Predictions
    # ----------------------------
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # ----------------------------
    # Plot Results
    # ----------------------------
    plt.figure(figsize=(12,6))
    plt.plot(y_test, color='blue', label=f'Actual {stock} Price')
    plt.plot(predictions, color='red', label=f'Predicted {stock} Price')
    plt.title(f'{stock} Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()


# --- Cell 31 ---
# ----------------------------
# 📌 Predict Next Day Price for Each Stock
# ----------------------------
for stock in ["MSFT","AMZN","TSLA","AAPL"]:
    print(f"\n🔮 Predicting next day price for {stock}...")

    # Get closing prices
    close_prices = data['Close'][stock].values.reshape(-1, 1)

    # Normalize
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(close_prices)

    # Prepare training data
    training_data_len = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:training_data_len]

    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build & train model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=0)

    # Get last 60 days
    last_60_days = scaled_data[-60:]
    last_60_days = np.reshape(last_60_days, (1, last_60_days.shape[0], 1))

    # Predict next day
    next_day_price = model.predict(last_60_days)
    next_day_price = scaler.inverse_transform(next_day_price)

    print(f"📈 Next Day Predicted Price of {stock}: {next_day_price[0][0]:.2f} USD")

# --- Cell 32 ---


# --- Cell 33 ---
# Step 1: Import libraries
import pandas as pd
import yfinance as yf
from google.colab import files

# Step 2: Download stock data
data = yf.download(["MSFT","AMZN","TSLA","AAPL"], start="2020-08-25", end="2025-08-25")

# Step 3: Reset index (so Date becomes a column)
data = data.reset_index()

# Step 4: Save to CSV
data.to_csv("stocks_data.csv", index=False)

# Step 5: Download CSV to local computer
files.download("stocks_data.csv")

# --- Cell 34 ---

