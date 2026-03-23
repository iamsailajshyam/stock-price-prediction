# NeuralStock 📈🧠

**NeuralStock** is an AI-powered stock market forecasting web application. It leverages deep learning (LSTM neural networks), advanced statistical modeling, and real-time market data to predict future stock trends, expected volatility, and analyze market sentiment.

## 🌟 Features

- **AI-Powered Predictions**: Uses trained LSTM (Long Short-Term Memory) models to forecast stock prices 7, 14, or 30 days into the future.
- **Real-Time Market Data**: Fetches the latest stock historical data and current prices using `yfinance`.
- **Technical Indicators**: Overlay Simple Moving Average (SMA), Relative Strength Index (RSI), and ARIMA baselines on the price charts.
- **Sentiment Analysis**: Provides a bullish/bearish NLP score based on recent news sentiment.
- **Interactive Dashboard**: Modern UI/UX with glassmorphism design, dark mode, and responsive, interactive charts using Chart.js.
- **Authentication System**: Built-in user registration and login functionality using FastAPI and SQLite.

## 🛠️ Technology Stack

**Frontend:**
- HTML5, CSS3 (Custom Glassdoor/Dark themes)
- Vanilla JavaScript
- Chart.js (for data visualization)

**Backend:**
- Python 3.x
- [FastAPI](https://fastapi.tiangolo.com/) (High-performance API framework)
- Uvicorn (ASGI server)

**Machine Learning & Data:**
- TensorFlow / Keras (LSTM Models)
- `scikit-learn` (Data scaling - MinMaxScaler)
- `yfinance` (Live market data fetching)
- Numpy, Pandas

## 🚀 Getting Started

### Prerequisites
Make sure you have Python 3.8+ installed on your machine.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/NeuralStock.git
   cd NeuralStock
   ```

2. **Set up the Backend Environment:**
   Navigate to the `backend` directory and install the required Python dependencies.
   ```bash
   cd backend
   pip install -r requirements.txt
   ```
   *(Ensure requirements include `fastapi`, `uvicorn`, `yfinance`, `tensorflow`, `scikit-learn`, `joblib`, `numpy`, and `pydantic`)*

3. **Run the FastAPI Server:**
   Start the backend server on `localhost:8080`.
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8080 --reload
   ```

4. **Launch the Frontend:**
   Simply open `index.html` in your modern web browser or serve it using a local static server.
   ```bash
   # from the project root
   npx serve .
   # or using Python's http.server
   python -m http.server 3000
   ```
   Then visit `http://localhost:3000` in your browser.

## 🤖 Model Architecture
The core prediction engine uses an LSTM (Long Short-Term Memory) network, specifically trained on historical time-series data of various tech stocks (AAPL, MSFT, AMZN, etc.).
- Each stock model `.h5` and its corresponding scaler `.pkl` is stored in `backend/models/`.
- The engine uses a rolling 60-day prediction window.
- **Fallback**: If the Deep Learning model is unavailable/unloaded, the backend seamlessly falls back to a mathematical stochastic simulation based on recent market volatility.

## 🔐 Authentication
The system securely manages user sessions using a basic email/password authentication flow stored locally in an SQLite database (`users.db`).

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](#).

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
