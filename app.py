import streamlit as st
import pandas as pd
import yfinance as yf
import ta

# Function to fetch stock indicators
def fetch_indicators(stock, interval='1d'):
    ticker = yf.Ticker(stock)
    data = ticker.history(period="1y", interval=interval)

    if data.empty or len(data) < 2:
        return {key: None for key in [
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
            'Upper_BB', 'Lower_BB', 'Volatility', 'Beta',
            'Close', 'Volume', 'SMA_50', 'SMA_200', 
            'EMA_12', 'EMA_26', 'Average_Volume', 
            'Average_Volume_10d', 'Pattern', 
            'Strength_Percentage', 'Bullish_Percentage', 'Bearish_Percentage'
        ]}

    # Calculate indicators
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
    macd = ta.trend.MACD(data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data['MACD_Hist'] = macd.macd_diff()
    bb = ta.volatility.BollingerBands(data['Close'], window=20, window_dev=2)
    data['Upper_BB'] = bb.bollinger_hband()
    data['Lower_BB'] = bb.bollinger_lband()
    data['Volatility'] = data['Close'].pct_change().rolling(window=21).std() * 100
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['EMA_12'] = ta.trend.EMAIndicator(data['Close'], window=12).ema_indicator()
    data['EMA_26'] = ta.trend.EMAIndicator(data['Close'], window=26).ema_indicator()
    
    average_volume = data['Volume'].mean()
    average_volume_10d = data['Volume'].rolling(window=10).mean().iloc[-1] if len(data['Volume']) >= 10 else None
    beta = ticker.info.get('beta', None)

    last_close = data['Close'].iloc[-1]
    pattern = detect_chart_pattern(data)

    return {
        'RSI': data['RSI'].iloc[-1],
        'MACD': data['MACD'].iloc[-1],
        'MACD_Signal': data['MACD_Signal'].iloc[-1],
        'MACD_Hist': data['MACD_Hist'].iloc[-1],
        'Upper_BB': data['Upper_BB'].iloc[-1],
        'Lower_BB': data['Lower_BB'].iloc[-1],
        'Volatility': data['Volatility'].iloc[-1],
        'Beta': beta,
        'Close': last_close,
        'Volume': data['Volume'].iloc[-1],
        'SMA_50': data['SMA_50'].iloc[-1],
        'SMA_200': data['SMA_200'].iloc[-1],
        'EMA_12': data['EMA_12'].iloc[-1],
        'EMA_26': data['EMA_26'].iloc[-1],
        'Average_Volume': average_volume,
        'Average_Volume_10d': average_volume_10d,
        'Pattern': pattern,
        'Strength_Percentage': ((last_close - data['SMA_50'].iloc[-1]) / data['SMA_50'].iloc[-1] * 100) if data['SMA_50'].iloc[-1] is not None else 0,
        'Bullish_Percentage': calculate_bullish_percentage(data),
        'Bearish_Percentage': calculate_bearish_percentage(data)
    }

# Function to detect chart patterns
def detect_chart_pattern(data):
    if len(data) < 30:  # Need at least 30 points to identify patterns
        return "No Pattern"

    recent_prices = data['Close'].tail(30).values
    patterns = {
        "Head and Shoulders": is_head_and_shoulders(recent_prices),
        "Double Top": is_double_top(recent_prices),
        "Double Bottom": is_double_bottom(recent_prices),
        "Symmetrical Triangle": is_symmetrical_triangle(recent_prices),
        "Ascending Triangle": is_ascending_triangle(recent_prices),
        "Descending Triangle": is_descending_triangle(recent_prices),
    }
    
    recognized_patterns = [(name, 'Daily') for name, detected in patterns.items() if detected]
    
    return recognized_patterns if recognized_patterns else ["No Recognized Pattern"]

# Head and Shoulders detection
def is_head_and_shoulders(prices):
    if len(prices) < 20:
        return False
    peaks = (prices[1:-1] > prices[:-2]) & (prices[1:-1] > prices[2:])
    valleys = (prices[1:-1] < prices[:-2]) & (prices[1:-1] < prices[2:])
    
    peak_indices = [i for i, p in enumerate(peaks, 1) if p]
    valley_indices = [i for i, v in enumerate(valleys, 1) if v]
    
    return len(peak_indices) >= 2 and len(valley_indices) >= 1

# Double Top detection
def is_double_top(prices):
    if len(prices) < 20:
        return False
    peaks = (prices[1:-1] > prices[:-2]) & (prices[1:-1] > prices[2:])
    peak_indices = [i for i, p in enumerate(peaks, 1) if p]
    
    return len(peak_indices) >= 2 and abs(prices[peak_indices[0]] - prices[peak_indices[1]]) < 0.01 * prices[peak_indices[0]]

# Double Bottom detection
def is_double_bottom(prices):
    if len(prices) < 20:
        return False
    valleys = (prices[1:-1] < prices[:-2]) & (prices[1:-1] < prices[2:])
    valley_indices = [i for i, v in enumerate(valleys, 1) if v]
    
    return len(valley_indices) >= 2 and abs(prices[valley_indices[0]] - prices[valley_indices[1]]) < 0.01 * prices[valley_indices[0]]

# Symmetrical Triangle detection
def is_symmetrical_triangle(prices):
    if len(prices) < 20:
        return False
    peaks = (prices[1:-1] > prices[:-2]) & (prices[1:-1] > prices[2:])
    valleys = (prices[1:-1] < prices[:-2]) & (prices[1:-1] < prices[2:])
    
    peak_indices = [i for i, p in enumerate(peaks, 1) if p]
    valley_indices = [i for i, v in enumerate(valleys, 1) if v]
    
    if len(peak_indices) < 2 or len(valley_indices) < 2:
        return False
    
    return (prices[peak_indices[-1]] < prices[peak_indices[0]]) and (prices[valley_indices[-1]] > prices[valley_indices[0]])

# Ascending Triangle detection
def is_ascending_triangle(prices):
    if len(prices) < 20:
        return False
    peaks = (prices[1:-1] > prices[:-2]) & (prices[1:-1] > prices[2:])
    valleys = (prices[1:-1] < prices[:-2]) & (prices[1:-1] < prices[2:])
    
    peak_indices = [i for i, p in enumerate(peaks, 1) if p]
    valley_indices = [i for i, v in enumerate(valleys, 1) if v]
    
    return (len(peak_indices) >= 2 and len(valley_indices) >= 2 and
            prices[valley_indices[-1]] > prices[valley_indices[0]] and
            prices[peak_indices[-1]] < prices[peak_indices[0]])

# Descending Triangle detection
def is_descending_triangle(prices):
    if len(prices) < 20:
        return False
    peaks = (prices[1:-1] > prices[:-2]) & (prices[1:-1] > prices[2:])
    valleys = (prices[1:-1] < prices[:-2]) & (prices[1:-1] < prices[2:])
    
    peak_indices = [i for i, p in enumerate(peaks, 1) if p]
    valley_indices = [i for i, v in enumerate(valleys, 1) if v]
    
    return (len(peak_indices) >= 2 and len(valley_indices) >= 2 and
            prices[valley_indices[-1]] < prices[valley_indices[0]] and
            prices[peak_indices[-1]] > prices[peak_indices[0]])

# Bullish percentage calculation
def calculate_bullish_percentage(data):
    bullish_count = sum(data['Close'].diff().dropna() > 0)
    total_count = len(data) - 1
    return (bullish_count / total_count * 100) if total_count > 0 else 0

# Bearish percentage calculation
def calculate_bearish_percentage(data):
    bearish_count = sum(data['Close'].diff().dropna() < 0)
    total_count = len(data) - 1
    return (bearish_count / total_count * 100) if total_count > 0 else 0

# Streamlit UI
st.title("Stock Recommendations App")

stock_input = st.text_input("Enter stock tickers (comma separated)", "AAPL, MSFT, TSLA")
if st.button("Generate Recommendations"):
    stock_list = [stock.strip().upper() for stock in stock_input.split(",")]
    recommendations = {}

    for stock in stock_list:
        indicators = fetch_indicators(stock)
        recommendations[stock] = indicators
    
    # Display recommendations as tables
    for stock, indicators in recommendations.items():
        st.subheader(f"{stock} Recommendations")
        if indicators:
            df = pd.DataFrame([indicators])  # Create DataFrame from a single dictionary

            # Handle any potential issues
            df = df.fillna('N/A')  # Fill NaN with a placeholder
            
            # Ensure all data types are consistent
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str)

            st.write(df.dtypes)  # Inspect data types
            st.dataframe(df)
        else:
            st.write("No recommendations available.")
