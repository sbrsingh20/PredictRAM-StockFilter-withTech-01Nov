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
            'Pattern_Interval', 'Strength_Percentage', 
            'Bullish_Percentage', 'Bearish_Percentage'
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
    pattern, interval = detect_chart_pattern(data)

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
        'Pattern_Interval': interval,
        'Strength_Percentage': ((last_close - data['SMA_50'].iloc[-1]) / data['SMA_50'].iloc[-1] * 100) if data['SMA_50'].iloc[-1] is not None else 0,
        'Bullish_Percentage': calculate_bullish_percentage(data),
        'Bearish_Percentage': calculate_bearish_percentage(data)
    }

# Function to detect chart patterns
def detect_chart_pattern(data):
    if len(data) < 30:  # Need at least 30 points to identify patterns
        return "No Pattern", None

    recent_prices = data['Close'].tail(30).values
    time_index = data.index[-30:]

    patterns = {
        "Head and Shoulders": is_head_and_shoulders(recent_prices, time_index),
        "Double Top": is_double_top(recent_prices, time_index),
        "Double Bottom": is_double_bottom(recent_prices, time_index),
    }
    
    recognized_patterns = {name: interval for name, (detected, interval) in patterns.items() if detected}
    
    return list(recognized_patterns.keys()) if recognized_patterns else ["No Recognized Pattern"], list(recognized_patterns.values())

# Head and Shoulders detection
def is_head_and_shoulders(prices, time_index):
    if len(prices) < 20:
        return False, None
    peaks = (prices[1:-1] > prices[:-2]) & (prices[1:-1] > prices[2:])
    valley_indices = [i for i, v in enumerate((prices[1:-1] < prices[:-2]) & (prices[1:-1] < prices[2:]), 1) if v]
    peak_indices = [i for i, p in enumerate(peaks, 1) if p]
    
    if len(peak_indices) >= 2 and len(valley_indices) >= 1:
        pattern_interval = f"{time_index[peak_indices[0]]} to {time_index[peak_indices[-1]]}"
        return True, pattern_interval

    return False, None

# Double Top detection
def is_double_top(prices, time_index):
    if len(prices) < 20:
        return False, None
    peaks = (prices[1:-1] > prices[:-2]) & (prices[1:-1] > prices[2:])
    peak_indices = [i for i, p in enumerate(peaks, 1) if p]
    
    if len(peak_indices) >= 2 and abs(prices[peak_indices[0]] - prices[peak_indices[1]]) < 0.01 * prices[peak_indices[0]]:
        pattern_interval = f"{time_index[peak_indices[0]]} to {time_index[peak_indices[1]]}"
        return True, pattern_interval

    return False, None

# Double Bottom detection
def is_double_bottom(prices, time_index):
    if len(prices) < 20:
        return False, None
    valleys = (prices[1:-1] < prices[:-2]) & (prices[1:-1] < prices[2:])
    valley_indices = [i for i, v in enumerate(valleys, 1) if v]
    
    if len(valley_indices) >= 2 and abs(prices[valley_indices[0]] - prices[valley_indices[1]]) < 0.01 * prices[valley_indices[0]]:
        pattern_interval = f"{time_index[valley_indices[0]]} to {time_index[valley_indices[1]]}"
        return True, pattern_interval

    return False, None

# Function to calculate bullish percentage
def calculate_bullish_percentage(data):
    bullish_count = sum(data['Close'].diff().dropna() > 0)
    total_count = len(data) - 1
    return (bullish_count / total_count * 100) if total_count > 0 else 0

# Function to calculate bearish percentage
def calculate_bearish_percentage(data):
    bearish_count = sum(data['Close'].diff().dropna() < 0)
    total_count = len(data) - 1
    return (bearish_count / total_count * 100) if total_count > 0 else 0

# Function to score stocks based on indicators for different terms
def score_stock(indicators, term):
    score = 0

    if term == 'Short Term':
        if indicators['RSI'] is not None:
            if indicators['RSI'] < 30 or indicators['RSI'] > 70:
                score += 2
            if 30 <= indicators['RSI'] <= 40 or 60 <= indicators['RSI'] <= 70:
                score += 1

        if indicators['MACD'] is not None:
            if indicators['MACD'] > 0 and indicators['MACD'] > indicators['MACD_Signal']:
                score += 2

    elif term == 'Medium Term':
        if indicators['RSI'] is not None:
            if 40 <= indicators['RSI'] <= 60:
                score += 2

        if indicators['MACD'] is not None:
            if abs(indicators['MACD']) < 0.01:
                score += 1

    elif term == 'Long Term':
        if indicators['RSI'] is not None:
            if 40 <= indicators['RSI'] <= 60:
                score += 2

        if indicators['Beta'] is not None:
            if 0.9 <= indicators['Beta'] <= 1.1:
                score += 2

    return score

# Function to generate recommendations based on different strategies
def generate_recommendations(indicators_list):
    recommendations = {
        'Short Term': [],
        'Medium Term': [],
        'Long Term': []
    }
    
    for stock, indicators in indicators_list.items():
        for term in recommendations.keys():
            score = score_stock(indicators, term)
            if score >= 3:  # Threshold for recommendation
                recommendations[term].append({
                    'Stock': stock,
                    'Score': score,
                    'Pattern': indicators['Pattern'],
                    'Pattern_Interval': indicators['Pattern_Interval']
                })
    
    return recommendations

# Main Streamlit application
st.title('Stock Indicator Analysis')

# Upload file
uploaded_file = st.file_uploader("Upload a CSV or Excel file with stock symbols", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        stock_df = pd.read_csv(uploaded_file)
    else:
        stock_df = pd.read_excel(uploaded_file)

    stock_symbols = stock_df['Stock'].tolist()  # Assuming the column is named 'Stock'
    
    # Fetch indicators for all stocks
    indicators_list = {}
    for stock in stock_symbols:
        indicators = fetch_indicators(stock)
        indicators_list[stock] = indicators

    # Generate recommendations based on the fetched indicators
    recommendations = generate_recommendations(indicators_list)

    # Display recommendations as tables
    for term, stocks in recommendations.items():
        st.subheader(f"{term} Recommendations")
        if stocks:
            df = pd.DataFrame(stocks)
            st.dataframe(df)
        else:
            st.write("No recommendations available.")
