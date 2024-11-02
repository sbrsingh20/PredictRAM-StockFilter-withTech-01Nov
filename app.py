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
    pattern, time_interval = detect_chart_pattern(data)

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
        'Time Interval': time_interval,
        'Strength_Percentage': ((last_close - data['SMA_50'].iloc[-1]) / data['SMA_50'].iloc[-1] * 100) if data['SMA_50'].iloc[-1] is not None else 0,
        'Bullish_Percentage': calculate_bullish_percentage(data),
        'Bearish_Percentage': calculate_bearish_percentage(data)
    }

# Function to detect chart patterns
def detect_chart_pattern(data):
    if len(data) < 30:  # Need at least 30 points to identify patterns
        return "No Pattern", None

    recent_prices = data['Close'].tail(30).values

    patterns = {
        "Head and Shoulders": is_head_and_shoulders(recent_prices),
        "Double Top": is_double_top(recent_prices),
        "Double Bottom": is_double_bottom(recent_prices),
        "Flags": is_flag(recent_prices),
        "Pennants": is_pennant(recent_prices),
        "Cup and Handle": is_cup_and_handle(recent_prices),
        "Rounding Bottom": is_rounding_bottom(recent_prices),
        "Symmetrical Triangle": is_symmetrical_triangle(recent_prices),
        "Ascending Triangle": is_ascending_triangle(recent_prices),
        "Descending Triangle": is_descending_triangle(recent_prices),
    }
    
    recognized_patterns = [(name, 'Daily') for name, detected in patterns.items() if detected]  # Assuming daily interval for recognized patterns
    recognized_patterns += [(name, 'Hourly') for name, detected in patterns.items() if detected and name in ["Flags", "Pennants", "Symmetrical Triangle", "Ascending Triangle", "Descending Triangle"]]

    return recognized_patterns if recognized_patterns else ["No Recognized Pattern", None]

# Define missing functions with proper logic
def is_head_and_shoulders(prices):
    # Implement head and shoulders detection logic
    # Return True or False
    return False  # Placeholder

def is_double_top(prices):
    # Implement double top detection logic
    return False  # Placeholder

def is_double_bottom(prices):
    # Implement double bottom detection logic
    return False  # Placeholder

def is_flag(prices):
    # Implement flag detection logic
    return False  # Placeholder

def is_pennant(prices):
    # Implement pennant detection logic
    return False  # Placeholder

def is_cup_and_handle(prices):
    # Implement cup and handle detection logic
    return False  # Placeholder

def is_rounding_bottom(prices):
    # Implement rounding bottom detection logic
    return False  # Placeholder

def is_symmetrical_triangle(prices):
    # Implement symmetrical triangle detection logic
    return False  # Placeholder

def is_ascending_triangle(prices):
    # Implement ascending triangle detection logic
    return False  # Placeholder

def is_descending_triangle(prices):
    # Implement descending triangle detection logic
    return False  # Placeholder

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
        current_price = indicators['Close']
        
        if current_price is not None:
            lower_buy_range = current_price * 0.995
            upper_buy_range = current_price * 1.005
            short_stop_loss = current_price * (1 - 0.03)
            short_target = current_price * (1 + 0.05)
            medium_stop_loss = current_price * (1 - 0.04)
            medium_target = current_price * (1 + 0.10)
            long_stop_loss = current_price * (1 - 0.05)
            long_target = current_price * (1 + 0.15)

            short_score = score_stock(indicators, 'Short Term')
            medium_score = score_stock(indicators, 'Medium Term')
            long_score = score_stock(indicators, 'Long Term')

            if short_score > 0:
                recommendations['Short Term'].append({
                    'Stock': stock.replace('.NS', ''),
                    'Current Price': current_price,
                    'Lower Buy Range': lower_buy_range,
                    'Upper Buy Range': upper_buy_range,
                    'Stop Loss': short_stop_loss,
                    'Target Price': short_target,
                    'Score': short_score,
                    'RSI': indicators['RSI'],
                    'MACD': indicators['MACD'],
                    'MACD_Signal': indicators['MACD_Signal'],
                    'Upper_BB': indicators['Upper_BB'],
                    'Lower_BB': indicators['Lower_BB'],
                    'Volatility': indicators['Volatility'],
                    'Beta': indicators['Beta'],
                    'Volume': indicators['Volume'],
                    'SMA_50': indicators['SMA_50'],
                    'SMA_200': indicators['SMA_200'],
                    'EMA_12': indicators['EMA_12'],
                    'EMA_26': indicators['EMA_26'],
                    'Average_Volume': indicators['Average_Volume'],
                    'Average_Volume_10d': indicators['Average_Volume_10d'],
                    'Pattern': indicators['Pattern'],
                    'Strength_Percentage': indicators['Strength_Percentage'],
                    'Bullish_Percentage': indicators['Bullish_Percentage'],
                    'Bearish_Percentage': indicators['Bearish_Percentage']
                })

            if medium_score > 0:
                recommendations['Medium Term'].append({
                    'Stock': stock.replace('.NS', ''),
                    'Current Price': current_price,
                    'Lower Buy Range': lower_buy_range,
                    'Upper Buy Range': upper_buy_range,
                    'Stop Loss': medium_stop_loss,
                    'Target Price': medium_target,
                    'Score': medium_score,
                    'RSI': indicators['RSI'],
                    'MACD': indicators['MACD'],
                    'MACD_Signal': indicators['MACD_Signal'],
                    'Upper_BB': indicators['Upper_BB'],
                    'Lower_BB': indicators['Lower_BB'],
                    'Volatility': indicators['Volatility'],
                    'Beta': indicators['Beta'],
                    'Volume': indicators['Volume'],
                    'SMA_50': indicators['SMA_50'],
                    'SMA_200': indicators['SMA_200'],
                    'EMA_12': indicators['EMA_12'],
                    'EMA_26': indicators['EMA_26'],
                    'Average_Volume': indicators['Average_Volume'],
                    'Average_Volume_10d': indicators['Average_Volume_10d'],
                    'Pattern': indicators['Pattern'],
                    'Strength_Percentage': indicators['Strength_Percentage'],
                    'Bullish_Percentage': indicators['Bullish_Percentage'],
                    'Bearish_Percentage': indicators['Bearish_Percentage']
                })

            if long_score > 0:
                recommendations['Long Term'].append({
                    'Stock': stock.replace('.NS', ''),
                    'Current Price': current_price,
                    'Lower Buy Range': lower_buy_range,
                    'Upper Buy Range': upper_buy_range,
                    'Stop Loss': long_stop_loss,
                    'Target Price': long_target,
                    'Score': long_score,
                    'RSI': indicators['RSI'],
                    'MACD': indicators['MACD'],
                    'MACD_Signal': indicators['MACD_Signal'],
                    'Upper_BB': indicators['Upper_BB'],
                    'Lower_BB': indicators['Lower_BB'],
                    'Volatility': indicators['Volatility'],
                    'Beta': indicators['Beta'],
                    'Volume': indicators['Volume'],
                    'SMA_50': indicators['SMA_50'],
                    'SMA_200': indicators['SMA_200'],
                    'EMA_12': indicators['EMA_12'],
                    'EMA_26': indicators['EMA_26'],
                    'Average_Volume': indicators['Average_Volume'],
                    'Average_Volume_10d': indicators['Average_Volume_10d'],
                    'Pattern': indicators['Pattern'],
                    'Strength_Percentage': indicators['Strength_Percentage'],
                    'Bullish_Percentage': indicators['Bullish_Percentage'],
                    'Bearish_Percentage': indicators['Bearish_Percentage']
                })

    # Limit the results to 40 stocks for each term
    for term in recommendations:
        recommendations[term] = recommendations[term][:40]

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
