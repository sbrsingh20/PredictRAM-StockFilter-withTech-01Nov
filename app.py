import streamlit as st
import pandas as pd
import yfinance as yf
import ta

# Function to fetch stock indicators
def fetch_indicators(stock):
    ticker = yf.Ticker(stock)
    data = ticker.history(period="1y")

    if data.empty or len(data) < 2:  # Check if DataFrame is empty or has less than 2 rows
        return {
            'RSI': None,
            'MACD': None,
            'MACD_Signal': None,
            'MACD_Hist': None,
            'Upper_BB': None,
            'Lower_BB': None,
            'Volatility': None,
            'Beta': None,
            'Close': None,
            'Volume': None,
            'SMA_50': None,
            'SMA_200': None,
            'EMA_12': None,
            'EMA_26': None,
            'Average_Volume': None,
            'Average_Volume_10d': None,
            'Pattern': None,
            'Strength_Percentage': None,
            'Bullish_Percentage': None,
            'Bearish_Percentage': None
        }

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
    data['Average_Volume'] = data['Volume'].mean()
    data['Average_Volume_10d'] = data['Volume'].rolling(window=10).mean()

    beta = ticker.info.get('beta', None)

    # Calculate additional metrics
    last_close = data['Close'].iloc[-1]
    previous_close = data['Close'].iloc[-2]
    
    if last_close > previous_close:
        pattern = "Bullish"
    elif last_close < previous_close:
        pattern = "Bearish"
    else:
        pattern = "Neutral"

    # Strength Percentage: % change from SMA_50
    strength_percentage = ((last_close - data['SMA_50'].iloc[-1]) / data['SMA_50'].iloc[-1] * 100) if data['SMA_50'].iloc[-1] is not None else 0

    # Bullish and Bearish Percentage
    recent_changes = data['Close'].pct_change().iloc[-20:]  # last 20 days
    bullish_days = (recent_changes > 0).sum()
    bearish_days = (recent_changes < 0).sum()
    total_days = bullish_days + bearish_days
    bullish_percentage = (bullish_days / total_days * 100) if total_days > 0 else 0
    bearish_percentage = (bearish_days / total_days * 100) if total_days > 0 else 0

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
        'Average_Volume': data['Average_Volume'].iloc[-1],
        'Average_Volume_10d': data['Average_Volume_10d'].iloc[-1],
        'Pattern': pattern,
        'Strength_Percentage': strength_percentage,
        'Bullish_Percentage': bullish_percentage,
        'Bearish_Percentage': bearish_percentage
    }

# Function to score stocks based on indicators for different terms
def score_stock(indicators, term):
    score = 0

    if term == 'Short Term':
        if indicators['RSI'] is not None:
            if indicators['RSI'] < 30 or indicators['RSI'] > 70:
                score += 2  # Good
            if 30 <= indicators['RSI'] <= 40 or 60 <= indicators['RSI'] <= 70:
                score += 1  # Neutral

        if indicators['MACD'] is not None:
            if indicators['MACD'] > 0 and indicators['MACD'] > indicators['MACD_Signal']:
                score += 2  # Good

    elif term == 'Medium Term':
        if indicators['RSI'] is not None:
            if 40 <= indicators['RSI'] <= 60:
                score += 2  # Good

        if indicators['MACD'] is not None:
            if abs(indicators['MACD']) < 0.01:  # Close to zero
                score += 1  # Neutral

    elif term == 'Long Term':
        if indicators['RSI'] is not None:
            if 40 <= indicators['RSI'] <= 60:
                score += 2  # Good

        if indicators['Beta'] is not None:
            if 0.9 <= indicators['Beta'] <= 1.1:
                score += 2  # Good

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
            lower_buy_range = current_price * 0.995  # 0.5% lower
            upper_buy_range = current_price * 1.005  # 0.5% higher
            short_stop_loss = current_price * (1 - 0.03)  # Max 3%
            short_target = current_price * (1 + 0.05)  # Min 5%
            medium_stop_loss = current_price * (1 - 0.04)  # Max 4%
            medium_target = current_price * (1 + 0.10)  # Min 10%
            long_stop_loss = current_price * (1 - 0.05)  # Max 5%
            long_target = current_price * (1 + 0.15)  # Min 15%

            short_score = score_stock(indicators, 'Short Term')
            medium_score = score_stock(indicators, 'Medium Term')
            long_score = score_stock(indicators, 'Long Term')

            if short_score > 0:
                recommendations['Short Term'].append({
                    'Stock': stock.replace('.NS', ''),  # Remove .NS
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

    # Display recommendations
    for term, stocks in recommendations.items():
        st.subheader(f"{term} Recommendations")
        for stock in stocks:
            st.write(stock)
