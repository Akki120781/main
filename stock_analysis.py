import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import streamlit as st
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Stock Market Analysis", layout="wide")

st.markdown("""
    <style>
    .main {background: linear-gradient(135deg, #0d1b2a 0%, #1b263b 100%);}
    .stButton>button {background: linear-gradient(90deg, #00ffcc, #00ccff); color: #0d1b2a; border-radius: 12px; font-weight: bold;}
    .stButton>button:hover {background: linear-gradient(90deg, #00ccff, #00ffcc); color: #ffffff;}
    .stSidebar {background: linear-gradient(180deg, #1e1e2f 0%, #2d2d44 100%); color: #e0e1dd;}
    h1 {color: #00ffcc; font-family: 'Arial'; text-shadow: 0 0 10px rgba(0, 255, 204, 0.5);}
    h2 {color: #00ccff; font-family: 'Arial'; text-shadow: 0 0 5px rgba(0, 204, 255, 0.3);}
    h3 {color: #ff007f; text-shadow: 0 0 5px rgba(255, 0, 127, 0.3);}
    .metric-box {background: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2); color: #e0e1dd;}
    .stDataFrame {background: rgba(255, 255, 255, 0.05); border-radius: 10px; padding: 10px;}
    </style>
""", unsafe_allow_html=True)

st.title("Stock Market Comparison Dashboard")

with st.sidebar:
    st.markdown("<h2 style='color: #00ffcc;'>Control Panel</h2>", unsafe_allow_html=True)
    
    available_stocks = [
        'BPCL.NS', 'RELIANCE.NS', 'TATAMOTORS.NS', 'INFY.NS', 'TCS.NS',
        'HDFCBANK.NS', 'SBIN.NS', 'ITC.NS', 'HINDUNILVR.NS', 'ASIANPAINT.NS',
        'ICICIBANK.NS', 'AXISBANK.NS', 'KOTAKBANK.NS', 'SUNPHARMA.NS', 'DRREDDY.NS',
        'MARUTI.NS', 'BAJFINANCE.NS', 'HCLTECH.NS', 'WIPRO.NS', 'ADANIENT.NS', 
        'ONGC.NS', 'NTPC.NS', 'COALINDIA.NS', 'LT.NS', 'TECHM.NS'
    ]
    
    selected_stocks = st.multiselect("Select Stocks", available_stocks, default=['BPCL.NS', 'RELIANCE.NS'])
    today = pd.to_datetime(datetime.now().date())
    start_date = st.date_input("Start Date", value=pd.to_datetime('2020-01-01'))
    end_date = st.date_input("End Date", value=today)
    test_size = st.slider("Test Size (%)", 10, 50, 20, 5)
    
    st.markdown("---")
    st.markdown("<h3 style='color: #00ffcc;'>Developer</h3>", unsafe_allow_html=True)
    st.markdown("""
        <div style='color: #e0e1dd;'>
            <strong>Akshat Soni</strong> (Reg: 12317750)<br>
            <a href='https://www.linkedin.com/in/harshchauhan7534/' target='_blank'>
                <img src='https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white'>
            </a>
        </div>
    """, unsafe_allow_html=True)

colors = {
    'BPCL.NS': ('#00ffcc', '#ff007f'), 'RELIANCE.NS': ('#00ccff', '#ffcc00'),
    'TATAMOTORS.NS': ('#ff007f', '#00ff99'), 'INFY.NS': ('#ffcc00', '#ff00ff'),
    'TCS.NS': ('#00ff99', '#ff6600'), 'HDFCBANK.NS': ('#ff00ff', '#00ccff'),
    'SBIN.NS': ('#ff6600', '#ff007f'), 'ITC.NS': ('#00ffcc', '#ffcc00'),
    'HINDUNILVR.NS': ('#ff007f', '#00ff99'), 'ASIANPAINT.NS': ('#ffcc00', '#ff00ff'),
    'ICICIBANK.NS': ('#00ff99', '#ff6600'), 'AXISBANK.NS': ('#ff00ff', '#00ccff'),
    'KOTAKBANK.NS': ('#ff6600', '#00ffcc'), 'SUNPHARMA.NS': ('#00ccff', '#ff007f'),
    'DRREDDY.NS': ('#ff007f', '#ffcc00'), 'MARUTI.NS': ('#00ff99', '#ff00ff'),
    'BAJFINANCE.NS': ('#ffcc00', '#ff6600'), 'HCLTECH.NS': ('#00ffcc', '#ff00ff'),
    'WIPRO.NS': ('#ff6600', '#00ff99'), 'ADANIENT.NS': ('#ff00ff', '#00ccff'),
    'ONGC.NS': ('#00ffcc', '#ff007f'), 'NTPC.NS': ('#ffcc00', '#00ff99'),
    'COALINDIA.NS': ('#ff007f', '#ff6600'), 'LT.NS': ('#00ccff', '#ff00ff'),
    'TECHM.NS': ('#00ff99', '#ffcc00')
}

features = ['Open', 'High', 'Low', 'Volume', 'MA5', 'MA20']

def get_stock_data(ticker, start_date, end_date):
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        if stock_data.empty:
            return None
        stock_data.reset_index(inplace=True)
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        stock_data['MA5'] = stock_data['Close'].rolling(window=5).mean()
        stock_data['MA20'] = stock_data['Close'].rolling(window=20).mean()
        stock_data.dropna(inplace=True)
        return stock_data[['Date', 'Open', 'High', 'Low', 'Volume', 'MA5', 'MA20', 'Close']]
    except Exception:
        return None

def predict_future_ohlc(stock, df, model, scaler, start_date, end_date):
    future_ohlc = []
    last_row = df.iloc[-1]
    current_date = start_date
    while current_date <= end_date:
        future_features = np.array([[float(last_row[f]) for f in features]])
        future_features_scaled = scaler.transform(future_features)
        predicted_close = model.predict(future_features_scaled)[0]
        future_ohlc.append({'Stock': stock, 'Date': current_date.strftime('%Y-%m-%d'), 'Close': float(predicted_close)})
        current_date += timedelta(days=1)
    return future_ohlc

def run_analysis(selected_stocks, start_date, end_date, test_size):
    results = {}
    ohlc_data = []
    future_predictions = []
    test_size = test_size / 100

    for stock in selected_stocks:
        df = get_stock_data(stock, start_date, min(today, pd.to_datetime(end_date)) + timedelta(days=1))
        if df is None or df.empty:
            continue
        
        last_day_df = df[df['Date'] < today]
        today_df = df[df['Date'] == today]
        if not last_day_df.empty:
            ohlc_data.append({
                'Stock': stock, 'Date': last_day_df['Date'].iloc[-1].strftime('%Y-%m-%d'),
                'Open': float(last_day_df['Open'].iloc[-1]), 'High': float(last_day_df['High'].iloc[-1]),
                'Low': float(last_day_df['Low'].iloc[-1]), 'Close': float(last_day_df['Close'].iloc[-1])
            })
        if not today_df.empty:
            ohlc_data.append({
                'Stock': stock, 'Date': today_df['Date'].iloc[0].strftime('%Y-%m-%d'),
                'Open': float(today_df['Open'].iloc[0]), 'High': float(today_df['High'].iloc[0]),
                'Low': float(today_df['Low'].iloc[0]), 'Close': float(today_df['Close'].iloc[0])
            })
        
        X = df[features]
        y = df['Close']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42, shuffle=False)
        
        dates_train = df['Date'][:len(X_train)]
        dates_test = df['Date'][len(X_train):]
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred_test)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred_test)
        
        results[stock] = {
            'df': df, 'dates_train': dates_train, 'dates_test': dates_test,
            'y_train': y_train, 'y_test': y_test, 'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test, 'mse': mse, 'rmse': rmse, 'r2': r2,
            'model': model, 'scaler': scaler
        }
        
        if pd.to_datetime(end_date) > today:
            future_predictions.extend(predict_future_ohlc(stock, df, model, scaler, today + timedelta(days=1), pd.to_datetime(end_date)))
    
    return results, ohlc_data, future_predictions

if st.button("Run Analysis"):
    if not selected_stocks:
        st.error("Please select at least one stock.")
    else:
        with st.spinner("Analyzing stocks..."):
            results, ohlc_data, future_predictions = run_analysis(selected_stocks, start_date, end_date, test_size)
        
        if ohlc_data:
            st.markdown("<h2>Latest Stock Data</h2>", unsafe_allow_html=True)
            ohlc_df = pd.DataFrame(ohlc_data)
            st.dataframe(ohlc_df.style.format("{:.2f}", subset=['Open', 'High', 'Low', 'Close']))

        if future_predictions:
            st.markdown("<h2>Future Predictions</h2>", unsafe_allow_html=True)
            future_df = pd.DataFrame(future_predictions)
            st.dataframe(future_df.style.format("{:.2f}", subset=['Close']))

        if results:
            tab1, tab2 = st.tabs(["Metrics & Plots", "Comparison"])
            
            with tab1:
                for stock in selected_stocks:
                    if stock in results:
                        st.markdown(f"<h3>{stock} Analysis</h3>", unsafe_allow_html=True)
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(f"<div class='metric-box'><b>MSE:</b> {results[stock]['mse']:.2f}</div>", unsafe_allow_html=True)
                        with col2:
                            st.markdown(f"<div class='metric-box'><b>RMSE:</b> {results[stock]['rmse']:.2f}</div>", unsafe_allow_html=True)
                        with col3:
                            st.markdown(f"<div class='metric-box'><b>R²:</b> {results[stock]['r2']:.2f}</div>", unsafe_allow_html=True)
                        
                        fig = plt.figure(figsize=(12, 4), facecolor='#0d1b2a')
                        ax = plt.gca()
                        ax.set_facecolor('#1b263b')
                        ax.plot(results[stock]['dates_train'], results[stock]['y_train'], label='Train', color='#00ffcc', alpha=0.7)
                        ax.plot(results[stock]['dates_train'], results[stock]['y_pred_train'], label='Pred Train', color='#ff007f', linestyle='--')
                        ax.plot(results[stock]['dates_test'], results[stock]['y_test'], label='Test', color='#00ccff')
                        ax.plot(results[stock]['dates_test'], results[stock]['y_pred_test'], label='Pred Test', color='#ffcc00', linestyle='--')
                        ax.set_title(f'{stock} Analysis', color='#e0e1dd')
                        ax.set_xlabel('Date', color='#e0e1dd')
                        ax.set_ylabel('Price', color='#e0e1dd')
                        ax.legend(facecolor='#1b263b', edgecolor='#00ffcc', labelcolor='#e0e1dd')
                        ax.grid(True, alpha=0.2, color='#e0e1dd')
                        ax.tick_params(axis='both', colors='#e0e1dd')
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
            
            with tab2:
                st.markdown("<h2>Stock Comparison</h2>", unsafe_allow_html=True)
                comparison_data = [{'Stock': stock, 'MSE': results[stock]['mse'], 'RMSE': results[stock]['rmse'], 'R²': results[stock]['r2']} 
                                 for stock in selected_stocks if stock in results]
                if comparison_data:
                    st.dataframe(pd.DataFrame(comparison_data).style.format("{:.2f}", subset=['MSE', 'RMSE', 'R²']))
                
                fig = plt.figure(figsize=(12, 6), facecolor='#0d1b2a')
                ax = plt.gca()
                ax.set_facecolor('#1b263b')
                for stock in selected_stocks:
                    if stock in results:
                        ax.plot(results[stock]['dates_test'], results[stock]['y_test'], label=f'{stock} Actual', color=colors[stock][0])
                        ax.plot(results[stock]['dates_test'], results[stock]['y_pred_test'], label=f'{stock} Pred', color=colors[stock][1], linestyle='--')
                ax.set_title('All Stocks - Test Period', color='#e0e1dd')
                ax.set_xlabel('Date', color='#e0e1dd')
                ax.set_ylabel('Price', color='#e0e1dd')
                ax.legend(facecolor='#1b263b', edgecolor='#00ffcc', labelcolor='#e0e1dd')
                ax.grid(True, alpha=0.2, color='#e0e1dd')
                ax.tick_params(axis='both', colors='#e0e1dd')
                plt.xticks(rotation=45)
                st.pyplot(fig)