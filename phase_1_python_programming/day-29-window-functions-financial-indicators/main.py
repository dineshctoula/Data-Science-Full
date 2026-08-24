"""
===============================================================================
DAY 29: ADVANCED WINDOWING FUNCTIONS, ROLLING METRICS & FINANCIAL INDICATORS
===============================================================================
Author: Dinesh (100-Day Data Science Challenge)
Phase: Phase 2 - Data Manipulation & Visualization
Topic: Rolling Windows, EWMA, Technical Indicators (RSI, MACD, Bollinger), & Grouped Analytics

This module implements a comprehensive financial quantitative analytics pipeline:
1. Multi-Asset OHLCV Time-Series Simulation (Geometric Brownian Motion with Drift)
2. Advanced Rolling Window Transformations & Expanding Window Drawdown Analysis
3. Complex Quantitative Technical Indicators (RSI, MACD, Bollinger Bands, ATR, Sharpe Ratio)
4. GroupBy Rolling Transformations and Cross-Asset Analytics
5. Automated Anomaly Signal Detection & CSV Export Pipeline
===============================================================================
"""

import os
import numpy as np
import pandas as pd

# Set fixed random seed for reproducible financial simulation
np.random.seed(42)


def generate_synthetic_ohlcv(tickers: list, start_date: str = "2025-01-01", periods: int = 365) -> pd.DataFrame:
    """
    Generates a realistic multi-asset Daily OHLCV (Open, High, Low, Close, Volume) 
    financial time-series dataset using Geometric Brownian Motion with drift.

    Parameters:
        tickers (list): List of asset ticker symbols (e.g., ['AAPL', 'GOOGL', 'MSFT', 'NVDA']).
        start_date (str): Starting timestamp for daily market calendar.
        periods (int): Number of trading days to simulate.

    Returns:
        pd.DataFrame: Clean multi-asset time series with DatetimeIndex and columns:
                      ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
    """
    print(f"[INFO] Generating synthetic OHLCV data for {len(tickers)} assets over {periods} trading days...")
    
    # Generate business day date range excluding weekends
    dates = pd.date_range(start=start_date, periods=periods, freq='B')
    
    records = []
    
    # Initial price parameters per ticker
    initial_prices = {'AAPL': 180.0, 'GOOGL': 140.0, 'MSFT': 400.0, 'NVDA': 120.0}
    annual_drifts = {'AAPL': 0.12, 'GOOGL': 0.10, 'MSFT': 0.15, 'NVDA': 0.35}
    annual_vols = {'AAPL': 0.22, 'GOOGL': 0.25, 'MSFT': 0.20, 'NVDA': 0.40}
    
    # Daily step scaling factor (assuming 252 trading days per year)
    dt = 1.0 / 252.0

    for ticker in tickers:
        s0 = initial_prices.get(ticker, 100.0)
        mu = annual_drifts.get(ticker, 0.10)
        sigma = annual_vols.get(ticker, 0.25)
        
        # Simulate daily returns via Geometric Brownian Motion
        # S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma * sqrt(dt) * Z)
        random_shocks = np.random.normal(0, 1, periods)
        drift_term = (mu - 0.5 * sigma**2) * dt
        diffusion_term = sigma * np.sqrt(dt) * random_shocks
        daily_log_returns = drift_term + diffusion_term
        
        # Calculate daily closing prices cumulative path
        close_prices = s0 * np.exp(np.cumsum(daily_log_returns))
        
        for i, date in enumerate(dates):
            close_p = float(close_prices[i])
            
            # Generate intraday High, Low, and Open relative to Close price
            intraday_vol = sigma * np.sqrt(dt) * close_p * 0.7
            open_p = close_p + np.random.normal(0, intraday_vol * 0.5)
            high_p = max(open_p, close_p) + abs(np.random.normal(0, intraday_vol))
            low_p = min(open_p, close_p) - abs(np.random.normal(0, intraday_vol))
            
            # Ensure logical validity: Low <= Open, Close <= High
            low_p = max(0.01, low_p)
            high_p = max(high_p, open_p, close_p)
            
            # Generate daily trading volume with log-normal distribution
            base_vol = 1_000_000 if ticker in ['AAPL', 'NVDA'] else 500_000
            volume = int(np.random.lognormal(mean=np.log(base_vol), sigma=0.4))
            
            records.append({
                'Date': date,
                'Ticker': ticker,
                'Open': round(open_p, 2),
                'High': round(high_p, 2),
                'Low': round(low_p, 2),
                'Close': round(close_p, 2),
                'Volume': volume
            })
            
    df = pd.DataFrame(records)
    df.sort_values(by=['Ticker', 'Date'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    print(f"[SUCCESS] Synthetic OHLCV dataset created successfully. Total Rows: {len(df)}")
    return df


def apply_rolling_and_expanding_transformations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Simple Moving Averages (SMA), Exponential Moving Averages (EMA),
    rolling volatility, expanding peak equity, and drawdown metrics per asset group.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing 'Ticker', 'Date', and 'Close' columns.

    Returns:
        pd.DataFrame: Augmented DataFrame with computed rolling and expanding columns.
    """
    print("\n[INFO] Computing rolling moving averages, EWMA, volatility, and drawdown analytics...")
    
    # Ensure dataset is properly sorted by Ticker and Date
    df = df.sort_values(by=['Ticker', 'Date']).copy()
    
    # Group by ticker to isolate rolling windows per asset without data leakage across boundaries
    grouped_close = df.groupby('Ticker')['Close']
    
    # 1. Simple Moving Averages (SMA)
    df['SMA_20'] = grouped_close.transform(lambda s: s.rolling(window=20, min_periods=1).mean())
    df['SMA_50'] = grouped_close.transform(lambda s: s.rolling(window=50, min_periods=1).mean())
    df['SMA_200'] = grouped_close.transform(lambda s: s.rolling(window=200, min_periods=1).mean())
    
    # 2. Exponentially Weighted Moving Averages (EMA / EWMA)
    # EMA_t = Alpha * Price_t + (1 - Alpha) * EMA_{t-1} where Alpha = 2 / (span + 1)
    df['EMA_12'] = grouped_close.transform(lambda s: s.ewm(span=12, adjust=False).mean())
    df['EMA_26'] = grouped_close.transform(lambda s: s.ewm(span=26, adjust=False).mean())
    
    # 3. Rolling Volatility and Higher-Order Statistical Moments
    df['Rolling_Std_20'] = grouped_close.transform(lambda s: s.rolling(window=20, min_periods=1).std().fillna(0.0))
    df['Rolling_Var_20'] = grouped_close.transform(lambda s: s.rolling(window=20, min_periods=1).var().fillna(0.0))
    df['Rolling_Skew_30'] = grouped_close.transform(lambda s: s.rolling(window=30, min_periods=5).skew().fillna(0.0))
    
    # 4. Daily Returns and Cumulative Performance
    df['Daily_Return'] = df.groupby('Ticker')['Close'].pct_change().fillna(0.0)
    df['Cumulative_Return'] = df.groupby('Ticker')['Daily_Return'].transform(lambda s: (1 + s).cumprod() - 1.0)
    
    # 5. Expanding Windows: Peak Close Price and Max Drawdown %
    # Peak Equity is the historical maximum closing price up to date t
    df['Peak_Close'] = df.groupby('Ticker')['Close'].cummax()
    
    # Drawdown % = (Current Close - Peak Close) / Peak Close * 100
    df['Drawdown_Pct'] = (df['Close'] - df['Peak_Close']) / df['Peak_Close'] * 100.0
    
    # Maximum Drawdown % to date t (Expanding Minimum of Drawdown_Pct)
    df['Max_Drawdown_Pct'] = df.groupby('Ticker')['Drawdown_Pct'].cummin()
    
    print(f"[SUCCESS] Rolling and expanding window metrics successfully engineered.")
    return df


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates advanced quantitative financial indicators including 
    RSI (Relative Strength Index), MACD (Moving Average Convergence Divergence), 
    Bollinger Bands, ATR (Average True Range), and annualized Sharpe Ratio.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing OHLC prices and moving averages.

    Returns:
        pd.DataFrame: Augmented DataFrame with quantitative technical indicators.
    """
    print("\n[INFO] Computing complex technical indicators (RSI, MACD, BB, ATR)...")
    df = df.copy()
    grouped = df.groupby('Ticker')

    # 1. RSI (Relative Strength Index) - 14 Day
    delta = grouped['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    # Use exponential moving average for smoothing RSI
    avg_gain = gain.groupby(df['Ticker']).ewm(com=13, adjust=False).mean().reset_index(level=0, drop=True)
    avg_loss = loss.groupby(df['Ticker']).ewm(com=13, adjust=False).mean().reset_index(level=0, drop=True)
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['RSI_14'] = 100.0 - (100.0 / (1.0 + rs))
    df['RSI_14'] = df['RSI_14'].fillna(50.0)  # Neutral RSI for early periods

    # 2. MACD (Moving Average Convergence Divergence)
    if 'EMA_12' not in df.columns or 'EMA_26' not in df.columns:
        df['EMA_12'] = grouped['Close'].transform(lambda s: s.ewm(span=12, adjust=False).mean())
        df['EMA_26'] = grouped['Close'].transform(lambda s: s.ewm(span=26, adjust=False).mean())
        
    df['MACD_Line'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = grouped['MACD_Line'].transform(lambda s: s.ewm(span=9, adjust=False).mean())
    df['MACD_Histogram'] = df['MACD_Line'] - df['MACD_Signal']

    # 3. Bollinger Bands (20-day SMA +/- 2 * 20-day Std)
    if 'SMA_20' not in df.columns or 'Rolling_Std_20' not in df.columns:
        df['SMA_20'] = grouped['Close'].transform(lambda s: s.rolling(window=20, min_periods=1).mean())
        df['Rolling_Std_20'] = grouped['Close'].transform(lambda s: s.rolling(window=20, min_periods=1).std().fillna(0.0))
        
    df['BB_Upper'] = df['SMA_20'] + (df['Rolling_Std_20'] * 2)
    df['BB_Lower'] = df['SMA_20'] - (df['Rolling_Std_20'] * 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['SMA_20']

    # 4. Average True Range (ATR) - 14 Day
    prev_close = grouped['Close'].shift(1)
    high_low = df['High'] - df['Low']
    high_pc = (df['High'] - prev_close).abs()
    low_pc = (df['Low'] - prev_close).abs()
    
    tr = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)
    df['TR'] = tr
    df['ATR_14'] = grouped['TR'].transform(lambda s: s.rolling(window=14, min_periods=1).mean())

    # 5. Rolling Annualized Sharpe Ratio (Assuming Risk-Free Rate = 0.0)
    if 'Daily_Return' not in df.columns:
        df['Daily_Return'] = grouped['Close'].pct_change().fillna(0.0)
        
    df['Rolling_Sharpe_60'] = grouped['Daily_Return'].transform(
        lambda s: (s.rolling(window=60, min_periods=10).mean() / 
                   s.rolling(window=60, min_periods=10).std().replace(0, np.nan)) * np.sqrt(252)
    ).fillna(0.0)

    print("[SUCCESS] Complex technical indicators engineered successfully.")
    return df


if __name__ == "__main__":
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'NVDA']
    raw_df = generate_synthetic_ohlcv(tickers=tickers, start_date="2025-01-01", periods=365)
    transformed_df = apply_rolling_and_expanding_transformations(raw_df)
    tech_df = compute_technical_indicators(transformed_df)
    
    print("\n--- Technical Indicators Sample (AAPL) ---")
    sample_cols = ['Date', 'Ticker', 'Close', 'RSI_14', 'MACD_Line', 'BB_Upper', 'ATR_14', 'Rolling_Sharpe_60']
    print(tech_df[tech_df['Ticker'] == 'AAPL'][sample_cols].tail(10))
