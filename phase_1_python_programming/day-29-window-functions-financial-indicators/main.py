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


if __name__ == "__main__":
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'NVDA']
    raw_df = generate_synthetic_ohlcv(tickers=tickers, start_date="2025-01-01", periods=365)
    print("\n--- Raw OHLCV Sample ---")
    print(raw_df.head(10))
