"""
Reusable data preprocessing and EDA utilities for GMF portfolio optimization project
"""

import yfinance as yf
import time
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller


def fetch_data(tickers, start_date, end_date, auto_adjust=False, sleep_seconds=2):
    """Fetch historical data from Yahoo Finance with rate-limit protection."""
    frames = []

    for ticker in tickers:
        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                auto_adjust=auto_adjust,
                group_by="column",
                progress=False
            )

            if df.empty:
                raise ValueError(f"No data returned for {ticker}")

            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.reset_index()
            df["Asset"] = ticker
            frames.append(df)

            # Sleep to avoid Yahoo rate limits
            time.sleep(sleep_seconds)

        except Exception as e:
            print(f"⚠️ Failed to download {ticker}: {e}")

    if not frames:
        raise RuntimeError("All downloads failed due to rate limiting")

    combined_df = pd.concat(frames, ignore_index=True)
    return combined_df


def clean_data(df):
    """Clean and prepare raw financial data."""
    df = df.sort_values(["Asset", "Date"])
    df = df.ffill()
    df = df.dropna()
    return df


def add_features(df, volatility_window=30):
    """Add daily returns and rolling volatility (robust & simple)."""
    df = df.copy()

    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"

    # Daily returns per asset (guaranteed Series)
    df["Daily_Return"] = df.groupby("Asset")[price_col].pct_change()

    # Rolling volatility
    df["Rolling_Volatility"] = (
        df.groupby("Asset")["Daily_Return"]
        .rolling(window=volatility_window)
        .std()
        .reset_index(level=0, drop=True)
    )

    return df


def adf_test(series):
    """Run Augmented Dickey-Fuller test."""
    result = adfuller(series.dropna())
    return {
        "test_statistic": result[0],
        "p_value": result[1],
        "critical_values": result[4],
    }


def calculate_var(returns, confidence=0.95):
    """Calculate historical Value at Risk."""
    return np.percentile(returns.dropna(), (1 - confidence) * 100)


def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate historical Sharpe Ratio."""
    daily_rf = risk_free_rate / 252
    return (returns.mean() - daily_rf) / returns.std()


def save_processed_data(df, path):
    """Save processed dataset to disk."""
    df.to_csv(path, index=False)
