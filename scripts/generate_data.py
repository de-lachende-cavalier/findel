#!/usr/bin/env python
import os
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate test data for financial models"
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="data/processed/test_data.csv",
        help="path to save the generated data",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="synthetic",
        choices=["synthetic", "real"],
        help="type of data to generate",
    )
    parser.add_argument(
        "--ticker", type=str, default="SPY", help="ticker symbol for real data"
    )
    parser.add_argument(
        "--start_date", type=str, default="2018-01-01", help="start date for real data"
    )
    parser.add_argument(
        "--end_date", type=str, default="2025-01-01", help="end date for real data"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1000,
        help="number of samples for synthetic data",
    )
    parser.add_argument(
        "--n_features",
        type=int,
        default=10,
        help="number of features for synthetic data",
    )

    return parser.parse_args()


def generate_synthetic_data(n_samples, n_features):
    """Generate synthetic financial data."""
    print("Generating synthetic financial data...")

    # generate time index
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=n_samples), periods=n_samples, freq="D"
    )

    # generate price series with random walk
    price = 100 * (1 + np.cumsum(np.random.normal(0.0005, 0.01, n_samples)))

    # calculate returns
    returns = np.diff(price) / price[:-1]
    returns = np.insert(returns, 0, 0)

    # generate features (technical indicators, etc.)
    features = {}

    # price-based features
    features["price"] = price
    features["returns"] = returns
    features["log_returns"] = np.log(price / np.roll(price, 1))
    features["log_returns"][0] = 0

    # moving averages
    features["ma_5"] = pd.Series(price).rolling(window=5).mean().values
    features["ma_20"] = pd.Series(price).rolling(window=20).mean().values

    # volatility
    features["volatility_10"] = pd.Series(returns).rolling(window=10).std().values

    # generate additional random features
    for i in range(n_features - len(features)):
        # create features with some correlation to returns
        noise = np.random.normal(0, 1, n_samples)
        feature = 0.7 * returns + 0.3 * noise
        features[f"feature_{i+1}"] = feature

    # create DataFrame
    df = pd.DataFrame(features, index=dates)

    # fill NaN values
    df.fillna(method="bfill", inplace=True)

    return df


def download_real_data(ticker, start_date, end_date):
    """Download real financial data using yfinance."""
    print(f"Downloading data for {ticker} from {start_date} to {end_date}...")

    # download data
    data = yf.download(ticker, start=start_date, end=end_date)

    # calculate returns
    data["returns"] = data["Adj Close"].pct_change()

    # calculate log returns
    data["log_returns"] = np.log(data["Adj Close"] / data["Adj Close"].shift(1))

    # calculate moving averages
    data["ma_5"] = data["Adj Close"].rolling(window=5).mean()
    data["ma_20"] = data["Adj Close"].rolling(window=20).mean()
    data["ma_50"] = data["Adj Close"].rolling(window=50).mean()

    # calculate volatility
    data["volatility_10"] = data["returns"].rolling(window=10).std()
    data["volatility_30"] = data["returns"].rolling(window=30).std()

    # calculate RSI
    delta = data["Adj Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data["rsi"] = 100 - (100 / (1 + rs))

    # drop NaN values
    data.dropna(inplace=True)

    return data


def main():
    """Main function."""
    args = parse_args()

    # create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    if args.data_type == "synthetic":
        df = generate_synthetic_data(args.n_samples, args.n_features)
    else:
        df = download_real_data(args.ticker, args.start_date, args.end_date)

    # save data
    df.to_csv(args.output_path)
    print(f"Data saved to {args.output_path}")

    # plot price and returns
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    if args.data_type == "synthetic":
        plt.plot(df.index, df["price"])
        plt.title("Synthetic Price")
    else:
        plt.plot(df.index, df["Adj Close"])
        plt.title(f"{args.ticker} Price")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(df.index, df["returns"])
    plt.title("Returns")
    plt.grid(True)

    plt.tight_layout()

    # save plot
    plot_path = os.path.join(os.path.dirname(args.output_path), "data_plot.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

    # print data statistics
    print("\nData Statistics:")
    print(f"Number of samples: {len(df)}")
    print(f"Number of features: {len(df.columns)}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print("\nFeature statistics:")
    print(df.describe())


if __name__ == "__main__":
    main()
