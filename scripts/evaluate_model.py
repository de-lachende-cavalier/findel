#!/usr/bin/env python
"""
evaluation script for financial deep learning models.
"""

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from models.finance_nn import (
    FinancialTimeSeriesTransformer,
    FinancialRiskAwareGRU,
    FinancialMultiTaskNetwork,
)


def parse_args():
    """parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="evaluate financial deep learning models"
    )

    # data arguments
    parser.add_argument(
        "--data_path", type=str, required=True, help="path to financial data csv"
    )
    parser.add_argument(
        "--target_column", type=str, default="returns", help="target column to predict"
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=60,
        help="sequence length for time series",
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.2, help="ratio of data for testing"
    )

    # model arguments
    parser.add_argument(
        "--model_type",
        type=str,
        default="transformer",
        choices=["transformer", "gru", "multitask"],
        help="type of model to evaluate",
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=128, help="hidden dimension size"
    )
    parser.add_argument("--num_layers", type=int, default=2, help="number of layers")
    parser.add_argument(
        "--model_path", type=str, required=True, help="path to saved model"
    )

    # output arguments
    parser.add_argument(
        "--output_dir", type=str, default="./output", help="output directory"
    )

    return parser.parse_args()


def prepare_test_data(args):
    """prepare financial data for evaluation."""
    print(f"loading data from {args.data_path}")

    # load data
    df = pd.read_csv(args.data_path)

    # ensure target column exists
    if args.target_column not in df.columns:
        raise ValueError(f"target column '{args.target_column}' not found in data")

    # use only the test portion
    test_size = int(len(df) * args.test_ratio)
    test_df = df.iloc[-test_size:]

    # create sequences
    sequences = []
    targets = []
    timestamps = []

    for i in range(len(test_df) - args.sequence_length):
        # get sequence of features
        seq = (
            test_df.iloc[i : i + args.sequence_length]
            .drop(columns=[args.target_column])
            .values
        )

        # get target (next return)
        target = test_df.iloc[i : i + args.sequence_length][args.target_column].values

        # get timestamp for the last point in the sequence
        timestamp = test_df.index[i + args.sequence_length - 1]

        sequences.append(seq)
        targets.append(target)
        timestamps.append(timestamp)

    # convert to tensors
    X = torch.tensor(np.array(sequences), dtype=torch.float32)
    y = torch.tensor(np.array(targets), dtype=torch.float32)

    return X, y, timestamps, test_df, X.shape[2]  # return input dimension


def create_model(args, input_dim):
    """create the specified model."""
    if args.model_type == "transformer":
        model = FinancialTimeSeriesTransformer(
            input_dim=input_dim,
            output_dim=1,
            d_model=args.hidden_dim,
            num_encoder_layers=args.num_layers,
            max_seq_length=args.sequence_length,
        )
    elif args.model_type == "gru":
        model = FinancialRiskAwareGRU(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=1,
            num_layers=args.num_layers,
        )
    elif args.model_type == "multitask":
        model = FinancialMultiTaskNetwork(
            input_dim=input_dim * args.sequence_length,  # flatten input for multi-task
            shared_dim=args.hidden_dim,
            task_specific_dim=args.hidden_dim // 2,
            num_tasks=3,  # return, volatility, drawdown
        )
    else:
        raise ValueError(f"unknown model type: {args.model_type}")

    # load model weights
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device("cpu")))
    model.eval()

    return model


def evaluate_model(model, X, y, device):
    """evaluate the model on test data."""
    model.eval()
    all_preds = []

    with torch.no_grad():
        for i in tqdm(range(0, len(X), 64), desc="evaluating"):
            # get batch
            batch_X = X[i : i + 64].to(device)

            # forward pass
            if isinstance(model, FinancialMultiTaskNetwork):
                # flatten input for multi-task network
                batch_size, seq_len, feat_dim = batch_X.shape
                data_flat = batch_X.reshape(batch_size, seq_len * feat_dim)
                output = model(data_flat)[0]  # take first task output (returns)
                output = output.view(-1, seq_len)
            elif isinstance(model, FinancialRiskAwareGRU):
                output = model(batch_X)["mean"]  # take mean prediction
                output = output.view(-1, seq_len)
            else:
                output = model(batch_X).squeeze(-1)  # remove last dimension if needed

            all_preds.append(output.cpu().numpy())

    # concatenate predictions
    predictions = np.vstack(all_preds)

    # convert targets to numpy
    targets = y.numpy()

    return predictions, targets


def calculate_metrics(predictions, targets):
    """calculate evaluation metrics."""
    # reshape if needed
    if predictions.ndim > 1 and predictions.shape[1] > 1:
        # use only the last prediction in the sequence
        predictions = predictions[:, -1]
        targets = targets[:, -1]

    # calculate metrics
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)

    # calculate financial metrics
    # sharpe ratio
    returns_diff = predictions - targets
    sharpe = np.mean(returns_diff) / (np.std(returns_diff) + 1e-6)

    # maximum drawdown
    cum_returns_pred = np.cumprod(1 + predictions)
    cum_returns_true = np.cumprod(1 + targets)

    running_max_pred = np.maximum.accumulate(cum_returns_pred)
    running_max_true = np.maximum.accumulate(cum_returns_true)

    drawdown_pred = (running_max_pred - cum_returns_pred) / running_max_pred
    drawdown_true = (running_max_true - cum_returns_true) / running_max_true

    max_drawdown_pred = np.max(drawdown_pred)
    max_drawdown_true = np.max(drawdown_true)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Sharpe": sharpe,
        "Max Drawdown (Pred)": max_drawdown_pred,
        "Max Drawdown (True)": max_drawdown_true,
    }


def plot_results(predictions, targets, timestamps, metrics, args):
    """plot evaluation results."""
    # create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # plot predictions vs targets
    plt.figure(figsize=(12, 6))

    # if we have sequence predictions, use only the last prediction
    if predictions.ndim > 1 and predictions.shape[1] > 1:
        pred_plot = predictions[:, -1]
        target_plot = targets[:, -1]
    else:
        pred_plot = predictions
        target_plot = targets

    plt.plot(pred_plot, label="predicted")
    plt.plot(target_plot, label="actual")
    plt.xlabel("time")
    plt.ylabel("returns")
    plt.title(f"predicted vs actual returns - {args.model_type.capitalize()} model")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"{args.model_type}_predictions.png"))

    # plot cumulative returns
    plt.figure(figsize=(12, 6))
    cum_returns_pred = np.cumprod(1 + pred_plot)
    cum_returns_true = np.cumprod(1 + target_plot)

    plt.plot(cum_returns_pred, label="predicted")
    plt.plot(cum_returns_true, label="actual")
    plt.xlabel("time")
    plt.ylabel("cumulative returns")
    plt.title(f"cumulative returns - {args.model_type.capitalize()} model")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(args.output_dir, f"{args.model_type}_cumulative_returns.png")
    )

    # plot metrics as a table
    plt.figure(figsize=(10, 6))
    plt.axis("off")
    metrics_table = plt.table(
        cellText=[
            [f"{v:.6f}" if isinstance(v, float) else v for v in metrics.values()]
        ],
        colLabels=list(metrics.keys()),
        loc="center",
        cellLoc="center",
    )
    metrics_table.auto_set_font_size(False)
    metrics_table.set_fontsize(10)
    metrics_table.scale(1, 1.5)
    plt.title(f"evaluation metrics - {args.model_type.capitalize()} model")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"{args.model_type}_metrics.png"))

    # save metrics to csv
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(
        os.path.join(args.output_dir, f"{args.model_type}_metrics.csv"), index=False
    )


def main():
    """main evaluation function."""
    args = parse_args()

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    # prepare test data
    X, y, timestamps, test_df, input_dim = prepare_test_data(args)

    # create and load model
    model = create_model(args, input_dim)
    model = model.to(device)
    print(f"loaded {args.model_type} model from {args.model_path}")

    # evaluate model
    predictions, targets = evaluate_model(model, X, y, device)

    # calculate metrics
    metrics = calculate_metrics(predictions, targets)
    print("evaluation metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")

    # plot results
    plot_results(predictions, targets, timestamps, metrics, args)
    print(f"results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
