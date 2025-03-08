#!/usr/bin/env python
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from models import TimeSeriesTransformer, RiskAwareGRU, MultiTaskNetwork

from losses import (
    SharpeRatioLoss,
    SortinoRatioLoss,
    MaxDrawdownLoss,
    FinancialRegularizerLoss,
)


def parse_args():
    """parse command line arguments."""
    parser = argparse.ArgumentParser(description="train financial deep learning models")

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
        "--train_ratio", type=float, default=0.7, help="ratio of data for training"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.15, help="ratio of data for validation"
    )

    # model arguments
    parser.add_argument(
        "--model_type",
        type=str,
        default="transformer",
        choices=["transformer", "gru", "multitask"],
        help="type of model to train",
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=128, help="hidden dimension size"
    )
    parser.add_argument("--num_layers", type=int, default=2, help="number of layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")

    # loss arguments
    parser.add_argument(
        "--loss_type",
        type=str,
        default="financial",
        choices=["mse", "sharpe", "sortino", "drawdown", "financial"],
        help="type of loss function to use",
    )
    parser.add_argument(
        "--sharpe_weight", type=float, default=0.1, help="weight for sharpe ratio loss"
    )
    parser.add_argument(
        "--sortino_weight",
        type=float,
        default=0.1,
        help="weight for sortino ratio loss",
    )
    parser.add_argument(
        "--drawdown_weight", type=float, default=0.1, help="weight for drawdown loss"
    )

    # training arguments
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument(
        "--early_stopping", type=int, default=10, help="early stopping patience"
    )

    # output arguments
    parser.add_argument(
        "--output_dir", type=str, default="./output", help="output directory"
    )
    parser.add_argument(
        "--model_name", type=str, default=None, help="model name for saving"
    )

    return parser.parse_args()


def prepare_data(args):
    """prepare financial data for training."""
    print(f"loading data from {args.data_path}")

    # load data
    df = pd.read_csv(args.data_path)

    # ensure target column exists
    if args.target_column not in df.columns:
        raise ValueError(f"target column '{args.target_column}' not found in data")

    # create sequences
    sequences = []
    targets = []

    for i in range(len(df) - args.sequence_length):
        # get sequence of features
        seq = (
            df.iloc[i : i + args.sequence_length]
            .drop(columns=[args.target_column])
            .values
        )

        # get target (next return)
        target = df.iloc[i : i + args.sequence_length][args.target_column].values

        sequences.append(seq)
        targets.append(target)

    # convert to tensors
    X = torch.tensor(np.array(sequences), dtype=torch.float32)
    y = torch.tensor(np.array(targets), dtype=torch.float32)

    # split data
    total_samples = len(X)
    train_size = int(total_samples * args.train_ratio)
    val_size = int(total_samples * args.val_ratio)
    test_size = total_samples - train_size - val_size

    # create datasets
    train_dataset = TensorDataset(X[:train_size], y[:train_size])
    val_dataset = TensorDataset(
        X[train_size : train_size + val_size], y[train_size : train_size + val_size]
    )
    test_dataset = TensorDataset(X[train_size + val_size :], y[train_size + val_size :])

    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    print(
        f"data prepared: {train_size} training, {val_size} validation, {test_size} test samples"
    )

    return train_loader, val_loader, test_loader, X.shape[2]  # return input dimension


def create_model(args, input_dim):
    """create the specified model."""
    if args.model_type == "transformer":
        model = TimeSeriesTransformer(
            input_dim=input_dim,
            output_dim=1,
            d_model=args.hidden_dim,
            num_encoder_layers=args.num_layers,
            dropout=args.dropout,
            max_seq_length=args.sequence_length,
        )
    elif args.model_type == "gru":
        model = RiskAwareGRU(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=1,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
    elif args.model_type == "multitask":
        model = MultiTaskNetwork(
            input_dim=input_dim * args.sequence_length,  # flatten input for multi-task
            shared_dim=args.hidden_dim,
            task_specific_dim=args.hidden_dim // 2,
            num_tasks=3,  # return, volatility, drawdown
            dropout=args.dropout,
        )
    else:
        raise ValueError(f"unknown model type: {args.model_type}")

    return model


def create_loss_function(args):
    """create the specified loss function."""
    if args.loss_type == "mse":
        return nn.MSELoss()
    elif args.loss_type == "sharpe":
        return SharpeRatioLoss()
    elif args.loss_type == "sortino":
        return SortinoRatioLoss()
    elif args.loss_type == "drawdown":
        return MaxDrawdownLoss()
    elif args.loss_type == "financial":
        return FinancialRegularizerLoss(
            sharpe_weight=args.sharpe_weight,
            sortino_weight=args.sortino_weight,
            drawdown_weight=args.drawdown_weight,
        )
    else:
        raise ValueError(f"unknown loss type: {args.loss_type}")


def train_epoch(model, train_loader, optimizer, loss_fn, device):
    """train for one epoch."""
    model.train()
    total_loss = 0

    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="training")):
        # move data to device
        data, target = data.to(device), target.to(device)

        # zero gradients
        optimizer.zero_grad()

        # forward pass
        if isinstance(model, MultiTaskNetwork):
            # flatten input for multi-task network
            batch_size, seq_len, feat_dim = data.shape
            data_flat = data.reshape(batch_size, seq_len * feat_dim)
            output = model(data_flat)[0]  # take first task output (returns)
            output = output.view(-1, seq_len)
        elif isinstance(model, RiskAwareGRU):
            output = model(data)["mean"]  # take mean prediction
            output = output.view(-1, seq_len)
        else:
            output = model(data).squeeze(-1)  # remove last dimension if needed

        # calculate loss
        loss = loss_fn(output, target)

        # backward pass and optimize
        loss.backward()
        optimizer.step()

        # accumulate loss
        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate(model, val_loader, loss_fn, device):
    """validate the model."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for data, target in tqdm(val_loader, desc="validation"):
            # move data to device
            data, target = data.to(device), target.to(device)

            # forward pass
            if isinstance(model, MultiTaskNetwork):
                # flatten input for multi-task network
                batch_size, seq_len, feat_dim = data.shape
                data_flat = data.reshape(batch_size, seq_len * feat_dim)
                output = model(data_flat)[0]  # take first task output (returns)
                output = output.view(-1, seq_len)
            elif isinstance(model, RiskAwareGRU):
                output = model(data)["mean"]  # take mean prediction
                output = output.view(-1, seq_len)
            else:
                output = model(data).squeeze(-1)  # remove last dimension if needed

            # calculate loss
            loss = loss_fn(output, target)

            # accumulate loss
            total_loss += loss.item()

    return total_loss / len(val_loader)


def main():
    """main training function."""
    args = parse_args()

    # create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    # prepare data
    train_loader, val_loader, test_loader, input_dim = prepare_data(args)

    # create model
    model = create_model(args, input_dim)
    model = model.to(device)
    print(
        f"created {args.model_type} model with {sum(p.numel() for p in model.parameters())} parameters"
    )

    # create loss function
    loss_fn = create_loss_function(args)
    print(f"using {args.loss_type} loss function")

    # create optimizer
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # training loop
    best_val_loss = float("inf")
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(args.epochs):
        print(f"epoch {epoch+1}/{args.epochs}")

        # train
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        train_losses.append(train_loss)

        # validate
        val_loss = validate(model, val_loader, loss_fn, device)
        val_losses.append(val_loss)

        print(f"train loss: {train_loss:.6f}, val loss: {val_loss:.6f}")

        # check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # save best model
            model_name = args.model_name or f"{args.model_type}_{args.loss_type}"
            torch.save(
                model.state_dict(),
                os.path.join(args.output_dir, f"{model_name}_best.pt"),
            )
            print(f"saved best model with val loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1
            print(f"no improvement for {patience_counter} epochs")

        # early stopping
        if patience_counter >= args.early_stopping:
            print(f"early stopping after {epoch+1} epochs")
            break

    # plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="train loss")
    plt.plot(val_losses, label="validation loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"training curves for {args.model_type} with {args.loss_type} loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, f"{model_name}_training_curve.png"))

    print("training completed!")


if __name__ == "__main__":
    main()
