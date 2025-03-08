import torch
import torch.nn as nn


class SharpeRatioLoss(nn.Module):
    """
    Loss function that maximizes the Sharpe ratio of returns.

    The Sharpe ratio is a measure of risk-adjusted return, calculated as:
    Sharpe = (mean_return - risk_free_rate) / std_return

    Maximizing Sharpe ratio encourages models to balance return and risk.
    """

    def __init__(self, risk_free_rate=0.0, annualization_factor=252, eps=1e-6):
        """
        Args:
            risk_free_rate: The risk-free rate (default: 0.0)
            annualization_factor: Factor to annualize returns (default: 252 for daily returns)
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.risk_free_rate = risk_free_rate
        self.annualization_factor = annualization_factor
        self.eps = eps

    def forward(self, returns):
        """
        Calculate the negative Sharpe ratio (for minimization).

        Args:
            returns: Tensor of returns [batch_size, sequence_length]

        Returns:
            Negative Sharpe ratio (scalar)
        """
        # Calculate mean and standard deviation of returns
        mean_return = torch.mean(returns, dim=1)
        std_return = torch.std(returns, dim=1) + self.eps

        # Calculate Sharpe ratio
        sharpe = (mean_return - self.risk_free_rate) / std_return

        # Annualize if needed
        if self.annualization_factor > 1:
            sharpe = sharpe * torch.sqrt(torch.tensor(self.annualization_factor))

        # Return negative Sharpe for minimization
        return -torch.mean(sharpe)
