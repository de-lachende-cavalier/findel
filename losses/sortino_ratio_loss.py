import torch
import torch.nn as nn


class SortinoRatioLoss(nn.Module):
    """
    Loss function that maximizes the Sortino ratio of returns.

    The Sortino ratio is similar to Sharpe but only penalizes downside volatility.
    Sortino = (mean_return - risk_free_rate) / downside_deviation
    """

    def __init__(
        self, risk_free_rate=0.0, target_return=0.0, annualization_factor=252, eps=1e-6
    ):
        """
        Args:
            risk_free_rate: The risk-free rate (default: 0.0)
            target_return: Minimum acceptable return (default: 0.0)
            annualization_factor: Factor to annualize returns (default: 252 for daily returns)
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.risk_free_rate = risk_free_rate
        self.target_return = target_return
        self.annualization_factor = annualization_factor
        self.eps = eps

    def forward(self, returns):
        """
        Calculate the negative Sortino ratio (for minimization).

        Args:
            returns: Tensor of returns [batch_size, sequence_length]

        Returns:
            Negative Sortino ratio (scalar)
        """
        # calculate mean return
        mean_return = torch.mean(returns, dim=1)

        # calculate downside deviation (only negative returns relative to target)
        downside_diff = torch.clamp(self.target_return - returns, min=0)
        downside_diff_squared = downside_diff**2
        downside_variance = torch.mean(downside_diff_squared, dim=1)
        downside_deviation = torch.sqrt(downside_variance + self.eps)

        # calculate sortino ratio
        sortino = (mean_return - self.risk_free_rate) / downside_deviation

        # annualize if needed
        if self.annualization_factor > 1:
            sortino = sortino * torch.sqrt(torch.tensor(self.annualization_factor))

        # return negative sortino for minimization
        return -torch.mean(sortino)
