import torch
import torch.nn as nn


class CalmarRatioLoss(nn.Module):
    """
    Loss function that maximizes the Calmar ratio.

    The Calmar ratio is the annualized return divided by the maximum drawdown.
    It's a risk-adjusted return metric that focuses on extreme downside risk.
    """

    def __init__(self, annualization_factor=252, lookback_period=252, eps=1e-6):
        """
        Args:
            annualization_factor: Factor to annualize returns (default: 252 for daily returns)
            lookback_period: Period over which to calculate max drawdown (default: 252 days)
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.annualization_factor = annualization_factor
        self.lookback_period = lookback_period
        self.eps = eps

    def forward(self, returns):
        """
        Calculate the negative Calmar ratio (for minimization).

        Args:
            returns: Tensor of returns [batch_size, sequence_length]

        Returns:
            Negative Calmar ratio (scalar)
        """
        # Calculate annualized return
        mean_return = torch.mean(returns, dim=1)
        annualized_return = (1 + mean_return) ** self.annualization_factor - 1

        # Convert returns to cumulative returns (starting at 1.0)
        cum_returns = torch.cumprod(1 + returns, dim=1)

        # Calculate running maximum
        running_max = torch.cummax(cum_returns, dim=1)[0]

        # Calculate drawdowns
        drawdowns = (running_max - cum_returns) / running_max

        # Get maximum drawdown for each sequence in the batch
        max_drawdowns = torch.max(drawdowns, dim=1)[0] + self.eps

        # Calculate Calmar ratio
        calmar_ratio = annualized_return / max_drawdowns

        # Return negative Calmar ratio for minimization
        return -torch.mean(calmar_ratio)
