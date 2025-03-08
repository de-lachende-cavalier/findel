import torch
import torch.nn as nn


class MaxDrawdownLoss(nn.Module):
    """
    Loss function that penalizes maximum drawdown.

    Maximum drawdown measures the largest peak-to-trough decline in portfolio value,
    which is a key risk metric in finance.
    """

    def __init__(self, alpha=1.0):
        """
        Args:
            alpha: Weight for the drawdown penalty
        """
        super().__init__()
        self.alpha = alpha

    def forward(self, returns):
        """
        Calculate the maximum drawdown loss.

        Args:
            returns: Tensor of returns [batch_size, sequence_length]

        Returns:
            Maximum drawdown loss (scalar)
        """
        # Convert returns to cumulative returns (starting at 1.0)
        batch_size = returns.shape[0]
        cum_returns = torch.cumprod(1 + returns, dim=1)

        # Calculate running maximum
        running_max = torch.cummax(cum_returns, dim=1)[0]

        # Calculate drawdowns
        drawdowns = (running_max - cum_returns) / running_max

        # Get maximum drawdown for each sequence in the batch
        max_drawdowns = torch.max(drawdowns, dim=1)[0]

        # Return weighted maximum drawdown
        return self.alpha * torch.mean(max_drawdowns)
