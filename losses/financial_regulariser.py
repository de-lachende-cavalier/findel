import torch
import torch.nn as nn

from .sharpe_ratio_loss import SharpeRatioLoss
from .sortino_ratio_loss import SortinoRatioLoss
from .max_drawdown_loss import MaxDrawdownLoss


class FinancialRegularizerLoss(nn.Module):
    """
    A composite loss function that combines financial metrics with standard ML losses.

    This loss encourages both prediction accuracy and good financial characteristics.
    """

    def __init__(
        self,
        base_criterion=nn.MSELoss(),
        sharpe_weight=0.1,
        sortino_weight=0.1,
        drawdown_weight=0.1,
        volatility_weight=0.1,
    ):
        """
        Args:
            base_criterion: Base loss function (default: MSELoss)
            sharpe_weight: Weight for Sharpe ratio component
            sortino_weight: Weight for Sortino ratio component
            drawdown_weight: Weight for drawdown component
            volatility_weight: Weight for volatility component
        """
        super().__init__()
        self.base_criterion = base_criterion
        self.sharpe_weight = sharpe_weight
        self.sortino_weight = sortino_weight
        self.drawdown_weight = drawdown_weight
        self.volatility_weight = volatility_weight

        # Financial loss components
        self.sharpe_loss = SharpeRatioLoss()
        self.sortino_loss = SortinoRatioLoss()
        self.drawdown_loss = MaxDrawdownLoss()

    def forward(self, pred_returns, true_returns, true_next_returns=None):
        """
        Calculate the combined loss.

        Args:
            pred_returns: Predicted returns [batch_size, sequence_length]
            true_returns: Actual historical returns [batch_size, sequence_length]
            true_next_returns: Actual future returns for evaluation [batch_size, horizon]
                               If none, uses pred_returns for financial metrics

        Returns:
            Combined loss (scalar)
        """
        # Base prediction loss
        base_loss = self.base_criterion(pred_returns, true_returns)

        # If true_next_returns is not provided, use pred_returns for financial metrics
        financial_returns = (
            true_next_returns if true_next_returns is not None else pred_returns
        )

        # Financial losses
        sharpe_loss = (
            self.sharpe_loss(financial_returns) if self.sharpe_weight > 0 else 0
        )
        sortino_loss = (
            self.sortino_loss(financial_returns) if self.sortino_weight > 0 else 0
        )
        drawdown_loss = (
            self.drawdown_loss(financial_returns) if self.drawdown_weight > 0 else 0
        )

        # Volatility penalty (if enabled)
        volatility_loss = 0
        if self.volatility_weight > 0:
            volatility = torch.std(financial_returns, dim=1)
            volatility_loss = torch.mean(volatility)

        # Combine all losses
        total_loss = (
            base_loss
            + self.sharpe_weight * sharpe_loss
            + self.sortino_weight * sortino_loss
            + self.drawdown_weight * drawdown_loss
            + self.volatility_weight * volatility_loss
        )

        return total_loss
