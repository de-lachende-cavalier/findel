"""
finance-specific loss functions that incorporate domain knowledge and financial risk metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SharpeRatioLoss(nn.Module):
    """
    loss function that maximizes the sharpe ratio of returns.

    the sharpe ratio is a measure of risk-adjusted return, calculated as:
    sharpe = (mean_return - risk_free_rate) / std_return

    maximizing sharpe ratio encourages models to balance return and risk.
    """

    def __init__(self, risk_free_rate=0.0, annualization_factor=252, eps=1e-6):
        """
        args:
            risk_free_rate: the risk-free rate (default: 0.0)
            annualization_factor: factor to annualize returns (default: 252 for daily returns)
            eps: small constant for numerical stability
        """
        super().__init__()
        self.risk_free_rate = risk_free_rate
        self.annualization_factor = annualization_factor
        self.eps = eps

    def forward(self, returns):
        """
        calculate the negative sharpe ratio (for minimization).

        args:
            returns: tensor of returns [batch_size, sequence_length]

        returns:
            negative sharpe ratio (scalar)
        """
        # calculate mean and standard deviation of returns
        mean_return = torch.mean(returns, dim=1)
        std_return = torch.std(returns, dim=1) + self.eps

        # calculate sharpe ratio
        sharpe = (mean_return - self.risk_free_rate) / std_return

        # annualize if needed
        if self.annualization_factor > 1:
            sharpe = sharpe * torch.sqrt(torch.tensor(self.annualization_factor))

        # return negative sharpe for minimization
        return -torch.mean(sharpe)


class SortinoRatioLoss(nn.Module):
    """
    loss function that maximizes the sortino ratio of returns.

    the sortino ratio is similar to sharpe but only penalizes downside volatility.
    sortino = (mean_return - risk_free_rate) / downside_deviation
    """

    def __init__(
        self, risk_free_rate=0.0, target_return=0.0, annualization_factor=252, eps=1e-6
    ):
        """
        args:
            risk_free_rate: the risk-free rate (default: 0.0)
            target_return: minimum acceptable return (default: 0.0)
            annualization_factor: factor to annualize returns (default: 252 for daily returns)
            eps: small constant for numerical stability
        """
        super().__init__()
        self.risk_free_rate = risk_free_rate
        self.target_return = target_return
        self.annualization_factor = annualization_factor
        self.eps = eps

    def forward(self, returns):
        """
        calculate the negative sortino ratio (for minimization).

        args:
            returns: tensor of returns [batch_size, sequence_length]

        returns:
            negative sortino ratio (scalar)
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


class MaxDrawdownLoss(nn.Module):
    """
    loss function that penalizes maximum drawdown.

    maximum drawdown measures the largest peak-to-trough decline in portfolio value,
    which is a key risk metric in finance.
    """

    def __init__(self, alpha=1.0):
        """
        args:
            alpha: weight for the drawdown penalty
        """
        super().__init__()
        self.alpha = alpha

    def forward(self, returns):
        """
        calculate the maximum drawdown loss.

        args:
            returns: tensor of returns [batch_size, sequence_length]

        returns:
            maximum drawdown loss (scalar)
        """
        # convert returns to cumulative returns (starting at 1.0)
        batch_size = returns.shape[0]
        cum_returns = torch.cumprod(1 + returns, dim=1)

        # calculate running maximum
        running_max = torch.cummax(cum_returns, dim=1)[0]

        # calculate drawdowns
        drawdowns = (running_max - cum_returns) / running_max

        # get maximum drawdown for each sequence in the batch
        max_drawdowns = torch.max(drawdowns, dim=1)[0]

        # return weighted maximum drawdown
        return self.alpha * torch.mean(max_drawdowns)


class CalmarRatioLoss(nn.Module):
    """
    loss function that maximizes the calmar ratio.

    the calmar ratio is the annualized return divided by the maximum drawdown.
    it's a risk-adjusted return metric that focuses on extreme downside risk.
    """

    def __init__(self, annualization_factor=252, lookback_period=252, eps=1e-6):
        """
        args:
            annualization_factor: factor to annualize returns (default: 252 for daily returns)
            lookback_period: period over which to calculate max drawdown (default: 252 days)
            eps: small constant for numerical stability
        """
        super().__init__()
        self.annualization_factor = annualization_factor
        self.lookback_period = lookback_period
        self.eps = eps

    def forward(self, returns):
        """
        calculate the negative calmar ratio (for minimization).

        args:
            returns: tensor of returns [batch_size, sequence_length]

        returns:
            negative calmar ratio (scalar)
        """
        # calculate annualized return
        mean_return = torch.mean(returns, dim=1)
        annualized_return = (1 + mean_return) ** self.annualization_factor - 1

        # convert returns to cumulative returns (starting at 1.0)
        cum_returns = torch.cumprod(1 + returns, dim=1)

        # calculate running maximum
        running_max = torch.cummax(cum_returns, dim=1)[0]

        # calculate drawdowns
        drawdowns = (running_max - cum_returns) / running_max

        # get maximum drawdown for each sequence in the batch
        max_drawdowns = torch.max(drawdowns, dim=1)[0] + self.eps

        # calculate calmar ratio
        calmar_ratio = annualized_return / max_drawdowns

        # return negative calmar ratio for minimization
        return -torch.mean(calmar_ratio)


class FinancialRegularizerLoss(nn.Module):
    """
    a composite loss function that combines financial metrics with standard ml losses.

    this loss encourages both prediction accuracy and good financial characteristics.
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
        args:
            base_criterion: base loss function (default: mseloss)
            sharpe_weight: weight for sharpe ratio component
            sortino_weight: weight for sortino ratio component
            drawdown_weight: weight for drawdown component
            volatility_weight: weight for volatility component
        """
        super().__init__()
        self.base_criterion = base_criterion
        self.sharpe_weight = sharpe_weight
        self.sortino_weight = sortino_weight
        self.drawdown_weight = drawdown_weight
        self.volatility_weight = volatility_weight

        # financial loss components
        self.sharpe_loss = SharpeRatioLoss()
        self.sortino_loss = SortinoRatioLoss()
        self.drawdown_loss = MaxDrawdownLoss()

    def forward(self, pred_returns, true_returns, true_next_returns=None):
        """
        calculate the combined loss.

        args:
            pred_returns: predicted returns [batch_size, sequence_length]
            true_returns: actual historical returns [batch_size, sequence_length]
            true_next_returns: actual future returns for evaluation [batch_size, horizon]
                               if none, uses pred_returns for financial metrics

        returns:
            combined loss (scalar)
        """
        # base prediction loss
        base_loss = self.base_criterion(pred_returns, true_returns)

        # if true_next_returns is not provided, use pred_returns for financial metrics
        financial_returns = (
            true_next_returns if true_next_returns is not None else pred_returns
        )

        # financial losses
        sharpe_loss = (
            self.sharpe_loss(financial_returns) if self.sharpe_weight > 0 else 0
        )
        sortino_loss = (
            self.sortino_loss(financial_returns) if self.sortino_weight > 0 else 0
        )
        drawdown_loss = (
            self.drawdown_loss(financial_returns) if self.drawdown_weight > 0 else 0
        )

        # volatility penalty (if enabled)
        volatility_loss = 0
        if self.volatility_weight > 0:
            volatility = torch.std(financial_returns, dim=1)
            volatility_loss = torch.mean(volatility)

        # combine all losses
        total_loss = (
            base_loss
            + self.sharpe_weight * sharpe_loss
            + self.sortino_weight * sortino_loss
            + self.drawdown_weight * drawdown_loss
            + self.volatility_weight * volatility_loss
        )

        return total_loss
