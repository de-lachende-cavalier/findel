import torch.nn as nn
import torch.nn.functional as F


class RiskAwareGRU(nn.Module):
    """
    A gru-based model with risk-aware components specifically designed for financial forecasting.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers=2,
        dropout=0.1,
        bidirectional=False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # gru layer
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # output dimension adjustment for bidirectional
        output_factor = 2 if bidirectional else 1

        # risk-aware output layers
        self.mean_estimator = nn.Linear(hidden_dim * output_factor, output_dim)
        self.volatility_estimator = nn.Linear(hidden_dim * output_factor, output_dim)

    def forward(self, x):
        """
        Forward pass with risk-aware outputs.

        Args:
            x: input tensor of shape [batch_size, seq_length, input_dim]

        Returns:
            dictionary containing:
                - mean: predicted mean values
                - volatility: predicted volatility (uncertainty)
        """
        output, hidden = self.gru(x)

        # use the output of the last time step
        last_output = output[:, -1, :]

        # estimate mean and volatility
        mean = self.mean_estimator(last_output)

        # ensure positive volatility with softplus
        volatility = F.softplus(self.volatility_estimator(last_output))

        return {
            "mean": mean,
            "volatility": volatility,
        }
