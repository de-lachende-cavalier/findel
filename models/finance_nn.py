"""
finance-specific neural network architectures that incorporate domain knowledge.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FinancialTimeSeriesTransformer(nn.Module):
    """
    a transformer-based model specifically designed for financial time series,
    with modifications to handle non-stationarity and temporal dependencies.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        d_model=128,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=512,
        dropout=0.1,
        activation="gelu",
        max_seq_length=252,  # typical number of trading days in a year
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.max_seq_length = max_seq_length

        # input embedding
        self.input_embedding = nn.Linear(input_dim, d_model)

        # positional encoding with learnable parameters for financial data
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_length, d_model))

        # time-aware embedding to capture market regimes
        self.time_embedding = nn.Parameter(torch.zeros(1, max_seq_length, d_model))

        # transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_encoder_layers
        )

        # output layers with risk-aware structure
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim),
        )

        # initialize weights
        self._init_weights()

    def _init_weights(self):
        """initialize weights with finance-specific considerations."""
        # initialize embeddings
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.time_embedding, mean=0.0, std=0.02)

        # initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # xavier initialization for better gradient flow
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, mask=None):
        """
        forward pass of the model.

        args:
            x: input tensor of shape [batch_size, seq_length, input_dim]
            mask: optional mask for the transformer

        returns:
            output tensor of shape [batch_size, seq_length, output_dim]
        """
        # get sequence length
        seq_length = x.size(1)

        # embed inputs
        x = self.input_embedding(x)

        # add positional encoding
        x = x + self.pos_embedding[:, :seq_length, :]

        # add time-aware embedding for market regimes
        x = x + self.time_embedding[:, :seq_length, :]

        # apply transformer encoder
        if mask is None:
            # create causal mask for autoregressive modeling
            mask = nn.Transformer.generate_square_subsequent_mask(seq_length).to(
                x.device
            )

        x = self.transformer_encoder(x, mask=mask)

        # apply output layer
        output = self.output_layer(x)

        return output


class FinancialRiskAwareGRU(nn.Module):
    """
    a gru-based model with risk-aware components specifically designed for financial forecasting.
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
        forward pass with risk-aware outputs.

        args:
            x: input tensor of shape [batch_size, seq_length, input_dim]

        returns:
            dictionary containing:
                - mean: predicted mean values
                - volatility: predicted volatility (uncertainty)
        """
        # apply gru
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


class FinancialMultiTaskNetwork(nn.Module):
    """
    a multi-task learning model for financial applications that jointly predicts
    multiple related financial metrics.
    """

    def __init__(
        self,
        input_dim,
        shared_dim=128,
        task_specific_dim=64,
        num_tasks=3,  # e.g., return prediction, volatility estimation, and risk metrics
        dropout=0.2,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.shared_dim = shared_dim
        self.task_specific_dim = task_specific_dim
        self.num_tasks = num_tasks

        # shared layers
        self.shared_network = nn.Sequential(
            nn.Linear(input_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # task-specific layers
        self.task_networks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(shared_dim, task_specific_dim),
                    nn.LayerNorm(task_specific_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(task_specific_dim, 1),
                )
                for _ in range(num_tasks)
            ]
        )

    def forward(self, x):
        """
        forward pass for multi-task prediction.

        args:
            x: input tensor of shape [batch_size, input_dim]

        returns:
            list of task-specific outputs
        """
        # shared representation
        shared_features = self.shared_network(x)

        # task-specific predictions
        outputs = [task_network(shared_features) for task_network in self.task_networks]

        return outputs
