import torch
import torch.nn as nn


class TimeSeriesTransformer(nn.Module):
    """
    A transformer-based model specifically designed for financial time series,
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
        """Initialize weights with finance-specific considerations."""
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
        Forward pass of the model.

        Args:
            x: input tensor of shape [batch_size, seq_length, input_dim]
            mask: optional mask for the transformer

        Returns:
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
