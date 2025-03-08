import torch.nn as nn


class MultiTaskNetwork(nn.Module):
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
