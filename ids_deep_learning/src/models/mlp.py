from torch import nn


class MLP(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden_sizes=(256, 128), dropout=0.3):
        super().__init__()
        layers = []
        in_features = n_features
        for hidden in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(in_features, hidden),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden),
                    nn.Dropout(dropout),
                ]
            )
            in_features = hidden
        layers.append(nn.Linear(in_features, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if x.ndim == 3:
            x = x.squeeze(-1)
        return self.net(x)

