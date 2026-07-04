from torch import nn


class BiLSTMAttention(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden_size=64, num_layers=2, num_heads=8):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.5 if num_layers > 1 else 0.0,
        )
        self.attention = nn.MultiheadAttention(hidden_size * 2, num_heads, batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(hidden_size * 2, 128), nn.Dropout(0.5), nn.Linear(128, n_classes))

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        out, _ = self.attention(out, out, out)
        return self.classifier(out.mean(dim=1))

