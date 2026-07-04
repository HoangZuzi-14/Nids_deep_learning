from __future__ import annotations

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent)


def fit_autoencoder(
    X_normal,
    X_val_normal,
    latent_dim: int = 16,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 256,
    patience: int = 5,
    seed: int = 42,
    device: str = "cpu",
):
    """Fit Autoencoder on normal-only feature rows."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_normal)
    X_val_scaled = scaler.transform(X_val_normal)

    train_ds = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device_obj = torch.device(device)
    model = Autoencoder(input_dim=X_normal.shape[1], latent_dim=latent_dim)
    model.to(device_obj)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    best_state = None
    no_improve_epochs = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device_obj)
            optimizer.zero_grad()
            reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch_x,) in val_loader:
                batch_x = batch_x.to(device_obj)
                reconstructed = model(batch_x)
                loss = criterion(reconstructed, batch_x)
                val_loss += loss.item() * batch_x.size(0)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, scaler


def compute_autoencoder_scores(model, scaler, X, device: str = "cpu"):
    """Return scores where higher means more anomalous (higher reconstruction MSE)."""
    model.eval()
    device_obj = torch.device(device)
    X_scaled = scaler.transform(X)

    ds = TensorDataset(torch.tensor(X_scaled, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=512, shuffle=False)

    scores = []
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device_obj)
            reconstructed = model(batch_x)
            mse = torch.mean((batch_x - reconstructed) ** 2, dim=1).cpu().numpy()
            scores.extend(mse)

    return np.array(scores)
