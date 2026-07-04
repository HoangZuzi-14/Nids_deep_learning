from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

from src.training.early_stopping import EarlyStopping


class TorchTrainer:
    def __init__(
        self,
        model,
        device=None,
        lr=1e-3,
        class_weights=None,
        grad_clip=1.0,
        criterion=None,
    ):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        weight = class_weights.to(self.device) if class_weights is not None else None
        self.criterion = criterion.to(self.device) if criterion is not None else nn.CrossEntropyLoss(weight=weight)
        self.grad_clip = grad_clip

    def _run_epoch(self, loader, train: bool):
        self.model.train(train)
        total_loss = 0.0
        correct = 0
        total = 0
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                if train:
                    self.optimizer.zero_grad()
                logits = self.model(X)
                loss = self.criterion(logits, y)
                if train:
                    loss.backward()
                    if self.grad_clip:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
                total_loss += float(loss.item())
                correct += int((logits.argmax(dim=1) == y).sum().item())
                total += int(y.numel())
        return total_loss / max(len(loader), 1), correct / max(total, 1)

    def fit(self, train_loader, val_loader, epochs=30, checkpoint_path=None, patience=10):
        early = EarlyStopping(patience)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        best_state = None
        for _ in range(epochs):
            train_loss, train_acc = self._run_epoch(train_loader, train=True)
            val_loss, val_acc = self._run_epoch(val_loader, train=False)
            scheduler.step(val_loss)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            if val_loss <= early.best:
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                if checkpoint_path:
                    path = Path(checkpoint_path)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(best_state, path)
            if early.step(val_loss):
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return history

    def predict(self, loader):
        self.model.eval()
        preds = []
        true = []
        with torch.no_grad():
            for X, y in loader:
                logits = self.model(X.to(self.device))
                preds.extend(logits.argmax(dim=1).cpu().numpy())
                true.extend(y.numpy())
        return true, preds

    def predict_proba(self, loader):
        self.model.eval()
        probs = []
        true = []
        with torch.no_grad():
            for X, y in loader:
                logits = self.model(X.to(self.device))
                p = torch.softmax(logits, dim=1).cpu().numpy()
                probs.extend(p)
                true.extend(y.numpy())
        return true, np.array(probs)

