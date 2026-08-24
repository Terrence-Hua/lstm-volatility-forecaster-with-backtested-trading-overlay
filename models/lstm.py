"""LSTM volatility forecaster.

Architecture: multi-layer LSTM with dropout, trained on sliding windows of
feature sequences to predict next-day realized variance.

The :class:`LSTMForecaster` wraps a PyTorch LSTM with a scikit-learn-style
fit/predict interface. Training uses AdamW with cosine annealing LR decay.

Usage
-----
>>> from models.lstm import LSTMForecaster
>>> model = LSTMForecaster(input_size=27, hidden_size=64, num_layers=2)
>>> model.fit(X_train_seq, y_train)     # X shape: (N, seq_len, n_features)
>>> preds = model.predict(X_test_seq)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# PyTorch module
# ---------------------------------------------------------------------------


class _LSTMNet(nn.Module):
    """Internal PyTorch LSTM network.

    Parameters
    ----------
    input_size:
        Number of input features per time step.
    hidden_size:
        LSTM hidden state dimension.
    num_layers:
        Number of stacked LSTM layers.
    dropout:
        Dropout probability applied between LSTM layers (0 disables it).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softplus(),  # enforce non-negative variance output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            Input tensor of shape (batch, seq_len, input_size).

        Returns
        -------
        torch.Tensor
            Shape (batch,) — one variance forecast per sequence.
        """
        _, (h_n, _) = self.lstm(x)  # h_n: (num_layers, batch, hidden)
        last_hidden = h_n[-1]       # (batch, hidden)
        out = self.head(last_hidden)
        return out.squeeze(-1)      # (batch,)


# ---------------------------------------------------------------------------
# High-level wrapper
# ---------------------------------------------------------------------------


def make_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice (X, y) into sliding windows of length *seq_len*.

    Parameters
    ----------
    X:
        Feature matrix of shape (T, n_features).
    y:
        Target array of shape (T,).
    seq_len:
        Lookback window length.

    Returns
    -------
    (X_seq, y_seq): tuple[np.ndarray, np.ndarray]
        X_seq has shape (T - seq_len, seq_len, n_features).
        y_seq has shape (T - seq_len,) — target at the end of each window.
    """
    n = len(X)
    if n <= seq_len:
        raise ValueError(f"seq_len ({seq_len}) must be < len(X) ({n})")
    xs, ys = [], []
    for i in range(seq_len, n):
        xs.append(X[i - seq_len : i])
        ys.append(y[i])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


class LSTMForecaster:
    """LSTM-based volatility forecaster.

    Parameters
    ----------
    input_size:
        Number of features per time step.
    hidden_size:
        LSTM hidden dimension (default 64).
    num_layers:
        Number of stacked LSTM layers (default 2).
    dropout:
        Dropout between LSTM layers (default 0.2).
    seq_len:
        Lookback sequence length in days (default 22).
    batch_size:
        Mini-batch size during training (default 64).
    lr:
        Initial learning rate (default 1e-3).
    epochs:
        Training epochs (default 50).
    seed:
        Random seed for reproducibility.
    device:
        Torch device string. Defaults to 'cuda' if available else 'cpu'.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        seq_len: int = 22,
        batch_size: int = 64,
        lr: float = 1e-3,
        epochs: int = 50,
        seed: int = 42,
        device: str | None = None,
    ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._net: _LSTMNet | None = None
        self.train_losses_: list[float] = []
        self._x_scaler: StandardScaler = StandardScaler()
        self._y_scaler: StandardScaler = StandardScaler()
        self._fitted: bool = False

    def _init_net(self) -> None:
        torch.manual_seed(self.seed)
        self._net = _LSTMNet(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = False,
    ) -> "LSTMForecaster":
        """Fit the LSTM on feature sequences.

        Parameters
        ----------
        X:
            Feature matrix of shape (T, n_features) or already-sequenced
            (T, seq_len, n_features). If 2-D, sequences are built internally.
        y:
            Target array of shape (T,). Must align with X.
        verbose:
            Print per-epoch loss if True.

        Returns
        -------
        self
        """
        self._init_net()
        assert self._net is not None

        if X.ndim == 2:
            # Fit scalers on training data
            self._x_scaler.fit(X)
            X_scaled = self._x_scaler.transform(X)
            y_2d = y.reshape(-1, 1)
            self._y_scaler.fit(y_2d)
            y_scaled = self._y_scaler.transform(y_2d).ravel()
            X_seq, y_seq = make_sequences(X_scaled, y_scaled, self.seq_len)
        else:
            X_seq, y_seq = X.astype(np.float32), y.astype(np.float32)

        self._fitted = True
        Xt = torch.from_numpy(X_seq).to(self.device)
        yt = torch.from_numpy(y_seq).to(self.device)

        dataset = TensorDataset(Xt, yt)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.lr * 0.01
        )
        loss_fn = nn.MSELoss()

        self._net.train()
        self.train_losses_ = []

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self._net(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            epoch_loss /= len(dataset)
            self.train_losses_.append(epoch_loss)
            scheduler.step()
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  epoch {epoch+1}/{self.epochs}  loss={epoch_loss:.6f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict next-day realized variance for each window in X.

        Parameters
        ----------
        X:
            Feature matrix (T, n_features) or sequenced (T, seq_len, n_features).
            If 2-D, sequences are built and a dummy y=zeros is used (targets ignored).

        Returns
        -------
        np.ndarray
            Variance forecasts. If 2-D input, length is T - seq_len.
        """
        if self._net is None:
            raise RuntimeError("Call fit() before predict().")

        if X.ndim == 2:
            if self._fitted:
                X_scaled = self._x_scaler.transform(X)
            else:
                X_scaled = X
            dummy_y = np.zeros(len(X_scaled), dtype=np.float32)
            X_seq, _ = make_sequences(X_scaled, dummy_y, self.seq_len)
        else:
            X_seq = X.astype(np.float32)

        Xt = torch.from_numpy(X_seq).to(self.device)
        self._net.eval()
        with torch.no_grad():
            preds_scaled = self._net(Xt).cpu().numpy()

        if self._fitted:
            preds = self._y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
        else:
            preds = preds_scaled
        return np.maximum(preds, 0.0)

    def save(self, path: str) -> None:
        """Serialize model weights to *path* (.pt file).

        Parameters
        ----------
        path:
            File path (e.g. 'results/models/lstm.pt').
        """
        if self._net is None:
            raise RuntimeError("Nothing to save — model not trained.")
        torch.save(
            {
                "state_dict": self._net.state_dict(),
                "config": {
                    "input_size": self.input_size,
                    "hidden_size": self.hidden_size,
                    "num_layers": self.num_layers,
                    "dropout": self.dropout,
                    "seq_len": self.seq_len,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str | None = None) -> "LSTMForecaster":
        """Load a previously saved forecaster.

        Parameters
        ----------
        path:
            Path to the .pt checkpoint saved by :meth:`save`.
        device:
            Override device for inference.

        Returns
        -------
        LSTMForecaster
        """
        ckpt = torch.load(path, map_location="cpu")
        cfg = ckpt["config"]
        obj = cls(
            input_size=cfg["input_size"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
            seq_len=cfg["seq_len"],
            device=device,
        )
        obj._init_net()
        assert obj._net is not None
        obj._net.load_state_dict(ckpt["state_dict"])
        return obj
