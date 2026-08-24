"""GRU volatility forecaster.

Identical interface to :class:`models.lstm.LSTMForecaster`, using a GRU cell
instead of LSTM. This lets us benchmark LSTM vs GRU in the walk-forward
evaluation without changing any surrounding code.

The architecture: multi-layer GRU → linear head → Softplus (non-negative
output for variance). Feature scaling (StandardScaler) is applied internally.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from models.lstm import make_sequences


# ---------------------------------------------------------------------------
# PyTorch module
# ---------------------------------------------------------------------------


class _GRUNet(nn.Module):
    """Internal GRU network.

    Parameters
    ----------
    input_size:
        Number of input features per time step.
    hidden_size:
        GRU hidden state dimension.
    num_layers:
        Number of stacked GRU layers.
    dropout:
        Dropout between GRU layers (0 disables it).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
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
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            Shape (batch, seq_len, input_size).

        Returns
        -------
        torch.Tensor
            Shape (batch,).
        """
        _, h_n = self.gru(x)   # h_n: (num_layers, batch, hidden)
        last_hidden = h_n[-1]  # (batch, hidden)
        out = self.head(last_hidden)
        return out.squeeze(-1)


# ---------------------------------------------------------------------------
# High-level wrapper
# ---------------------------------------------------------------------------


class GRUForecaster:
    """GRU-based volatility forecaster.

    Same interface as :class:`models.lstm.LSTMForecaster`. See that class for
    full parameter documentation.

    Parameters
    ----------
    input_size:
        Number of features per time step.
    hidden_size:
        GRU hidden dimension (default 64).
    num_layers:
        Stacked GRU layers (default 2).
    dropout:
        Dropout between layers (default 0.2).
    seq_len:
        Lookback window in days (default 22).
    batch_size:
        Mini-batch size (default 64).
    lr:
        Initial learning rate (default 1e-3).
    epochs:
        Training epochs (default 50).
    seed:
        Random seed.
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
        self._net: _GRUNet | None = None
        self.train_losses_: list[float] = []
        self._x_scaler: StandardScaler = StandardScaler()
        self._y_scaler: StandardScaler = StandardScaler()
        self._fitted: bool = False

    def _init_net(self) -> None:
        torch.manual_seed(self.seed)
        self._net = _GRUNet(
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
    ) -> "GRUForecaster":
        """Fit the GRU on feature sequences.

        Parameters
        ----------
        X:
            Feature matrix (T, n_features) or sequenced (T, seq_len, n_features).
        y:
            Target array (T,).
        verbose:
            Print per-epoch loss if True.

        Returns
        -------
        self
        """
        self._init_net()
        assert self._net is not None

        if X.ndim == 2:
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
        """Predict next-day realized variance.

        Parameters
        ----------
        X:
            Feature matrix (T, n_features) or sequenced (T, seq_len, n_features).

        Returns
        -------
        np.ndarray
            Variance forecasts.
        """
        if self._net is None:
            raise RuntimeError("Call fit() before predict().")

        if X.ndim == 2:
            X_scaled = self._x_scaler.transform(X) if self._fitted else X
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
        """Save model weights to *path*.

        Parameters
        ----------
        path:
            File path ending in .pt.
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
    def load(cls, path: str, device: str | None = None) -> "GRUForecaster":
        """Load a previously saved GRU forecaster.

        Parameters
        ----------
        path:
            Path to .pt checkpoint.
        device:
            Override device for inference.

        Returns
        -------
        GRUForecaster
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
