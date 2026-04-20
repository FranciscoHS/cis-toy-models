"""Minimal nn.Module classes that exactly mirror our CiS architecture,
so they can be decomposed by the new SPD framework.

Plain:     y = W_out @ ReLU(W_in @ x)                 (F -> n -> F)
Embedded:  y = E^T @ W_out @ ReLU(W_in @ E @ x)       (F -> D -> n -> D -> F)

No residual connection in either case (unlike the paper's target).

Naming: target_module_patterns = ["mlp_in", "mlp_out"] will pick up both
Linear layers in either model.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PlainCiSModel(nn.Module):
    """Plain non-embedded CiS model: y = W_out @ ReLU(W_in @ x)."""

    def __init__(self, n_features: int, d_mlp: int):
        super().__init__()
        self.n_features = n_features
        self.d_mlp = d_mlp
        self.mlp_in = nn.Linear(n_features, d_mlp, bias=False)
        self.mlp_out = nn.Linear(d_mlp, n_features, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp_out(F.relu(self.mlp_in(x)))

    @classmethod
    def from_our_weights(cls, weights_path: str, n_features: int, d_mlp: int,
                         device: str = "cpu") -> "PlainCiSModel":
        sd = torch.load(weights_path, map_location=device, weights_only=True)
        W_in = sd["W_in"]      # shape (n, F) in our convention (W_in @ x)
        W_out = sd["W_out"]    # shape (F, n) in our convention (W_out @ h)
        assert W_in.shape == (d_mlp, n_features), f"W_in {W_in.shape} vs {(d_mlp, n_features)}"
        assert W_out.shape == (n_features, d_mlp)

        model = cls(n_features=n_features, d_mlp=d_mlp).to(device)
        # nn.Linear.weight shape is (out_features, in_features). For mlp_in that's
        # (d_mlp, n_features) which matches our W_in directly. For mlp_out it is
        # (n_features, d_mlp) which matches our W_out directly.
        with torch.no_grad():
            model.mlp_in.weight.copy_(W_in.to(device))
            model.mlp_out.weight.copy_(W_out.to(device))
        return model


class EmbeddedCiSModel(nn.Module):
    """Embedded CiS model: y = E^T @ W_out @ ReLU(W_in @ E @ x).
    E (D x F) is a fixed buffer (not decomposed).
    """

    def __init__(self, n_features: int, d_embed: int, d_mlp: int):
        super().__init__()
        self.n_features = n_features
        self.d_embed = d_embed
        self.d_mlp = d_mlp
        # Buffer for the fixed random embedding (D x F)
        self.register_buffer("E", torch.zeros(d_embed, n_features))
        self.mlp_in = nn.Linear(d_embed, d_mlp, bias=False)
        self.mlp_out = nn.Linear(d_mlp, d_embed, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        r = x @ self.E.T                      # (..., D)  embed
        z = F.relu(self.mlp_in(r))            # (..., n)
        r_out = self.mlp_out(z)               # (..., D)
        return r_out @ self.E                 # (..., F)  unembed

    @classmethod
    def from_our_weights(cls, weights_path: str, device: str = "cpu") -> "EmbeddedCiSModel":
        d = torch.load(weights_path, map_location=device, weights_only=True)
        W_in = d["W_in"]     # shape (n, D)
        W_out = d["W_out"]   # shape (D, n)
        E = d["E"]           # shape (D, F)
        D, F_ = E.shape
        n = W_in.shape[0]
        assert W_in.shape == (n, D)
        assert W_out.shape == (D, n)

        model = cls(n_features=F_, d_embed=D, d_mlp=n).to(device)
        with torch.no_grad():
            model.E.copy_(E.to(device))
            # mlp_in.weight shape (n, D) = our W_in; mlp_out.weight shape (D, n) = our W_out
            model.mlp_in.weight.copy_(W_in.to(device))
            model.mlp_out.weight.copy_(W_out.to(device))
        return model
