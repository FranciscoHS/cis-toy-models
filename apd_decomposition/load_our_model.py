"""Wrap our trained CiS models as ResidualMLPModel / ResidualMLPSPDModel
instances suitable for APD decomposition.

Our architecture:  y = E^T W_out ReLU(W_in E x),  no residual connection.
APD's architecture: y = W_U (residual + MLP(residual)),  residual = W_E x.

To match, we patch the forward method to skip the residual add. We also
disable the output activation.

For our 'plain' (non-embedded) models, W_E = I_F and W_U = I_F (set as fixed
identity).
For our 'embedded' models, W_E[:, :] = E^T.T = E (shape n_instances x F x D)
and W_U = E^T (shape n_instances x D x F).

Note: our W_E/W_U orientation: ResidualMLPModel stores W_E with shape
(n_instances, n_features, d_embed) so x @ W_E gives residual of shape
(batch, n_instances, d_embed). For our embedded case our E is (D, F), and
we want residual = E @ x in R^D. Via einsum "batch f, f d -> batch d" with
W_E of shape (f=F, d=D). So W_E[j, :] = E[:, j]. i.e. W_E = E.T.
Then W_U of shape (D, F) with x @ W_U = y in R^F means W_U[d, j] = E[d, j],
i.e. W_U = E. So for E^T unembedding: y = E^T @ r_out, and via
"batch d, d f -> batch f" we want W_U[d, j] = (E^T)[j, d] = E[d, j],
i.e. W_U = E. Yes, W_U = E and W_E = E.T.
"""

import json
from pathlib import Path
from typing import Any

import einops
import torch
import torch.nn.functional as F
import yaml
from torch import nn, Tensor

from spd.experiments.resid_mlp.models import (
    ResidualMLPModel,
    ResidualMLPConfig,
    ResidualMLPSPDModel,
    ResidualMLPSPDConfig,
)


def _forward_no_residual(self, x, return_residual: bool = False):
    """Forward pass that REPLACES the residual instead of adding to it.
    Matches our architecture: y = W_U @ MLP(W_E @ x).
    """
    assert x.shape[1] == self.config.n_instances, "n_instances mismatch"
    assert x.shape[2] == self.config.n_features, "n_features mismatch"
    residual = einops.einsum(
        x, self.W_E,
        "batch n_instances n_features, n_instances n_features d_embed "
        "-> batch n_instances d_embed",
    )
    for layer in self.layers:
        residual = layer(residual)
    out = einops.einsum(
        residual, self.W_U,
        "batch n_instances d_embed, n_instances d_embed n_features "
        "-> batch n_instances n_features",
    )
    if self.config.apply_output_act_fn:
        out = self.act_fn(out)
    return residual if return_residual else out


def _spd_forward_no_residual(self, x, topk_mask=None):
    """Same, for the SPD model."""
    residual = einops.einsum(
        x, self.W_E,
        "batch n_instances n_features, n_instances n_features d_embed "
        "-> batch n_instances d_embed",
    )
    for layer in self.layers:
        residual = layer(residual, topk_mask)
    out = einops.einsum(
        residual, self.W_U,
        "batch n_instances d_embed, n_instances d_embed n_features "
        "-> batch n_instances n_features",
    )
    if self.config.apply_output_act_fn:
        out = self.act_fn(out)
    return out


def install_no_residual_patch():
    ResidualMLPModel.forward = _forward_no_residual
    ResidualMLPSPDModel.forward = _spd_forward_no_residual


def load_plain_model(weights_path: str, n_features: int, d_mlp: int,
                     device: str = "cpu"):
    """Create a ResidualMLPModel with identity W_E/W_U and our plain weights
    (W_in, W_out) loaded in.
    Returns: (model, metadata_dict)
    """
    cfg = ResidualMLPConfig(
        n_instances=1,
        n_features=n_features,
        d_embed=n_features,        # W_E = I_F, so d_embed = n_features
        d_mlp=d_mlp,
        n_layers=1,
        act_fn_name="relu",
        apply_output_act_fn=False,
        in_bias=False,
        out_bias=False,
        init_scale=1.0,
    )
    model = ResidualMLPModel(cfg).to(device)

    # Fix W_E and W_U to identity
    with torch.no_grad():
        model.W_E.data.zero_()
        model.W_U.data.zero_()
        for i in range(n_features):
            model.W_E.data[0, i, i] = 1.0
            model.W_U.data[0, i, i] = 1.0
    model.W_E.requires_grad_(False)
    model.W_U.requires_grad_(False)

    # Load our saved W_in (n x F) and W_out (F x n)
    sd = torch.load(weights_path, map_location=device, weights_only=True)
    W_in = sd["W_in"]         # (n, F)
    W_out = sd["W_out"]       # (F, n)
    assert W_in.shape == (d_mlp, n_features), f"{W_in.shape} vs {(d_mlp, n_features)}"
    assert W_out.shape == (n_features, d_mlp)

    # APD's Linear.weight has shape (n_instances, d_in, d_out); forward is
    # out = einsum("b i, i o -> b o", x, weight). So mlp_in.weight is (1, F, n)
    # while our W_in is (n, F); store W_in.T. Similarly mlp_out.weight is (1, n, F)
    # while our W_out is (F, n); store W_out.T.
    with torch.no_grad():
        model.layers[0].mlp_in.weight.data[0] = W_in.T.contiguous().to(device)
        model.layers[0].mlp_out.weight.data[0] = W_out.T.contiguous().to(device)

    meta = dict(n_features=n_features, d_embed=n_features, d_mlp=d_mlp,
                variant="plain", weights_path=weights_path)
    return model, meta


def load_embedded_model(weights_path: str, device: str = "cpu"):
    """Create a ResidualMLPModel with W_E = E.T, W_U = E and our embedded
    weights loaded in.
    """
    d = torch.load(weights_path, map_location=device, weights_only=True)
    W_in = d["W_in"]        # (n, D)
    W_out = d["W_out"]      # (D, n)
    E = d["E"]              # (D, F)
    meta_d = d["meta"] if "meta" in d else {}
    D, F = E.shape
    n = W_in.shape[0]
    assert W_out.shape == (D, n)

    cfg = ResidualMLPConfig(
        n_instances=1,
        n_features=F,
        d_embed=D,
        d_mlp=n,
        n_layers=1,
        act_fn_name="relu",
        apply_output_act_fn=False,
        in_bias=False,
        out_bias=False,
        init_scale=1.0,
    )
    model = ResidualMLPModel(cfg).to(device)

    # W_E of shape (n_instances=1, F, D); contractions "batch f, f d -> batch d"
    # require W_E[j, :] = E[:, j], i.e. W_E = E.T.
    # W_U of shape (n_instances=1, D, F); contraction "batch d, d f -> batch f"
    # with y = E.T @ r = sum_d r_d * E[d, j] means W_U[d, j] = E[d, j], i.e. W_U = E.
    with torch.no_grad():
        model.W_E.data[0] = E.T.to(device)
        model.W_U.data[0] = E.to(device)
    model.W_E.requires_grad_(False)
    model.W_U.requires_grad_(False)

    with torch.no_grad():
        # mlp_in.weight shape (1, D, n) = W_in.T; mlp_out.weight shape (1, n, D) = W_out.T
        model.layers[0].mlp_in.weight.data[0] = W_in.T.contiguous().to(device)
        model.layers[0].mlp_out.weight.data[0] = W_out.T.contiguous().to(device)

    meta = dict(n_features=F, d_embed=D, d_mlp=n, variant="embedded",
                weights_path=weights_path, **meta_d)
    return model, meta


def verify_model_matches(model, weights_path: str, variant: str, p: float = 0.02,
                         n_samples: int = 512, device: str = "cpu"):
    """Sanity check: APD-wrapped model should produce the same output as our
    direct implementation."""
    sd = torch.load(weights_path, map_location=device, weights_only=True)
    F_ = model.config.n_features
    D = model.config.d_embed
    n = model.config.d_mlp

    torch.manual_seed(0)
    x = torch.zeros(n_samples, F_, device=device)
    mask = torch.rand(n_samples, F_, device=device) < p
    x[mask] = torch.rand(int(mask.sum()), device=device) * 2 - 1

    if variant == "plain":
        W_in = sd["W_in"].to(device)
        W_out = sd["W_out"].to(device)
        y_ref = (W_out @ torch.relu(W_in @ x.T)).T
    else:
        W_in = sd["W_in"].to(device)
        W_out = sd["W_out"].to(device)
        E = sd["E"].to(device)
        r = x @ E.T                           # (B, D)
        z = r @ W_in.T                        # (B, n)
        h = torch.relu(z)
        r_out = h @ W_out.T                   # (B, D)
        y_ref = r_out @ E                     # (B, F)

    x_expanded = x.unsqueeze(1)  # add n_instances=1 dim
    with torch.no_grad():
        y_apd = model(x_expanded).squeeze(1)

    diff = (y_apd - y_ref).abs().max().item()
    return diff


if __name__ == "__main__":
    install_no_residual_patch()
    device = "cpu"

    # Plain 20f/5n
    m, meta = load_plain_model("weights/small_20f_5n_L4.pt", 20, 5, device)
    diff = verify_model_matches(m, "weights/small_20f_5n_L4.pt", "plain")
    print(f"plain 20f/5n: diff = {diff:.2e}")

    # Plain 20f/2n
    m, meta = load_plain_model("weights/small_20f_2n_L4.pt", 20, 2, device)
    diff = verify_model_matches(m, "weights/small_20f_2n_L4.pt", "plain")
    print(f"plain 20f/2n: diff = {diff:.2e}")

    # Plain 100f/10n
    m, meta = load_plain_model("weights/small_100f_10n_L4.pt", 100, 10, device)
    diff = verify_model_matches(m, "weights/small_100f_10n_L4.pt", "plain")
    print(f"plain 100f/10n: diff = {diff:.2e}")

    # Embedded 20f/5n, D=80
    m, meta = load_embedded_model("weights/embed_20f_5n_D80_unit_L4.pt", device)
    diff = verify_model_matches(m, "weights/embed_20f_5n_D80_unit_L4.pt", "embedded")
    print(f"embedded 20f/5n D=80: diff = {diff:.2e}")

    # Embedded 20f/2n, D=40
    m, meta = load_embedded_model("weights/embed_20f_2n_D40_unit_L4.pt", device)
    diff = verify_model_matches(m, "weights/embed_20f_2n_D40_unit_L4.pt", "embedded")
    print(f"embedded 20f/2n D=40: diff = {diff:.2e}")
