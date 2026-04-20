"""
Step 2: CiS model with fixed random embedding/unembedding.

Architecture:
    r_in  = E @ x         (D-dim, E is D x F fixed random)
    z     = W_in @ r_in   (n-dim)
    h     = ReLU(z)
    r_out = W_out @ h     (D-dim)
    y     = E^T @ r_out   (F-dim)

Trainable: W_in (n x D), W_out (D x n). E is fixed.

Note: if D >= F and E has full column rank (true w.p. 1 for Gaussian), this is
gauge-equivalent to the no-E model: W_in @ E can be any (n x F) matrix and
E^T @ W_out can be any (F x n) matrix. So in the absence of other constraints,
the embedding is a rotation/change-of-basis. The purpose here is to (i) verify
that L^4 still finds the CiS solution (in rotated coordinates), (ii) check
whether the mechanism (codeword sign patterns in feature basis, rank-n
projection) is preserved or whether a different solution is reached.

Run at F=20, n=2 first (smallest non-trivial) with D=F (square E, random
orthogonal) and D=2F (overcomplete, random unit-norm rows).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_DIR = Path("weights")
WEIGHTS_DIR.mkdir(exist_ok=True)


def make_embedding(F, D, kind="gaussian_unit_rows", seed=0):
    """Return an F x D matrix E (so input embedding is E^T applied to x,
    or equivalently, we read cols of E as the embeddings of features).

    Convention: E_{d, j} = component d of feature j's embedding vector.
    So E has shape D x F, and r = E @ x gives the D-dim residual.
    """
    g = torch.Generator().manual_seed(seed)
    if kind == "orthogonal":
        # For D == F: random orthogonal matrix (D x F, with D = F)
        assert D == F
        M = torch.randn(F, F, generator=g)
        Q, _ = torch.linalg.qr(M)
        return Q  # D x F, columns are orthonormal basis of R^F
    elif kind == "gaussian_unit_cols":
        # D x F, each column (= feature embedding) is unit-norm
        M = torch.randn(D, F, generator=g)
        M = M / M.norm(dim=0, keepdim=True)
        return M
    elif kind == "gaussian":
        # D x F, entries iid N(0, 1/D) so E x has roughly unit variance per dim
        M = torch.randn(D, F, generator=g) / (D ** 0.5)
        return M
    else:
        raise ValueError(kind)


class EmbeddedMLP(nn.Module):
    def __init__(self, F, n, D, E):
        super().__init__()
        self.F, self.n, self.D = F, n, D
        self.register_buffer("E", E.clone())            # D x F
        self.W_in = nn.Parameter(torch.randn(n, D) * 0.01)
        self.W_out = nn.Parameter(torch.randn(D, n) * 0.01)

    def forward(self, x):
        # x: (batch, F) -> r: (batch, D)
        r = x @ self.E.T                                 # (B, D)
        z = r @ self.W_in.T                              # (B, n)
        h = torch.relu(z)
        r_out = h @ self.W_out.T                         # (B, D)
        y = r_out @ self.E                                # (B, F)  [= r_out @ E, since E is D x F]
        return y


def generate_batch(batch_size, F, p, device=DEVICE):
    mask = (torch.rand(batch_size, F, device=device) < p).float()
    values = torch.rand(batch_size, F, device=device) * 2 - 1
    x = mask * values
    y = torch.relu(x)
    return x, y


def train(F, n, D, E_kind, loss_exp, p, n_batches=10000, seed=0, lr=0.003, verbose=False):
    torch.manual_seed(seed)
    E = make_embedding(F, D, kind=E_kind, seed=seed + 100)
    model = EmbeddedMLP(F, n, D, E).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_batches)
    history = []
    for step in range(n_batches):
        x, y = generate_batch(2048, F, p)
        y_hat = model(x)
        loss = (torch.abs(y_hat - y) ** loss_exp).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if verbose and step % 1000 == 0:
            history.append((step, loss.item()))
    return model, E


def evaluate_per_feature(model, F, p, n_eval=100):
    model.eval()
    per_feat = torch.zeros(F, device=DEVICE)
    count = torch.zeros(F, device=DEVICE)
    with torch.no_grad():
        for _ in range(n_eval):
            x, y = generate_batch(2048, F, p)
            active = (x > 0).float()
            err = ((model(x) - y) ** 2 * active).sum(dim=0)
            per_feat += err
            count += active.sum(dim=0)
    return (per_feat / count.clamp(min=1)).cpu().numpy()


def effective_weights(model):
    """Compute effective W_in_eff = W_in @ E (n x F) and W_out_eff = E^T @ W_out (F x n).
    These are what correspond to the plain-model weights under gauge absorption.
    """
    W_in_eff = model.W_in.detach().cpu() @ model.E.detach().cpu()          # (n, F)
    W_out_eff = model.E.detach().cpu().T @ model.W_out.detach().cpu()      # (F, n)
    return W_in_eff.numpy(), W_out_eff.numpy()


if __name__ == "__main__":
    np.random.seed(42)
    print(f"Device: {DEVICE}")

    configs = [
        # (F, n, D, E_kind, loss_exp, p, tag)
        (20, 2, 20, "orthogonal", 2, 0.02, "embed_20f_2n_D20_orth_L2"),
        (20, 2, 20, "orthogonal", 4, 0.02, "embed_20f_2n_D20_orth_L4"),
        (20, 2, 40, "gaussian_unit_cols", 2, 0.02, "embed_20f_2n_D40_unit_L2"),
        (20, 2, 40, "gaussian_unit_cols", 4, 0.02, "embed_20f_2n_D40_unit_L4"),
        (20, 2, 80, "gaussian_unit_cols", 4, 0.02, "embed_20f_2n_D80_unit_L4"),
        (20, 5, 20, "orthogonal", 4, 0.02, "embed_20f_5n_D20_orth_L4"),
        (20, 5, 40, "gaussian_unit_cols", 4, 0.02, "embed_20f_5n_D40_unit_L4"),
        (20, 5, 80, "gaussian_unit_cols", 4, 0.02, "embed_20f_5n_D80_unit_L4"),
        (20, 5, 200, "gaussian_unit_cols", 4, 0.02, "embed_20f_5n_D200_unit_L4"),
    ]

    for F, n, D, E_kind, loss_exp, p, tag in configs:
        path = WEIGHTS_DIR / f"{tag}.pt"
        if path.exists():
            print(f"[skip] {path}")
            continue
        print(f"Training {tag}: F={F}, n={n}, D={D}, E={E_kind}, L{loss_exp}, p={p}")
        model, E = train(F, n, D, E_kind, loss_exp, p)
        torch.save({
            "W_in": model.W_in.detach().cpu(),
            "W_out": model.W_out.detach().cpu(),
            "E": model.E.detach().cpu(),
            "meta": dict(F=F, n=n, D=D, E_kind=E_kind, loss_exp=loss_exp, p=p),
        }, path)
        per_feat = evaluate_per_feature(model, F, p)
        print(f"  per-feat MSE: mean={per_feat.mean():.5f}, "
              f"min={per_feat.min():.5f}, max={per_feat.max():.5f}")
        # compare to ignorance (1/6) and naive
        n_well = (per_feat < 0.05).sum()
        n_ignored = (per_feat > 0.12).sum()
        print(f"  {n_well}/{F} well-computed, {n_ignored}/{F} effectively ignored")
    print("Done")
