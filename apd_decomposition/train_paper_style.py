"""Train a paper-style target model (Braun/APD toy model of compressed
computation) using OUR code, to use as a baseline for comparing APD results.

Target function: y = x + ReLU(c * x)  for random c[i] in [1, 2].
Architecture: plain model WITHOUT residual (y_hat = W_out @ ReLU(W_in @ x)),
BUT with target = x + ReLU(c*x), so the model has to learn "identity + ReLU".
Loss: MSE (L^2).
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path


class SimpleMLP(nn.Module):
    def __init__(self, n_features, n_neurons):
        super().__init__()
        self.W_in = nn.Parameter(torch.randn(n_neurons, n_features) * 0.01)
        self.W_out = nn.Parameter(torch.randn(n_features, n_neurons) * 0.01)

    def forward(self, x):
        return (self.W_out @ torch.relu(self.W_in @ x.T)).T


def train(F, n, p, coeffs, steps=10000, batch_size=2048, lr=0.003, seed=0):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleMLP(F, n).to(device)
    coeffs = coeffs.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    for step in range(steps):
        mask = (torch.rand(batch_size, F, device=device) < p).float()
        values = torch.rand(batch_size, F, device=device) * 2 - 1
        x = mask * values
        # Paper label: y = x + ReLU(c * x)  (but we'll predict only the ReLU part)
        # Target = ReLU(c*x) only (no +x residual, since our architecture has none).
        # This makes per-feature coefficients distinguish each feature's mechanism,
        # enabling APD to find per-feature components.
        y = torch.relu(coeffs * x)
        y_hat = model(x)
        loss = ((y_hat - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 2000 == 0:
            print(f"  step {step}: loss = {loss.item():.5f}")
    return model, coeffs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--F", type=int, default=20)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--p", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="weights/paper_style_20f_5n.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    coeffs = torch.rand(args.F) + 1.0   # uniform [1, 2]
    print(f"Training paper-style target: F={args.F}, n={args.n}, p={args.p}")
    model, c = train(args.F, args.n, args.p, coeffs, steps=args.steps,
                     seed=args.seed)
    sd = {
        "W_in": model.W_in.detach().cpu(),
        "W_out": model.W_out.detach().cpu(),
        "coeffs": coeffs.cpu(),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, args.out)
    print(f"Saved to {args.out}")
