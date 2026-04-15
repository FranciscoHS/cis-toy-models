"""
Repeat the clipping-vs-routing test at 100f/10n.
"""
import torch, torch.nn as nn, torch.optim as optim, numpy as np
import torch.nn.functional as F

torch.manual_seed(0)
n_feat, n_neur, p = 100, 10, 0.02
BATCH, STEPS = 2048, 10000

def gen(bs):
    mask = (torch.rand(bs, n_feat) < p).float()
    vals = torch.rand(bs, n_feat) * 2 - 1
    x = mask * vals
    return x, torch.relu(x)

class PosWinReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_in = nn.Parameter(torch.randn(n_neur, n_feat) * 0.01)
        self.W_out = nn.Parameter(torch.randn(n_feat, n_neur) * 0.01)
    @property
    def W_in(self):
        return F.softplus(self.raw_in)
    def forward(self, x):
        return (self.W_out @ torch.relu(self.W_in @ x.T)).T

class ReLUMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_in = nn.Parameter(torch.randn(n_neur, n_feat) * 0.01)
        self.W_out = nn.Parameter(torch.randn(n_feat, n_neur) * 0.01)
    def forward(self, x):
        return (self.W_out @ torch.relu(self.W_in @ x.T)).T

class LinearMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_in = nn.Parameter(torch.randn(n_neur, n_feat) * 0.01)
        self.W_out = nn.Parameter(torch.randn(n_feat, n_neur) * 0.01)
    def forward(self, x):
        return (self.W_out @ (self.W_in @ x.T)).T

def l4(model, n_batches=200):
    losses = []
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = gen(BATCH)
            yh = model(x) if model is not None else torch.zeros_like(y)
            losses.append(((yh - y).abs() ** 4).mean().item())
    return float(np.mean(losses))

def train_fresh(model):
    opt = optim.Adam(model.parameters(), lr=0.003)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    for step in range(STEPS):
        x, y = gen(BATCH)
        loss = ((model(x) - y).abs() ** 4).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return model

# Load existing ReLU and linear models, evaluate
relu = ReLUMLP(); relu.load_state_dict(torch.load("weights/small_100f_10n_L4.pt", map_location="cpu"))
lin = LinearMLP(); lin.load_state_dict(torch.load("weights/linear_100f_10n_L4.pt", map_location="cpu"))

print("Existing models:")
print(f"  unconstrained ReLU L4: {l4(relu):.6f}")
print(f"  linear L4:             {l4(lin):.6f}")
print(f"  naive baseline L4:     {l4(None):.6f}")

# Train positive-W_in ReLU
print("\nTraining PosW_in ReLU (100f/10n, all-group-A with rank-10)...")
pos = train_fresh(PosWinReLU())
print(f"  POS W_in ReLU L4:      {l4(pos):.6f}")

W_in_np = pos.W_in.detach().numpy()
print(f"\nPosW_in: rank={np.linalg.matrix_rank(W_in_np, tol=1e-3)}, min={W_in_np.min():.4f}, max={W_in_np.max():.4f}")
torch.save(pos.state_dict(), "weights/poswin_100f_10n_L4.pt")
