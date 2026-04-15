"""
Compare:
 - REAL 20f/2n L4 model (rank-2 W_in)
 - All-group-A model: W_in constrained to have identical rows -> rank-1
   (then W_out is free; train on L4)
 - Naive baseline (y_hat = 0)
Report L4 loss AND check single-feature-active off-diagonal outputs.
"""
import torch, torch.nn as nn, torch.optim as optim, numpy as np

torch.manual_seed(0)
n_feat, n_neur, p = 20, 2, 0.02
BATCH, STEPS = 2048, 10000

def gen(bs):
    mask = (torch.rand(bs, n_feat) < p).float()
    vals = torch.rand(bs, n_feat) * 2 - 1
    x = mask * vals
    return x, torch.relu(x)

class AllGroupA(nn.Module):
    # W_in has both rows equal (tied) -> W_in is rank 1, sign pattern (+,+) on every feature.
    def __init__(self):
        super().__init__()
        self.row = nn.Parameter(torch.randn(n_feat) * 0.01)  # shared row
        self.W_out = nn.Parameter(torch.randn(n_feat, n_neur) * 0.01)
    def forward(self, x):
        W_in = self.row.unsqueeze(0).expand(n_neur, -1)  # (2, 20), both rows equal
        return (self.W_out @ torch.relu(W_in @ x.T)).T

def l4(model, n_batches=200):
    losses = []
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = gen(BATCH)
            yh = model(x) if model is not None else torch.zeros_like(y)
            losses.append(((yh - y).abs() ** 4).mean().item())
    return float(np.mean(losses))

# Train the all-group-A model
m = AllGroupA()
opt = optim.Adam(m.parameters(), lr=0.003)
sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
for step in range(STEPS):
    x, y = gen(BATCH)
    loss = ((m(x) - y).abs() ** 4).mean()
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
print(f"All-group-A (rank-1 W_in) final L4: {l4(m):.6f}")

# Real model
W = torch.load("weights/small_20f_2n_L4.pt", map_location="cpu")
class Real(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_in = nn.Parameter(W["W_in"])
        self.W_out = nn.Parameter(W["W_out"])
    def forward(self, x):
        return (self.W_out @ torch.relu(self.W_in @ x.T)).T
real = Real()
print(f"REAL (rank-2 W_in) L4:              {l4(real):.6f}")
print(f"Naive (y_hat=0) L4:                 {l4(None):.6f}")

# Off-diagonal check on both.
print("\nSingle-feature positive probe, x = +1.0 on feat j:")
for j in [0, 2, 13]:
    x = torch.zeros(1, n_feat); x[0, j] = 1.0
    y_real = real(x).detach().numpy()[0]
    y_A = m(x).detach().numpy()[0]
    rms_off = lambda y: np.sqrt(((y**2).sum() - y[j]**2) / (n_feat - 1))
    print(f"  feat {j:>2}: REAL y[{j}]={y_real[j]:+.3f} RMS(others)={rms_off(y_real):.3f}  |"
          f"  all-A y[{j}]={y_A[j]:+.3f} RMS(others)={rms_off(y_A):.3f}")

# Print what the all-group-A model learned for c.
print(f"\nAll-group-A: row values: mean={m.row.mean().item():.3f} std={m.row.std().item():.4f}")
print(f"  (If uniform, rank-1 both-on has all entries c^2, so c≈{(m.row.mean().item()**2):.3f} per diag)")
