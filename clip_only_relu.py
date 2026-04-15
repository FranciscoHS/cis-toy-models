"""
ReLU 20f/2n L4 model with W_in constrained to be non-negative
(all features in group A, W_in rank-2). Isolates "clipping" from "routing".
"""
import torch, torch.nn as nn, torch.optim as optim, numpy as np
import torch.nn.functional as F

torch.manual_seed(0)
n_feat, n_neur, p = 20, 2, 0.02
BATCH, STEPS = 2048, 10000

def gen(bs):
    mask = (torch.rand(bs, n_feat) < p).float()
    vals = torch.rand(bs, n_feat) * 2 - 1
    x = mask * vals
    return x, torch.relu(x)

class PosWinReLU(nn.Module):
    # W_in = softplus(raw) -> strictly positive entries (all features in group A),
    # but rows of W_in are independent (rank can be up to 2).
    def __init__(self):
        super().__init__()
        self.raw_in = nn.Parameter(torch.randn(n_neur, n_feat) * 0.01)
        self.W_out = nn.Parameter(torch.randn(n_feat, n_neur) * 0.01)
    @property
    def W_in(self):
        return F.softplus(self.raw_in)
    def forward(self, x):
        return (self.W_out @ torch.relu(self.W_in @ x.T)).T

m = PosWinReLU()
opt = optim.Adam(m.parameters(), lr=0.003)
sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
for step in range(STEPS):
    x, y = gen(BATCH)
    loss = ((m(x) - y).abs() ** 4).mean()
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()

def l4(model, n_batches=200):
    losses = []
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = gen(BATCH)
            yh = model(x) if model is not None else torch.zeros_like(y)
            losses.append(((yh - y).abs() ** 4).mean().item())
    return float(np.mean(losses))

loss = l4(m)
print(f"POS W_in ReLU (all group A, rank-2) L4: {loss:.6f}")

# Compare to previously measured:
print(f"  real unconstrained ReLU L4:            0.000665")
print(f"  linear rank-2 L4:                      0.000819")
print(f"  all-group-A tied-rows (rank-1) L4:     0.000920")
print(f"  naive baseline:                        0.001994")

# Inspect W_in structure.
W_in_np = m.W_in.detach().numpy()
print(f"\nW_in shape {W_in_np.shape}, min={W_in_np.min():.4f}, max={W_in_np.max():.4f}")
print(f"rank(W_in) = {np.linalg.matrix_rank(W_in_np, tol=1e-3)}")
# Diag of both-on map
W_out_np = m.W_out.detach().numpy()
A_both = W_out_np @ W_in_np
print(f"both-on diag: mean={np.diag(A_both).mean():.3f}, off RMS/row: {np.sqrt(((A_both - np.diag(np.diag(A_both)))**2).sum(1)/19).mean():.3f}")

torch.save(m.state_dict(), "weights/poswin_20f_2n_L4.pt")
