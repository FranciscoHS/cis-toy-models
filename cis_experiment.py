"""
Experiment: Can we elicit Computation in Superposition (CiS)?

Setup: 500 features, 50 neurons, p=0.02, M=0 (no mixing matrix).
Task: y_i = ReLU(x_i) for all i.
Compare L2 vs L4 loss to see if higher-order loss incentivizes CiS.

The naive baseline dedicates 50 neurons to 50 features and ignores the other 450.
A CiS solution would compute all 500 features using superposition.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

N_FEATURES = 500
N_NEURONS = 50
BATCH_SIZE = 2048
LR = 0.003
N_BATCHES = 10_000
P = 0.02


# ── Data generation ─────────────────────────────────────────────────────────

def generate_batch(batch_size, n_features=N_FEATURES, p=P, device=DEVICE):
    """Generate sparse inputs and labels y = ReLU(x). No mixing matrix."""
    mask = (torch.rand(batch_size, n_features, device=device) < p).float()
    values = torch.rand(batch_size, n_features, device=device) * 2 - 1
    x = mask * values
    y = torch.relu(x)
    return x, y


# ── Model ───────────────────────────────────────────────────────────────────

class SimpleMLP(nn.Module):
    def __init__(self, n_features=N_FEATURES, n_neurons=N_NEURONS):
        super().__init__()
        self.W_in = nn.Parameter(torch.randn(n_neurons, n_features) * 0.01)
        self.W_out = nn.Parameter(torch.randn(n_features, n_neurons) * 0.01)

    def forward(self, x):
        return (self.W_out @ torch.relu(self.W_in @ x.T)).T


# ── Training ────────────────────────────────────────────────────────────────

def train_model(loss_exponent=2, n_batches=N_BATCHES, lr=LR, return_history=False):
    model = SimpleMLP().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_batches)

    history = []
    for step in range(n_batches):
        x, y = generate_batch(BATCH_SIZE)
        y_hat = model(x)
        loss = (torch.abs(y_hat - y) ** loss_exponent).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if return_history and step % 10 == 0:
            history.append(loss.item())

    if return_history:
        return model, history
    return model


def evaluate_model(model, loss_exponent=2, n_eval=50):
    """Evaluate model. Always reports L2 loss for comparability, plus native loss."""
    model.eval()
    total_l2 = 0.0
    total_native = 0.0
    with torch.no_grad():
        for _ in range(n_eval):
            x, y = generate_batch(BATCH_SIZE)
            y_hat = model(x)
            total_l2 += ((y_hat - y) ** 2).mean().item()
            total_native += (torch.abs(y_hat - y) ** loss_exponent).mean().item()
    model.eval()  # redundant but safe
    model.train()
    return total_l2 / n_eval, total_native / n_eval


def compute_naive_loss(loss_exponent=2, n_eval=100):
    """Naive: perfectly compute first N_NEURONS features, output 0 for rest."""
    total_l2 = 0.0
    total_native = 0.0
    for _ in range(n_eval):
        x, y = generate_batch(BATCH_SIZE)
        y_hat = y.clone()
        y_hat[:, N_NEURONS:] = 0.0
        total_l2 += ((y_hat - y) ** 2).mean().item()
        total_native += (torch.abs(y_hat - y) ** loss_exponent).mean().item()
    return total_l2 / n_eval, total_native / n_eval


# ── Analysis ────────────────────────────────────────────────────────────────

def analyze_feature_coverage(model, label=""):
    """Check how many features the model actually computes well."""
    model.eval()
    per_feature_l2 = torch.zeros(N_FEATURES, device=DEVICE)
    count = 0
    with torch.no_grad():
        for _ in range(100):
            x, y = generate_batch(BATCH_SIZE)
            errors = (model(x) - y) ** 2
            # Only count when feature is active (y > 0 means x > 0)
            active = (x > 0).float()
            per_feature_l2 += (errors * active).sum(dim=0)
            count += active.sum(dim=0)

    per_feature_l2 = (per_feature_l2 / count.clamp(min=1)).cpu().numpy()
    model.train()

    # Sort features by error
    sorted_errors = np.sort(per_feature_l2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: per-feature error (sorted)
    axes[0].plot(sorted_errors, 'o', markersize=2)
    axes[0].axhline(y=1/6, color='r', linestyle='--', alpha=0.7,
                     label='E[ReLU(x)²] = 1/6 (complete ignorance)')
    axes[0].axvline(x=N_NEURONS, color='gray', linestyle=':', alpha=0.7,
                     label=f'n_neurons={N_NEURONS}')
    axes[0].set_xlabel("Feature index (sorted by error)")
    axes[0].set_ylabel("MSE when feature is active")
    axes[0].set_title(f"Per-feature error ({label})")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Right: histogram of errors
    axes[1].hist(per_feature_l2, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=1/6, color='r', linestyle='--', alpha=0.7,
                     label='Complete ignorance')
    axes[1].set_xlabel("MSE when feature is active")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Distribution of per-feature errors ({label})")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    safe_label = label.replace(" ", "_").replace("=", "").replace("^", "")
    fig.savefig(FIGURES_DIR / f"cis_feature_coverage_{safe_label}.png", dpi=150)
    plt.close(fig)

    # Summary stats
    n_well_computed = np.sum(per_feature_l2 < 0.05)  # somewhat arbitrary threshold
    n_ignored = np.sum(per_feature_l2 > 0.12)  # close to 1/6
    print(f"  {label}: {n_well_computed} features well-computed, "
          f"{n_ignored} effectively ignored, "
          f"mean error = {per_feature_l2.mean():.6f}")

    return per_feature_l2


# ── Main experiment ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    print(f"Device: {DEVICE}")
    print(f"Setup: {N_FEATURES} features, {N_NEURONS} neurons, p={P}")
    print(f"E[active features per input] = {N_FEATURES * P}")
    print()

    # Naive baseline
    print("Computing naive baseline...")
    naive_l2, _ = compute_naive_loss(loss_exponent=2)
    naive_l2_4, naive_l4 = compute_naive_loss(loss_exponent=4)
    print(f"  Naive L2 loss: {naive_l2:.6f}")
    print(f"  Naive L4 loss: {naive_l4:.6f}")
    print()

    # Train with L2
    print("Training with L2 loss...")
    model_l2, history_l2 = train_model(loss_exponent=2, return_history=True)
    l2_l2, _ = evaluate_model(model_l2, loss_exponent=2)
    print(f"  L2 model - L2 loss: {l2_l2:.6f} (L2/p = {l2_l2/P:.6f})")
    print(f"  vs naive: {'BEATS' if l2_l2 < naive_l2 else 'DOES NOT BEAT'} naive")
    print()

    # Train with L4
    print("Training with L4 loss...")
    model_l4, history_l4 = train_model(loss_exponent=4, return_history=True)
    l4_l2, l4_l4 = evaluate_model(model_l4, loss_exponent=4)
    print(f"  L4 model - L2 loss: {l4_l2:.6f} (L2/p = {l4_l2/P:.6f})")
    print(f"  L4 model - L4 loss: {l4_l4:.6f}")
    print(f"  vs naive (L2): {'BEATS' if l4_l2 < naive_l2 else 'DOES NOT BEAT'} naive")
    print(f"  vs naive (L4): {'BEATS' if l4_l4 < naive_l4 else 'DOES NOT BEAT'} naive")
    print()

    # Feature coverage analysis
    print("Analyzing feature coverage...")
    errors_l2 = analyze_feature_coverage(model_l2, label="L2 loss")
    errors_l4 = analyze_feature_coverage(model_l4, label="L4 loss")

    # Comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Training curves (both in their native loss)
    axes[0].plot(np.arange(len(history_l2)) * 10, history_l2, alpha=0.7, label='L2 training loss')
    axes[0].plot(np.arange(len(history_l4)) * 10, history_l4, alpha=0.7, label='L4 training loss')
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training curves")
    axes[0].set_yscale('log')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Per-feature error comparison (sorted independently)
    axes[1].plot(np.sort(errors_l2), 'o', markersize=2, label='L2-trained', alpha=0.7)
    axes[1].plot(np.sort(errors_l4), 'o', markersize=2, label='L4-trained', alpha=0.7)
    axes[1].axhline(y=1/6, color='r', linestyle='--', alpha=0.5,
                     label='Complete ignorance (1/6)')
    axes[1].axvline(x=N_NEURONS, color='gray', linestyle=':', alpha=0.5,
                     label=f'n_neurons={N_NEURONS}')
    axes[1].set_xlabel("Feature index (sorted by error)")
    axes[1].set_ylabel("MSE when feature is active")
    axes[1].set_title(f"Per-feature error: L2 vs L4\n({N_FEATURES} features, {N_NEURONS} neurons, p={P})")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cis_l2_vs_l4_comparison.png", dpi=150)
    plt.close(fig)
    print("\n  -> Saved cis_l2_vs_l4_comparison.png")

    # Save weights
    WEIGHTS_DIR = Path("weights")
    WEIGHTS_DIR.mkdir(exist_ok=True)
    torch.save(model_l2.state_dict(), WEIGHTS_DIR / "model_l2.pt")
    torch.save(model_l4.state_dict(), WEIGHTS_DIR / "model_l4.pt")
    print(f"\n  -> Saved weights to {WEIGHTS_DIR}/")

    print("\nDone!")
