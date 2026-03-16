"""
Replication of "Compressed Computation is (probably) not Computation in Superposition"
by Bhagat, Molas Medina, Giglemiani, StefanHex (LessWrong, June 2025).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import math
import json
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIGURES_DIR = Path("figures")
DATA_DIR = Path("data")
FIGURES_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

N_FEATURES = 100
N_NEURONS = 50
BATCH_SIZE = 2048
LR = 0.003
N_BATCHES = 10_000


def save_results(name, data):
    """Save experiment results as numpy/json."""
    path = DATA_DIR / f"{name}.npz"
    np.savez(path, **{k: np.array(v) for k, v in data.items()})
    print(f"  -> Saved data to {path}")


def save_model(name, model, M):
    """Save model weights and mixing matrix."""
    path = DATA_DIR / f"{name}_model.pt"
    torch.save({
        'W_in': model.W_in.detach().cpu(),
        'W_out': model.W_out.detach().cpu(),
        'M': M,
    }, path)


# ── Mixing matrix constructors ──────────────────────────────────────────────

def make_embedding_M(n_features=N_FEATURES, d_resid=1000):
    """M = I - W_E W_E^T where W_E has random unit-norm rows (Braun et al.)."""
    W_E = torch.randn(n_features, d_resid)
    W_E = W_E / W_E.norm(dim=1, keepdim=True)
    M = torch.eye(n_features) - W_E @ W_E.T
    return M


def make_symmetric_random_M(n_features=N_FEATURES, sigma=0.032):
    M = torch.randn(n_features, n_features) * sigma
    M = (M + M.T) / math.sqrt(2)
    return M


def make_asymmetric_random_M(n_features=N_FEATURES, sigma=0.023):
    return torch.randn(n_features, n_features) * sigma


def make_clean_M(n_features=N_FEATURES):
    return torch.zeros(n_features, n_features)


# ── Data generation ─────────────────────────────────────────────────────────

def generate_batch(batch_size, n_features, p, M, device=DEVICE, maximally_sparse=False):
    if maximally_sparse:
        x = torch.zeros(batch_size, n_features, device=device)
        indices = torch.randint(0, n_features, (batch_size,), device=device)
        values = torch.rand(batch_size, device=device) * 2 - 1
        x[torch.arange(batch_size, device=device), indices] = values
    else:
        mask = (torch.rand(batch_size, n_features, device=device) < p).float()
        values = torch.rand(batch_size, n_features, device=device) * 2 - 1
        x = mask * values

    M_dev = M.to(device)
    y = torch.relu(x) + x @ M_dev.T
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

def train_model(M, p, n_batches=N_BATCHES, batch_size=BATCH_SIZE, lr=LR,
                maximally_sparse=False, return_history=False, model=None,
                optimizer=None):
    if model is None:
        model = SimpleMLP().to(DEVICE)
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_batches)

    history = []
    batch_count = 0
    while batch_count < n_batches:
        x, y = generate_batch(batch_size, N_FEATURES, p, M,
                              maximally_sparse=maximally_sparse)
        if x.abs().sum() == 0:
            continue

        y_hat = model(x)
        loss = ((y_hat - y) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        batch_count += 1

        if return_history:
            history.append(loss.item())

    if return_history:
        return model, optimizer, history
    return model


def evaluate_model(model, M, p, n_eval=50, batch_size=BATCH_SIZE, maximally_sparse=False):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for _ in range(n_eval):
            x, y = generate_batch(batch_size, N_FEATURES, p, M,
                                  maximally_sparse=maximally_sparse)
            if x.abs().sum() == 0:
                continue
            y_hat = model(x)
            total_loss += ((y_hat - y) ** 2).mean().item()
            count += 1
    model.train()
    return total_loss / max(count, 1)


def compute_naive_loss(p, M, maximally_sparse=False):
    """Naive: perfectly compute top N_NEURONS features, output 0 for rest."""
    n_eval = 100
    total_loss = 0.0
    for _ in range(n_eval):
        x, y = generate_batch(BATCH_SIZE, N_FEATURES, p, M,
                              maximally_sparse=maximally_sparse)
        y_hat = y.clone()
        y_hat[:, N_NEURONS:] = 0.0
        total_loss += ((y_hat - y) ** 2).mean().item()
    return total_loss / n_eval


# ── Experiment 1: Loss vs eval sparsity (Figure 2) ─────────────────────────

def exp_loss_vs_sparsity():
    print("=" * 60)
    print("Exp 1: Loss vs eval sparsity (Figure 2)")
    print("=" * 60)

    M = make_symmetric_random_M(sigma=0.02)

    p_trains = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    p_evals = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

    all_losses = {}
    for p_train in p_trains:
        ms = (p_train <= 0.01)
        print(f"  Training at p={p_train} (maximally_sparse={ms})")
        t0 = time.time()
        model = train_model(M, p_train, maximally_sparse=ms)
        print(f"    trained in {time.time()-t0:.1f}s")

        losses = []
        for p_eval in p_evals:
            ms_eval = (p_eval <= 0.01)
            loss = evaluate_model(model, M, p_eval, maximally_sparse=ms_eval)
            losses.append(loss / p_eval)
        all_losses[str(p_train)] = losses

    # Naive baseline
    naive_losses = []
    for p_eval in p_evals:
        ms_eval = (p_eval <= 0.01)
        nl = compute_naive_loss(p_eval, M, maximally_sparse=ms_eval)
        naive_losses.append(nl / p_eval)

    save_results("exp1_loss_vs_sparsity", {
        'p_trains': p_trains, 'p_evals': p_evals,
        'naive_losses': naive_losses,
        **{f'losses_p{p}': v for p, v in all_losses.items()},
    })

    # Plot
    cmap = plt.cm.viridis
    colors = [cmap(i / (len(p_trains) - 1)) for i in range(len(p_trains))]
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    for i, p_train in enumerate(p_trains):
        ax.plot(p_evals, all_losses[str(p_train)], '-',
                label=f'p={p_train}', color=colors[i])

    # Black dots: eval at training p
    train_at_train = [all_losses[str(p)][p_evals.index(p)] for p in p_trains]
    ax.plot(p_trains, train_at_train, 'ko--', label='p_eval = p_train',
            markersize=6, zorder=10)

    ax.plot(p_evals, naive_losses, 'k--', label="Naive loss", linewidth=2)
    ax.set_xscale('log')
    ax.set_xlabel("Feature probability p")
    ax.set_ylabel("Loss per feature L/p")
    ax.set_title("Loss over input sparsity for different training probabilities p (σ=0.02)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_loss_vs_sparsity.png", dpi=150)
    plt.close(fig)
    print("  -> Saved fig1_loss_vs_sparsity.png\n")


# ── Experiment 2: Loss vs sparsity for different M types (Figure 5a) ───────

def exp_M_type_comparison():
    print("=" * 60)
    print("Exp 2: M-type comparison (Figure 5a)")
    print("=" * 60)

    M_embed = make_embedding_M()
    M_sym = make_symmetric_random_M(sigma=0.032)
    M_asym = make_asymmetric_random_M(sigma=0.023)
    M_clean = make_clean_M()

    print(f"  Embedding M std: {M_embed.std():.4f}")

    configs = [
        (f"Embedding M (std={M_embed.std():.3f})", M_embed, "tab:blue"),
        ("Symmetric M (σ=0.032)", M_sym, "tab:red"),
        ("Asymmetric M (σ=0.023)", M_asym, "tab:green"),
        ("No mixing M=0", M_clean, "tab:orange"),
    ]

    ps = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    results = {}

    for name, M, color in configs:
        losses = []
        for p in ps:
            ms = (p <= 0.01)
            print(f"  Training: {name}, p={p}")
            t0 = time.time()
            model = train_model(M, p, maximally_sparse=ms)
            loss = evaluate_model(model, M, p, maximally_sparse=ms)
            losses.append(loss / p)
            print(f"    {time.time()-t0:.1f}s, L/p={losses[-1]:.5f}")
        ax.plot(ps, losses, 'o-', label=name, color=color)
        results[name] = losses

    # Naive baseline (computed with embedding M)
    naive_losses = []
    for p in ps:
        ms = (p <= 0.01)
        nl = compute_naive_loss(p, M_embed, maximally_sparse=ms)
        naive_losses.append(nl / p)
    ax.plot(ps, naive_losses, 'k--', label="Naive loss", linewidth=2)

    save_results("exp2_M_type_comparison", {
        'ps': ps, 'naive_losses': naive_losses,
        **{f'losses_{i}': v for i, (name, _, _) in enumerate(configs) for v in [results[name]]},
        'config_names': [n for n, _, _ in configs],
    })

    ax.set_xscale('log')
    ax.set_xlabel("Feature probability p")
    ax.set_ylabel("Loss per feature L/p")
    ax.set_title("Loss over input sparsity for different mixing matrices M")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_M_type_comparison.png", dpi=150)
    plt.close(fig)
    print("  -> Saved fig2_M_type_comparison.png\n")


# ── Experiment 3: Loss vs sigma (Figure 5b) ─────────────────────────────────

def exp_loss_vs_sigma():
    print("=" * 60)
    print("Exp 3: Loss vs σ (Figure 5b)")
    print("=" * 60)

    sigmas = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.08]
    p = 0.01

    sym_losses = []
    asym_losses = []

    for sigma in sigmas:
        print(f"  σ={sigma}")
        t0 = time.time()

        M_sym = make_symmetric_random_M(sigma=sigma) if sigma > 0 else make_clean_M()
        model = train_model(M_sym, p, maximally_sparse=True)
        loss = evaluate_model(model, M_sym, p, maximally_sparse=True)
        sym_losses.append(loss / p)

        M_asym = make_asymmetric_random_M(sigma=sigma) if sigma > 0 else make_clean_M()
        model = train_model(M_asym, p, maximally_sparse=True)
        loss = evaluate_model(model, M_asym, p, maximally_sparse=True)
        asym_losses.append(loss / p)

        print(f"    {time.time()-t0:.1f}s, sym={sym_losses[-1]:.5f}, asym={asym_losses[-1]:.5f}")

    naive = compute_naive_loss(p, make_clean_M(), maximally_sparse=True) / p

    save_results("exp3_loss_vs_sigma", {
        'sigmas': sigmas, 'sym_losses': sym_losses,
        'asym_losses': asym_losses, 'naive': [naive],
    })

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(sigmas, sym_losses, 'o-', color='tab:green', label='Symmetric noise')
    ax.plot(sigmas, asym_losses, 'o-', color='tab:orange', label='Asymmetric noise')
    ax.axhline(naive, color='k', linestyle='--', label='Naive loss')
    ax.set_xlabel("Mixing matrix magnitude σ")
    ax.set_ylabel("Loss per feature L/p")
    ax.set_title("Loss over a range of mixing matrix magnitudes")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_loss_vs_sigma.png", dpi=150)
    plt.close(fig)
    print("  -> Saved fig3_loss_vs_sigma.png\n")


# ── Experiment 4: Transplant (Figure 6) ─────────────────────────────────────

def exp_transplant():
    print("=" * 60)
    print("Exp 4: Transplant (Figure 6)")
    print("=" * 60)

    M_noisy = make_symmetric_random_M(sigma=0.025)
    M_clean = make_clean_M()
    p = 0.01

    print("  Phase 1: Training on noisy dataset (10k steps)...")
    t0 = time.time()
    model, optimizer, history_noisy = train_model(
        M_noisy, p, n_batches=10000, maximally_sparse=True, return_history=True)
    print(f"    {time.time()-t0:.1f}s")

    print("  Phase 2: Fine-tuning on clean dataset (10k steps, same optimizer)...")
    t0 = time.time()
    _, _, history_clean = train_model(
        M_clean, p, n_batches=10000, maximally_sparse=True, return_history=True,
        model=model, optimizer=optimizer)
    print(f"    {time.time()-t0:.1f}s")

    save_results("exp4_transplant", {
        'history_noisy': history_noisy, 'history_clean': history_clean, 'p': [p],
    })

    def smooth(arr, window=100):
        return np.convolve(arr, np.ones(window) / window, mode='valid')

    h1 = smooth(history_noisy)
    h2 = smooth(history_clean)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(h1)), h1 / p, color='tab:blue', label='Train with M ~ N(0, 0.025²)')
    ax.plot(range(len(h1), len(h1) + len(h2)), h2 / p, color='tab:orange',
            label='Fine-tune with M=0')
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss per feature L/p (running average)")
    ax.set_yscale('log')
    ax.set_title("Transplanting weights from noisy case to clean case")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_transplant.png", dpi=150)
    plt.close(fig)
    print("  -> Saved fig4_transplant.png\n")


# ── Experiment 5: SNMF (Figure 9) ──────────────────────────────────────────

def semi_nmf(A, k, n_iter=200):
    n, m = A.shape
    W_out = np.random.randn(n, k) * 0.01
    W_in = np.abs(np.random.randn(k, m)) * 0.01
    for _ in range(n_iter):
        W_out = A @ W_in.T @ np.linalg.pinv(W_in @ W_in.T)
        W_in = np.linalg.pinv(W_out.T @ W_out) @ W_out.T @ A
        W_in = np.maximum(W_in, 0)
    return W_out, W_in


def exp_snmf_solution():
    print("=" * 60)
    print("Exp 5: SNMF analytical solution (Figure 9)")
    print("=" * 60)

    sigmas = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]
    p = 0.01

    snmf_losses = []
    for sigma in sigmas:
        print(f"  SNMF σ={sigma}")
        M = make_symmetric_random_M(sigma=sigma) if sigma > 0 else make_clean_M()
        target = M.numpy() + np.eye(N_FEATURES)
        W_out_snmf, W_in_snmf = semi_nmf(target, N_NEURONS)

        model = SimpleMLP().to(DEVICE)
        with torch.no_grad():
            model.W_in.copy_(torch.tensor(W_in_snmf, dtype=torch.float32))
            model.W_out.copy_(torch.tensor(W_out_snmf, dtype=torch.float32))

        loss = evaluate_model(model, M, p, maximally_sparse=True)
        snmf_losses.append(loss / p)

    naive = compute_naive_loss(p, make_clean_M(), maximally_sparse=True) / p

    save_results("exp5_snmf", {
        'sigmas': sigmas, 'snmf_losses': snmf_losses, 'naive': [naive],
    })

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(sigmas, snmf_losses, 'o-', color='tab:green', label='Semi-NMF solution')
    ax.axhline(naive, color='k', linestyle='--', label='Naive loss')
    ax.set_xlabel("Mixing matrix magnitude σ")
    ax.set_ylabel("Loss per feature L/p")
    ax.set_title("SNMF solution vs naive baseline")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_snmf_solution.png", dpi=150)
    plt.close(fig)
    print("  -> Saved fig5_snmf_solution.png\n")


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    print(f"Using device: {DEVICE}")
    t_start = time.time()

    exp_loss_vs_sparsity()      # Fig 2:  10 models
    exp_M_type_comparison()     # Fig 5a: 28 models
    exp_loss_vs_sigma()         # Fig 5b: 22 models
    exp_transplant()            # Fig 6:  1 model (2 phases)
    exp_snmf_solution()         # Fig 9:  0 trained (analytical)

    print(f"All done in {time.time()-t_start:.0f}s. Figures in {FIGURES_DIR}/, data in {DATA_DIR}/")
