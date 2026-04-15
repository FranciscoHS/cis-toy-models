import torch, numpy as np
W = torch.load("weights/linear_20f_2n_L4.pt", map_location="cpu")
W_in, W_out = W["W_in"], W["W_out"]
A = (W_out @ W_in).numpy()
n = 20
print(f"rank = {np.linalg.matrix_rank(A, tol=1e-4)}")
print(f"diag: mean={np.diag(A).mean():.3f} std={np.diag(A).std():.3f}")
off = A - np.diag(np.diag(A))
print(f"off-diag RMS per row: {np.sqrt((off**2).sum(1)/(n-1)).mean():.3f}")
for j in [0, 2, 13]:
    x = np.zeros(n); x[j] = 1.0
    y = A @ x
    rms_off = np.sqrt(((y**2).sum() - y[j]**2) / (n-1))
    print(f"  feat {j}: y[{j}]={y[j]:+.3f}  RMS(others)={rms_off:.3f}")
