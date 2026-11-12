#!/usr/bin/env python3
"""Self-coupling computational images as conductivity.

Each image: a 2-D grid updated by a local nonlinear rule whose parameters
are driven by the grid's own output statistics. A ring of images couples
through a shared conductivity field. GPU path via PyTorch (sm_120 / 5080).
"""
import math
import torch
import torch.nn.functional as F
import numpy as np

# ------------------------------- config -------------------------------------
GRID = 256
N_RING = 8
STEPS = 600
DT = 0.05
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------- update rule ---------------------------------
# State u on grid. Local Laplacian + nonlinear self-coupling + parameter field.
# du/dt = D*lap(u) - u^3 + a*u - b*conv(u, K) + noise
# a, b are the meta-parameters read from the image's own statistics.

def laplace(u):
    # 5-point stencil, periodic boundaries
    return (torch.roll(u, 1, 0) + torch.roll(u, -1, 0) +
            torch.roll(u, 1, 1) + torch.roll(u, -1, 1) - 4 * u)

def stats(u):
    # readout: mean, variance, dominant spatial frequency proxy
    mean = u.mean()
    var = u.var()
    # spectral centroid via FFT magnitude
    spec = torch.fft.fft2(u).abs()
    freq = spec.sum() / (spec.mean() + 1e-8)
    return mean, var, freq

def step(u, a, b, D=0.1, noise=0.01):
    k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                     device=u.device, dtype=u.dtype).view(1, 1, 3, 3)
    conv = F.conv2d(u.view(1, 1, GRID, GRID), k, padding=1).view(GRID, GRID)
    du = D * laplace(u) - u ** 3 + a * u - b * conv
    du = du + noise * torch.randn_like(u)
    return u + DT * du

# ------------------------------- ring ---------------------------------------
def run_ring(seed=0):
    torch.manual_seed(seed)
    # each image: independent initial condition
    u = torch.randn(N_RING, GRID, GRID, device=DEVICE) * 0.1
    # per-image meta-parameters (a, b), driven by own stats
    a = torch.full((N_RING,), 0.5, device=DEVICE)
    b = torch.full((N_RING,), 0.2, device=DEVICE)
    # conductivity: shared field = mean across ring
    history = []
    for t in range(STEPS):
        for i in range(N_RING):
            u[i] = step(u[i], a[i], b[i])
        # readout stats per image, update meta-params (strange loop)
        for i in range(N_RING):
            m, v, f = stats(u[i])
            # a seeks variance (exploration), b seeks spectral richness
            a[i] = 0.95 * a[i] + 0.05 * (0.5 + 2.0 * v.item())
            b[i] = 0.95 * b[i] + 0.05 * (0.1 + 0.01 * f.item())
        # conductivity field
        cond = u.mean(dim=0)
        if t % 50 == 0:
            history.append(cond.detach().cpu().numpy().copy())
    return history, a.cpu().numpy(), b.cpu().numpy()

# ------------------------------- main ---------------------------------------
if __name__ == "__main__":
    print(f"device: {DEVICE}")
    hist, a_final, b_final = run_ring()
    print(f"ring of {N_RING} images, {STEPS} steps, grid {GRID}x{GRID}")
    print(f"final a: {np.round(a_final, 3)}")
    print(f"final b: {np.round(b_final, 3)}")
    # save evolution strip
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = len(hist)
        fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
        for ax, h in zip(axes, hist):
            ax.imshow(h, cmap="magma")
            ax.axis("off")
        plt.tight_layout()
        plt.savefig("ring_evolution.png", dpi=120)
        print("saved ring_evolution.png")
    except ImportError:
        print("matplotlib not installed; skipping plot")
