#!/usr/bin/env python3
"""MNIST via self-coupled computational image as feature extractor.

Compares THREE conditions on equal footing:
  1. Raw pixels + linear head (no dynamics)
  2. Single image evolved under self-coupling rule + linear head
  3. Ring of N images (conductivity = shared mean field) + linear head

The ablation: does the ring's shared conductivity field carry more
class-relevant information than a single evolved image?
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import json, time

GRID = 64
T_STEPS = 40
DT = 0.08
N_RING = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------- dynamics -----------------------------------
def evolve_single(u, a=0.8, b=0.15, D=0.2, steps=T_STEPS):
    """Evolve one image grid under the self-coupling rule."""
    k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                     device=u.device, dtype=u.dtype).view(1, 1, 3, 3)
    for _ in range(steps):
        conv = F.conv2d(u, k, padding=1)
        du = (D * (torch.roll(u, 1, 2) + torch.roll(u, -1, 2) +
                   torch.roll(u, 1, 3) + torch.roll(u, -1, 3) - 4 * u)
              - u ** 3 + a * u - b * conv)
        u = u + DT * du
    return u

def evolve_ring(u, a=0.8, b=0.15, D=0.2, steps=T_STEPS):
    """Evolve N images; each step, conductivity = mean across ring feeds back."""
    k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                     device=u.device, dtype=u.dtype).view(1, 1, 3, 3)
    for _ in range(steps):
        # each image evolves with shared conductivity as extra coupling term
        cond = u.mean(dim=0, keepdim=True)  # (1,1,H,W)
        conv = F.conv2d(u, k, padding=1)
        cond_conv = F.conv2d(cond, k, padding=1)
        du = (D * (torch.roll(u, 1, 2) + torch.roll(u, -1, 2) +
                   torch.roll(u, 1, 3) + torch.roll(u, -1, 3) - 4 * u)
              - u ** 3 + a * u - b * conv
              + 0.3 * (cond - u)          # pull toward shared field
              - 0.1 * cond_conv)           # conductivity's own diffusion
        u = u + DT * du
    return u

# ------------------------------- features -----------------------------------
def feats_single(u):
    f = []
    f.append(u.mean(dim=(2, 3))); f.append(u.var(dim=(2, 3)))
    f.append(u.amax(dim=(2, 3)).values); f.append(u.amin(dim=(2, 3)).values)
    h, w = u.shape[2], u.shape[3]
    f.append(u[:, :, :h//2, :w//2].mean(dim=(2, 3)))
    f.append(u[:, :, :h//2, w//2:].mean(dim=(2, 3)))
    f.append(u[:, :, h//2:, :w//2].mean(dim=(2, 3)))
    f.append(u[:, :, h//2:, w//2:].mean(dim=(2, 3)))
    spec = torch.fft.fft2(u).abs()
    f.append(spec.mean(dim=(2, 3))); f.append(spec.amax(dim=(2, 3)).values)
    return torch.stack(f, dim=1)  # (B, 10)

def feats_ring(u):
    """Features from the conductivity field (mean across ring) + per-image spread."""
    cond = u.mean(dim=0)                       # (1,1,H,W) shared state
    spread = u.std(dim=0)                      # disagreement across images
    f = []
    f.append(cond.mean(dim=(2, 3))); f.append(cond.var(dim=(2, 3)))
    f.append(cond.amax(dim=(2, 3)).values); f.append(cond.amin(dim=(2, 3)).values)
    h, w = cond.shape[2], cond.shape[3]
    f.append(cond[:, :, :h//2, :w//2].mean(dim=(2, 3)))
    f.append(cond[:, :, :h//2, w//2:].mean(dim=(2, 3)))
    f.append(cond[:, :, h//2:, :w//2].mean(dim=(2, 3)))
    f.append(cond[:, :, h//2:, w//2:].mean(dim=(2, 3)))
    spec = torch.fft.fft2(cond).abs()
    f.append(spec.mean(dim=(2, 3))); f.append(spec.amax(dim=(2, 3)).values)
    # conductivity-specific: how much the ring disagrees
    f.append(spread.mean(dim=(2, 3)))
    return torch.stack(f, dim=1)  # (B, 11)

# ------------------------------- models -------------------------------------
class FeatHead(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.fc = nn.Linear(n_feat, 10)
    def forward(self, z):
        return self.fc(z)

class MLP(nn.Module):
    def __init__(self, n_in=784, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(n_in, hidden),
                                 nn.ReLU(), nn.Linear(hidden, 10))
    def forward(self, x):
        return self.net(x)

# ------------------------------- data ---------------------------------------
def load_data(n_train=20000, n_test=5000, batch=256):
    tfm = transforms.Compose([transforms.Resize((GRID, GRID)), transforms.ToTensor()])
    train = datasets.MNIST("./", train=True, download=True, transform=tfm)
    test = datasets.MNIST("./", train=False, download=True, transform=tfm)
    tr = DataLoader(torch.utils.data.Subset(train, torch.randperm(len(train))[:n_train]),
                    batch_size=batch, shuffle=True)
    te = DataLoader(torch.utils.data.Subset(test, torch.randperm(len(test))[:n_test]),
                    batch_size=batch)
    return tr, te

@torch.no_grad()
def accuracy(model, ld, feat_fn=None):
    model.eval(); correct = total = 0
    for x, y in ld:
        x, y = x.to(DEVICE), y.to(DEVICE)
        z = feat_fn(x) if feat_fn is not None else x
        correct += (model(z).argmax(1) == y).sum().item()
        total += y.nelement()
    return correct / total

def train_head(model, ld, feat_fn, steps=8, lr=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(steps):
        model.train(); loss_acc = 0; n = 0
        for x, y in ld:
            x, y = x.to(DEVICE), y.to(DEVICE)
            z = feat_fn(x)
            loss = F.cross_entropy(model(z), y)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_acc += loss.item(); n += 1
        print(f"    epoch {epoch+1}/{steps}  loss {loss_acc/n:.4f}")

# ------------------------------- main ---------------------------------------
def main():
    print(f"device: {DEVICE}")
    train_ld, test_ld = load_data()
    results = {"device": str(DEVICE), "grid": GRID, "steps": T_STEPS, "ring_size": N_RING}

    # --- condition 1: raw pixels, linear head ---
    print("\n[1] raw pixels + linear head")
    t0 = time.time()
    raw = FeatHead(GRID * GRID).to(DEVICE)
    train_head(raw, train_ld, None, steps=8, lr=1e-3)
    results["raw_acc"] = accuracy(raw, test_ld)
    results["raw_time_s"] = round(time.time() - t0, 1)
    print(f"  acc {results['raw_acc']:.4f}  ({results['raw_time_s']}s)")

    # --- condition 2: single evolved image ---
    print("\n[2] single image evolved + linear head")
    t0 = time.time()
    def f_single(x): return feats_single(evolve_single(x))
    h1 = FeatHead(10).to(DEVICE)
    train_head(h1, train_ld, f_single, steps=8, lr=1e-2)
    results["single_acc"] = accuracy(h1, test_ld, f_single)
    results["single_time_s"] = round(time.time() - t0, 1)
    print(f"  acc {results['single_acc']:.4f}  ({results['single_time_s']}s)")

    # --- condition 3: ring with conductivity ---
    print("\n[3] ring of", N_RING, "images + conductivity features + linear head")
    t0 = time.time()
    def f_ring(x):
        # replicate digit across ring as initial conditions (different noise per copy)
        xs = x.repeat(1, N_RING, 1, 1)
        xs = xs + 0.02 * torch.randn_like(xs)  # orthogonal-ish initial spread
        return feats_ring(evolve_ring(xs))
    h2 = FeatHead(11).to(DEVICE)
    train_head(h2, train_ld, f_ring, steps=8, lr=1e-2)
    results["ring_acc"] = accuracy(h2, test_ld, f_ring)
    results["ring_time_s"] = round(time.time() - t0, 1)
    print(f"  acc {results['ring_acc']:.4f}  ({results['ring_time_s']}s)")

    # --- condition 4: MLP baseline ---
    print("\n[4] MLP baseline")
    t0 = time.time()
    mlp = MLP(hidden=16).to(DEVICE)
    train_head(mlp, train_ld, None, steps=8, lr=1e-3)
    results["mlp_acc"] = accuracy(mlp, test_ld)
    results["mlp_params"] = sum(p.nelement() for p in mlp.parameters())
    results["mlp_time_s"] = round(time.time() - t0, 1)
    print(f"  acc {results['mlp_acc']:.4f}  params {results['mlp_params']}  ({results['mlp_time_s']}s)")

    # ablation delta
    results["ring_minus_single"] = results["ring_acc"] - results["single_acc"]
    results["single_minus_raw"] = results["single_acc"] - results["raw_acc"]
    print("\nABLATION DELTAS:")
    print(f"  ring - single:   {results['ring_minus_single']:+.4f}  (conductivity gain)")
    print(f"  single - raw:    {results['single_minus_raw']:+.4f}  (dynamics gain)")

    with open(" mnist_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved mnist_results.json")

if __name__ == "__main__":
    main()
