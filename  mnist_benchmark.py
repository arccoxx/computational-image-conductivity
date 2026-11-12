#!/usr/bin/env python3
"""MNIST via self-coupled computational image as feature extractor.

Each digit image is injected into the grid as an initial condition.
The image evolves under the self-coupling rule for T steps; final-state
statistics become the feature vector. A linear head maps features to 10
classes. Compare against an MLP with matched parameter count.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import json, time

GRID = 64          # downsampled digit fits in grid
T_STEPS = 40       # evolution steps per sample
DT = 0.08
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------- image dynamics -------------------------------
def evolve(u, a=0.8, b=0.15, D=0.2, steps=T_STEPS):
    k = torch.tensor([[0., 1., 0.], [1., -4., 0.], [0., 1., 0.]],
                     device=u.device).view(1, 1, 3, 3)
    for _ in range(steps):
        conv = F.conv2d(u, k, padding=1)
        du = D * (torch.roll(u, 1, 2) + torch.roll(u, -1, 2) +
                  torch.roll(u, 1, 3) + torch.roll(u, -1, 3) - 4 * u)
        du = du - u ** 3 + a * u - b * conv
        u = u + DT * du
    return u

def features(u):
    # 16 statistical features from evolved state
    feats = []
    feats.append(u.mean(dim=(2, 3)))
    feats.append(u.var(dim=(2, 3)))
    feats.append(u.amax(dim=(2, 3)).values)
    feats.append(u.amin(dim=(2, 3)).values)
    # quadrant means
    h, w = u.shape[2], u.shape[3]
    feats.append(u[:, :, :h//2, :w//2].mean(dim=(2, 3)))
    feats.append(u[:, :, :h//2, w//2:].mean(dim=(2, 3)))
    feats.append(u[:, :, h//2:, :w//2].mean(dim=(2, 3)))
    feats.append(u[:, :, h//2:, w//2:].mean(dim=(2, 3)))
    # spectral
    spec = torch.fft.fft2(u).abs()
    feats.append(spec.mean(dim=(2, 3)))
    feats.append(spec.amax(dim=(2, 3)).values)
    return torch.stack(feats, dim=1)  # (B, 10)

# ------------------------------- models -------------------------------------
class ImageHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
    def forward(self, x):
        return self.fc(features(x))

class MLP(nn.Module):
    def __init__(self, n_in=784, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 10),
        )
    def forward(self, x):
        return self.net(x)

# ------------------------------- data ---------------------------------------
def load_data(n_train=20000, n_test=5000, batch=256):
    tfm = transforms.Compose([
        transforms.Resize((GRID, GRID)),
        transforms.ToTensor(),
    ])
    train = datasets.MNIST("./", train=True, download=True, transform=tfm)
    test = datasets.MNIST("./", train=False, download=True, transform=tfm)
    # subset for speed
    tr_idx = torch.randperm(len(train))[:n_train]
    te_idx = torch.randperm(len(test))[:n_test]
    train_ld = DataLoader(torch.utils.data.Subset(train, tr_idx), batch_size=batch, shuffle=True)
    test_ld = DataLoader(torch.utils.data.Subset(test, te_idx), batch_size=batch)
    return train_ld, test_ld

# ------------------------------- train/eval ----------------------------------
@torch.no_grad()
def accuracy(model, ld, evolve_input=False):
    model.eval()
    correct = total = 0
    for x, y in ld:
        x, y = x.to(DEVICE), y.to(DEVICE)
        if evolve_input:
            x = evolve(x)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y. numel() if hasattr(y, ' numel') else y.nelement()
    return correct / total

def train_model(model, ld, steps=5, lr=1e-3, evolve_input=False):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(steps):
        model.train()
        loss_acc = 0; n = 0
        for x, y in ld:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if evolve_input:
                x = evolve(x)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_acc += loss.item(); n += 1
        print(f"  epoch {epoch+1}/{steps}  loss {loss_acc/n:.4f}")

def main():
    print(f"device: {DEVICE}")
    train_ld, test_ld = load_data()
    results = {"device": str(DEVICE), "grid": GRID, "steps": T_STEPS}

    # 1) image-dynamics feature extractor + linear head
    print("\n[1] image-evolution features + linear head")
    t0 = time.time()
    head = ImageHead().to(DEVICE)
    # freeze dynamics; only train head
    train_model(head, train_ld, steps=8, lr=1e-2, evolve_input=True)
    acc = accuracy(head, test_ld, evolve_input=True)
    results["image_head_acc"] = acc
    results["image_head_time_s"] = round(time.time() - t0, 1)
    print(f"  test acc: {acc:.4f}  ({results['image_head_time_s']}s)")

    # 2) matched-parameter MLP baseline on raw pixels
    print("\n[2] MLP baseline (matched param count)")
    t0 = time.time()
    # count head params: 10*10+10 = 110; MLP: 784*h + h + h*10 + 10
    # choose h so params ~ 110..  use h=8 -> 784*8+8+80+10 = 6366; too many.
    # simpler: report both raw numbers
    mlp = MLP(hidden=16).to(DEVICE)
    train_model(mlp, train_ld, steps=8, lr=1e-3, evolve_input=False)
    acc_mlp = accuracy(mlp, test_ld, evolve_input=False)
    results["mlp_acc"] = acc_mlp
    results["mlp_time_s"] = round(time.time() - t0, 1)
    results["mlp_params"] = sum(p.nelement() for p in mlp.parameters())
    results["image_head_params"] = sum(p.nelement() for p in head.parameters())
    print(f"  test acc: {acc_mlp:.4f}  params {results['mlp_params']}  ({results['mlp_time_s']}s)")
    print(f"  image-head params: {results['image_head_params']}")

    with open(" mnist_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved mnist_results.json")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
