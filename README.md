# Computational Image Conductivity

Self-coupling computational images as a conductivity substrate: strange-loop shaders on RTX 5080, MNIST benchmarks, residual-stream rebalancing, modulation theory, and the compression-as-intelligence thesis.

Companion repos: Eurorack-Onwards, true-quantum-compute, language-circuit, recirculating-buffer, reality-synth-modules.

## The core claim

A computational image is not a picture. It is a state in a high-dimensional dynamical system, rendered as a 2-D slice for human inspection. The image updates by a fixed rule; the rule's parameters are driven by statistics of the image's own previous output. That is a strange loop with a learned transition. When two such images modulate each other, mutual coupling appears. A ring of them yields shared state — conductivity — through which inquiry flows.

The frame rate is the clock. The shader is discrete hardware running a continuous-in-parameter-space update. That is an analog device wearing a digital body.

## Files

- `conductivity_sim.py` — core simulation: grid state, self-coupling update, parameter readout, ring of coupled images, GPU path via PyTorch.
- ` mnist_benchmark.py` — MNIST via the image as feature extractor; compares against a small MLP baseline; logs parameter trajectories.
- `theory_notes.md` — formalism: conductivity as shared state, compression-as-intelligence (Hutter Prize lineage), modulation theory, residual-stream rebalancing, why for-loops are the wrong abstraction.
- `requirements.txt` — PyTorch CUDA 12.8 for sm_120 (5080).

## Run

```bash
pip install -r requirements.txt
python conductivity_sim.py          # ring demo, saves ring_evolution.png
python mnist_benchmark.py           # MNIST, saves mnist_results.json
```

## Hardware target

RTX 5080 (16 GB, sm_120). All tensors float32, grid 256x256 default. Ring of 8 images fits comfortably; 32-image rings still fit with reduced resolution.

## Status

Working prototype. MNIST accuracy is the first real number to watch — if the self-coupled image beats or matches a same-parameter-count MLP, the conductivity thesis earns its next experiment.
