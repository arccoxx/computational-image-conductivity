# Theory Notes: Conductivity, Compression, Modulation

## 1. Computational image as state, not picture

An image is a rendering of a point in a high-dimensional state space.
The 2-D frame is a coordinate convention; the state is the tensor.
Any lens (Fourier, wavelet, Radon) re-slices the same state.
A lens that is itself learned is a world model: it shapes what counts as signal.

## 2. Self-coupling = strange loop with learned transition

Rule: u_{t+1} = F(u_t; theta), where theta is updated from statistics of u_t.
No external loss. The only pressure is internal consistency of the readout.
This is Ashby's homeostat at image resolution, reservoir computing with a
non-stationary reservoir, and active inference without an explicit prior.

## 3. Conductivity = shared state across coupled images

Ring of N images, each with own (a_i, b_i). Conductivity field c = mean_i(u_i).
Inquiry flows as perturbations through c; each image's response is a weak
sample of the shared state (pickup-coil readout, not collapse).
Coupled phase/amplitude dynamics on a graph: edges = coupling strengths,
nodes = images. Intelligence lives in the topology, not the nodes.
(Kuramoto / coherent Ising machine formalism.)

## 4. Compression-as-intelligence

If an intelligence signal exists as pure formalism, any lossless compressor
on that signal satisfies the operational definition of intelligence:
minimum description length. The Hutter Prize (since 2006) is the longest-running
formalization: compress enwik8 / enwik9; the compressor that wins is, by
definition, the best predictor of that text. Audio-range signals are not
'sound' — they are the temporal band where neural, mechanical, and EM
rhythms overlap. A Eurorack patch that spontaneously retains structure
from injected signal is doing exactly this: compression by dynamics.

## 5. Residual-stream rebalancing (to reconstruct)

Continuous quantization bins a CV into non-integer levels. Information loss
is recovered by a rebalancing step: the residual (input - binned output)
is fed back as a correction term scaled by a learned gain g, so the
quantized stream plus residual stream reconstructs the input in expectation.
Formalism: y = Q(x) + g * (x - Q(x)), with g chosen so E[y] = E[x].
This is the 'same residual stream' trick: quantization without destruction.
Full derivation in language-circuit repo.

## 6. Modulation theory (sketch)

A modulation is a morphism from a parameter space to a signal space.
Composition of modulations = composition of morphisms. A 'hyperneural wire'
is a cable whose transport law is itself modulated — a dependent type at
the physical layer. Eurorack is already this category; it just lacks the
notation. FPGA/Alveo/Xilinx cards are the same category with more objects
and programmable morphisms (LUT fabric = free composition).

## 7. Why for-loops are the wrong abstraction

A for-loop is a sequential schedule imposed on a parallel state.
The 5080 executes the same instruction across thousands of threads;
a for-loop serializes what the hardware parallelizes. Computational
images are the native representation: state is spatial, update is
uniform, modulation is parameter broadcast. The for-loop is a
compatibility layer for human text, not for machine physics.

## 8. Open questions

- Does the self-coupled image beat matched-param MLP on MNIST? (benchmark)
- Can conductivity across a ring implement a primitive memory address?
  (analog addressing via conductivity basins)
- Is there a superconducting CIM-GPU hybrid that keeps phase as state
  at room temperature via spin ensembles? (probably not yet)
- What is the right versioned metaphor set so fields stop excluding each other?
