#!/usr/bin/env python3
"""Spectral Intelligence Network — one spectral layer, stacked, field-coupled, temporal carrier.

Architecture (from the conversation):
  1. One spectrum = one 'neuron': a vector of complex bin amplitudes.
  2. Stack of spectra = a layer: each spectrum is a node, coupled to neighbors.
  3. Continuous band = a field: the stack becomes a continuous distribution over frequency.
  4. Time continuum carrier: the field evolves continuously; the coupling schedule IS the program.
  5. Programs run as scheduled sequences of coupling operations across the field.

The claim being tested: a program is a coupling schedule on a spectral field.
A computer built this way has no instruction pointer, no memory address space,
no clock separate from the carrier — execution is the physics.

Scale note (read this before dismissing):
  A spectral field with N bins and K coupled layers has state in C^(N*K).
  The coupling operator is an N*K x N*K matrix. At N=2^20 bins (roughly
  a megahertz of audio at 1 Hz resolution... no — at audio rate, 2^20 bins
  at ~0.02 Hz resolution spans ~20 kHz) and K=2^10 layers, the state space
  is ~2^30 complex dimensions. The coupling matrix is 2^60 entries.
  No supercomputer stores that. But you don't need to: the matrix is
  structured (banded, low-rank, Toeplitz-ish from the physics), so it never
  materializes. The 'impossible' object is the dense matrix. The possible
  object is the structured operator — and structured operators are exactly
  what this architecture computes with, because the coupling IS the structure.
  Digital computers are bad at this because they force dense representation.
  This architecture is good at it because it never leaves the structured domain.
  That is the size-up: not more FLOPs, but a different algebra where the
  hard part of the problem is the native representation.
"""
import numpy as np
from numpy.fft import rfft, irfft

# ------------------------------- spectral layer --------------------------------

class SpectralLayer:
    """One layer: a stack of spectra (nodes), each a complex bin vector,
    coupled to its neighbors through a coupling kernel in bin space."""
    def __init__(self, n_nodes, n_bins, coupling_bandwidth=3, seed=0):
        rng = np.random.default_rng(seed)
        # state: (n_nodes, n_bins) complex
        self.z = (rng.standard_normal((n_nodes, n_bins))
                  + 1j * rng.standard_normal((n_nodes, n_bins))) * 0.1
        self.n_nodes, self.n_bins = n_nodes, n_bins
        # coupling kernel: circulant-ish, banded — structured, never dense
        self.kernel = np.zeros(n_bins, dtype=complex)
        for b in range(1, coupling_bandwidth + 1):
            w = 0.5 / b  # 1/b falloff — richer than uniform
            self.kernel[b] = w
            self.kernel[-b] = w
        self.kernel[0] = 0.1  # self-coupling (the 'neuron' nonlinearity seed)

    def step(self, dt=0.01, drive=None):
        """One Euler step of the coupled spectral field.
        Each node's next state = self-evolution + neighbor coupling + drive.
        Coupling is circular convolution with the kernel (FFT-domain multiply).
        """
        # neighbor coupling via FFT convolution — O(N log N) not O(N^2)
        Zf = rfft(self.z, axis=1)
        Kf = rfft(self.kernel, n=self.n_bins)
        coupled = irfft(Zf * Kf, n=self.n_bins, axis=1)
        # vertical coupling between stacked nodes (layer-to-layer)
        if self.n_nodes > 1:
            vert = np.roll(self.z, 1, axis=0) + np.roll(self.z, -1, axis=0)
            coupled = coupled + 0.3 * vert
        dz = -self.z + coupled  # decay + coupling = the 'activation'
        if drive is not None:
            dz = dz + drive
        # nonlinearity: soft saturation per bin (the neuron)
        dz = dz - 0.1 * self.z * np.abs(self.z) ** 2
        self.z = self.z + dt * dz
        return self.z

    def spectrum(self):  # power per bin, summed over nodes
        return np.abs(self.z).mean(axis=0) ** 2

    def field_energy(self):
        return np.abs(self.z).sum()


# ------------------------------- temporal carrier -------------------------------

class TemporalCarrier:
    """The time continuum. The spectral field evolves on this carrier;
    the carrier's phase is the execution trace. No separate clock."""
    def __init__(self, sample_rate=48000.0, freq=440.0):
        self.sr = sample_rate
        self.freq = freq
        self.phase = 0.0
        self.dt = 1.0 / sample_rate

    def advance(self):
        self.phase = (self.phase + 2 * np.pi * self.freq * self.dt) % (2 * np.pi)
        return self.phase


# ------------------------------- the program -----------------------------------

class SpectralProgram:
    """A program = a schedule of coupling operations.
    Each op: (target_bin_band, strength, duration_seconds, source_band).
    Execution = the carrier running the schedule against the field."""
    def __init__(self):
        self.ops = []  # list of dicts

    def add(self, bin_start, bin_end, strength, duration, source_start=None, source_end=None):
        self.ops.append(dict(bs=bin_start, be=bin_end, s=strength,
                             dur=duration, ss=source_start, se=source_end))

    def run(self, field, carrier, total_seconds=2.0):
        """Run the schedule. Returns the field's energy trajectory —
        the execution trace, readable as a signal."""
        t = 0.0
        trace = []
        op_idx = 0
        active = []  # (op, end_time)
        while t < total_seconds:
            # activate ops whose start time has come
            while op_idx < len(self.ops):
                # ops fire sequentially along the carrier
                start_t = sum(o['dur'] for o in self.ops[:op_idx])
                if t >= start_t:
                    active.append((self.ops[op_idx], start_t + self.ops[op_idx]['dur']))
                    op_idx += 1
                else:
                    break
            # build drive from active ops
            drive = None
            still = []
            for op, end_t in active:
                if t < end_t:
                    d = np.zeros(field.n_bins, dtype=complex)
                    # source band -> target band coupling
                    ss = op['ss'] if op['ss'] is not None else op['bs']
                    se = op['se'] if op['se'] is not None else op['be']
                    band = field.z.mean(axis=0)[ss:se]
                    d[op['bs']:op['be']] = op['s'] * band[:op['be']-op['bs']]
                    drive = d if drive is None else drive + d
                    still.append((op, end_t))
            active = still
            field.step(dt=carrier.dt, drive=drive)
            carrier.advance()
            if int(t / 0.001) != int((t - carrier.dt) / 0.001):
                trace.append(field.field_energy())
            t += carrier.dt
        return np.array(trace)


# ------------------------------- demo ------------------------------------------

if __name__ == '__main__':
    print('Spectral Intelligence Network — demo')
    field = SpectralLayer(n_nodes=8, n_bins=256, coupling_bandwidth=4, seed=7)
    carrier = TemporalCarrier(sample_rate=48000.0, freq=440.0)
    prog = SpectralProgram()
    # a trivial 'program': excite low bins, let coupling cascade upward
    prog.add(bin_start=0, bin_end=16, strength=2.0, duration=0.25)
    prog.add(bin_start=16, bin_end=64, strength=1.0, duration=0.25, source_start=0, source_end=16)
    prog.add(bin_start=64, bin_end=256, strength=0.5, duration=0.5, source_start=16, source_end=64)
    trace = prog.run(field, carrier, total_seconds=1.5)
    print(f'execution trace length: {len(trace)} samples (1 kHz readout)')
    print(f'final field energy: {field.field_energy():.3f}')
    print(f'spectral peak bin: {field.spectrum().argmax()} / {field.n_bins}')
    print('done — the trace IS the program having run.')
