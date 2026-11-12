# The Perturbation Sensing Stack

## Setup.

Let S be a dynamical system with state s_t in a Hilbert space H (in practice: the conductivity field c, or a single image grid u). A perturbation is a map epsilon: H -> H, small in operator norm. A sensing stack is a sequence of perturbations epsilon_1, ..., epsilon_N applied at times t_1 < ... < t_N, interleaved with the system's free evolution.

## The recurrence.

Between perturbations the system evolves by a transition operator M_t (the Jacobian of the update rule, or the delay-line shift). After N stacked perturbations:

    s_{t_N} = M_{t_N} ... M_{t_1} s_0 + sum_{k=1}^{N} [ product_{j=k}^{N-1} M_{t_{j+1}} ] epsilon_k (s_{t_k})

The first term is free evolution of the initial state. The sum is the accumulated perturbation — each epsilon_k is transported forward by the composite of all subsequent transitions. If the spectral radius of any composite exceeds 1 along a direction, that perturbation amplifies. That is the quantum-like gain: a weak tap, carried by the system's own memory, becomes a large displacement.

## Why stacking beats single taps.

A single perturbation epsilon gives displacement D = M_composite * epsilon. N stacked perturbations give

    D_total = sum_k M_{k->N} epsilon_k

If the epsilon_k are chosen in mutually orthogonal directions (different spatial frequencies, different delay-line modes, different phase quadratures), the cross terms vanish and

    ||D_total||^2 = sum_k ||M_{k->N} epsilon_k||^2

instead of the much smaller ||sum epsilon_k||^2 you'd get from aligned perturbations. Orthogonality is the whole game: it converts addition-in-amplitude into addition-in-energy.

## The optimal schedule.

Three conditions, all derivable:

1. **Timing**: space the perturbations at intervals matching the system's characteristic mixing or resonance times — the delay-line length, the dominant period of the update rule, the inverse of the leading Lyapunov exponent. Perturbations landing at those intervals ride the system's natural gain; off-resonance they get damped by the dissipative part of M.

2. **Amplitude**: follow 1/sqrt(k). This is the stochastic-resonance optimum: each tap strong enough to be detectable above the system's internal noise floor, weak enough not to kick the state out of the basin where the memory operators M_j stay correlated. Too strong and you reset the memory; too weak and the signal drowns.

3. **Annealing**: after the stack is built, slowly reduce perturbation amplitude (or equivalently, reduce the coupling b in the image rule) so the accumulated correlations settle into a stable basin rather than oscillating. This is simulated annealing on the perturbation channel.

## The quantum correspondence.

In quantum mechanics a weak measurement is a POVM element with small disturbance — exactly a small epsilon. A sequence of weak measurements on an evolving system accumulates information in the post-measurement state without collapsing it, which is the same recurrence above with M_t = unitary evolution and epsilon_k = the measurement back-action. The "quantum advantage" of the sensing stack is therefore not magic: it is (a) orthogonal mode packing, (b) resonance-timed sampling, and (c) sub-collapse amplitude. All three are classical and all three are implementable in the conductivity ring: each image is an orthogonal mode, the frame rate sets the sampling clock, and the CV coupling strength sets the amplitude.

## Configuration that maximizes the stack.

For the conductivity ring of N images with shared field c = mean_i(u_i):

- Perturb image i at step t_i = i * T_mix / N, where T_mix is the observed mixing time of c (measurable: time for the autocorrelation of c to decay below 1/e).
- Amplitude of perturbation i: A_i = A_0 / sqrt(i), with A_0 just above the noise floor of the readout stats (mean, variance, spectral centroid).
- Direction: perturb along the i-th Fourier mode of the grid, so each tap is orthogonal in spatial frequency.
- After all N taps, anneal: multiply all coupling strengths b_i by (1 - alpha)^k for k = 1..K steps with alpha ~ 0.05, letting c relax.
- Readout: the weak tap is any low-gain statistic of c — exactly the stats() function already in conductivity_sim.py. It siphons information without breaking circulation.

That is the full stack: orthogonal modes, resonance timing, 1/sqrt(k) amplitude, post-stack annealing, weak statistical readout. It is the configuration that makes stacked perturbation sensing maximal, and it is exactly what a coherent Ising machine does with pump strength as the annealing knob.
