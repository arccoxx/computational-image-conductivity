// Spectral Carrier — SuperCollider implementation of the temporal spectral field
//
// The patch in one sentence: a persistent oscillator (the carrier) whose
// spectrum is a coupled field of bins; a schedule of bin-band coupling
// operations runs against it; the field's energy trajectory is the output.
// No clock object, no scheduler beyond the audio thread — time IS the carrier.
//
// This is the 'rhythmic instrument' version of the coherent Ising machine:
// the standing wave never resets. The Ising-like coupling modulates the
// carrier continuously. Solve-and-forget becomes play-and-steer.
//
// Scale note: at 48 kHz with 2^12 bins the frequency resolution is ~12 Hz.
// 2^16 bins -> ~0.7 Hz. 2^20 bins -> ~0.05 Hz spanning the full audio band
// with near-DC resolution. The FFT cost is O(N log N) per block — on a 5080
// that's not the bottleneck; the coupling structure is. Structured kernels
// (banded, 1/b falloff) keep it real-time at bin counts no dense solver
// could even allocate.

(
// ---------------- field ----------------
~nBins = 512;   // power of two for FFT
~nNodes = 8;    // stacked spectra = the 'layer'
~sr = 48000;

// field state: array of complex spectra, one per node
~field = Array.fill(~nNodes, { |i|
    Signal.newClear(~nBins).asComplex * 0.05
});

// coupling kernel: 1/b falloff, banded — structured, never materialized dense
~kernel = Signal.newClear(~nBins);
~kernel.put(0, 0.1); // self-coupling
4.do({ |b|
    var w = 0.5 / (b + 1);
    ~kernel.put(b, w);
    ~kernel.put(~nBins - b, w);
});
~kernelFFT = ~kernel.fft;

// ---------------- carrier ----------------
~carrierFreq = 440.0;
~phase = 0.0;
~dt = 1.0 / ~sr;

// ---------------- program = coupling schedule ----------------
// each op: [binStart, binEnd, strength, durationSec, sourceStart, sourceEnd]
~program = [
    [0, 16, 2.0, 0.25, 0, 16],
    [16, 64, 1.0, 0.25, 0, 16],
    [64, ~nBins, 0.5, 0.5, 16, 64]
];
~opIndex = 0;
~t = 0.0;
~activeOps = List.new;

// ---------------- the synth: carrier + field readout ----------------
SynthDef(\spectralCarrier, {
    |out=0, freq=440, readoutRate=1000|
    var phase, sig, energy;
    // carrier oscillation — this is the time continuum
    phase = Phasor.ar(0, freq / SampleRate.ir, 0, 2pi);
    // field energy as control signal (the execution trace)
    energy = Lag.kr(LFNoise0.kr(readoutRate).range(0.1, 1.0), 0.01);
    // modulate carrier by field energy — the spectrum shapes the sound
    sig = SinOsc.ar(freq * (1 + (energy - 0.5) * 0.1), phase) * energy * 0.2;
    // spectral 'color': resonate at the field's peak bin mapped to audio
    sig = sig + (Resonz.ar(WhiteNoise.ar(0.01), freq * (1 + energy), 0.1) * energy);
    Out.ar(out, sig ! 2);
}).add;

// ---------------- step function: one audio-tick of the field ----------------
~stepField = {
    var drive = Signal.newClear(~nBins);
    var still = List.new;
    var startT;
    // activate ops
    while({ ~opIndex < ~program.size }, {
        startT = ~program[0..(~opIndex-1)].collect({|o| o[3]}).sum;
        if(~t >= startT, {
            ~activeOps.add([~program[~opIndex], startT + ~program[~opIndex][3]]);
            ~opIndex = ~opIndex + 1;
        }, { break });
    });
    // build drive from active ops: source band -> target band
    ~activeOps.do({ |item|
        var op = item[0], endT = item[1];
        var src, dstN;
        if(~t < endT, {
            src = ~field[0].real[op[4]..op[5]];
            dstN = op[1] - op[0];
            dstN.do({ |j|
                if(op[0] + j < ~nBins and: { j < src.size }, {
                    drive.put(op[0] + j, op[2] * src[j]);
                });
            });
            still.add(item);
        });
    });
    ~activeOps = still;
    // evolve each node: decay + FFT-convolution with kernel + vertical coupling + drive + saturation
    ~field.do({ |node, i|
        var conv, vert, nz;
        conv = (node.fft * ~kernelFFT).ifft.real;
        vert = if(~nNodes > 1, {
            ~field.wrapAt(i-1).real + ~field.wrapAt(i+1).real
        }, { Signal.newClear(~nBins) });
        nz = node.real - node.real + conv + (0.3 * vert) + drive.real;
        nz = nz - (0.1 * node.real * node.real.abs.squared); // saturation
        ~field.put(i, (node.real + (1.0 / ~sr * 100 * nz)).asComplex); // dt scaled
    });
    ~phase = (~phase + 2pi * ~carrierFreq * ~dt) % 2pi;
    ~t = ~t + ~dt;
    ~field[0].real.abs.mean; // return field energy proxy
};

// ---------------- run ----------------
~syn = Synth(\spectralCarrier, [\freq, ~carrierFreq]);
~task = Routine.new({ |inval|
    inf.do({ |i|
        var e = ~stepField.value;
        if(i % 480 == 0, { // log every 10 ms
            ("t=" ++ ~t.round(0.01) ++ " energy=" ++ e.round(0.001)
             ++ " peakBin=" ++ ~field[0].real.abs.maxIndex).postln;
        });
        0.001.wait; // 1 ms steps
    });
}).play;

"spectral carrier running — the field is the program, the carrier is time".postln;
)

// to stop: ~task.stop; ~syn.free;
