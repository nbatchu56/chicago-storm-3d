"""A soothing ocean bed for the reel: slow wave swells (filtered noise) under a soft pad, stereo, 44.1 kHz.
usage: python3 -I music.py <seconds> <out.wav>"""
import sys, wave, numpy as np
SR = 44100
dur = float(sys.argv[1]); out = sys.argv[2]
n = int(dur * SR); t = np.arange(n) / SR
rng = np.random.default_rng(7)

def lowpass(x, fc):  # one-pole, run twice for a gentler slope
    a = np.exp(-2 * np.pi * fc / SR); y = np.empty_like(x); acc = 0.0
    for _ in range(2):
        for i in range(len(x)): acc = a * acc + (1 - a) * x[i]; y[i] = acc
        x = y.copy()
    return y

def ocean(seed):
    r = np.random.default_rng(seed)
    white = r.standard_normal(n)
    brown = np.cumsum(white); brown -= lowpass(brown, 0.5)          # brown noise without the drift
    brown /= np.max(np.abs(brown)) + 1e-9
    hiss = lowpass(r.standard_normal(n), 2500) * 0.35                 # the hiss of the wash
    return lowpass(brown, 600) * 1.0 + hiss

def swells(seed):  # the waves: a bump every ~7-10 s, each a slow rise and a longer fall, over a steady bed
    r = np.random.default_rng(seed); env = np.full(n, 0.28); tt = 1.5
    while tt < dur + 6:
        rise, fall, amp = r.uniform(1.6, 2.4), r.uniform(3.0, 4.5), r.uniform(0.6, 1.0)
        i0 = int(tt * SR)
        k = np.arange(int((rise + fall) * SR)); s = k / SR
        bump = np.where(s < rise, 0.5 - 0.5 * np.cos(np.pi * s / rise), 0.5 + 0.5 * np.cos(np.pi * (s - rise) / fall))
        j = min(n, i0 + len(k));
        if i0 < n: env[i0:j] += amp * bump[: j - i0]
        tt += r.uniform(6.5, 9.5)
    return np.clip(env, 0, 1.3)

def pad():  # two warm chords, A major 9 and F# minor 9, swapping every 8 s with a long crossfade; sines with a little 2nd harmonic
    chords = [[110.0, 164.81, 220.0, 277.18, 329.63, 493.88], [92.5, 138.59, 185.0, 220.0, 277.18, 415.30]]
    out = np.zeros(n)
    for ci, freqs in enumerate(chords):
        w = 0.5 + 0.5 * np.cos(2 * np.pi * t / 16 + (0 if ci == 0 else np.pi))   # 16 s cycle, chords alternate
        w = w ** 1.5
        for k, f in enumerate(freqs):
            det = 1 + 0.0012 * np.sin(2 * np.pi * (0.07 + 0.013 * k) * t)        # slow chorus
            tone = np.sin(2 * np.pi * f * det * t) + 0.18 * np.sin(2 * np.pi * 2 * f * det * t + 0.3)
            out += tone * w * (0.9 if k < 3 else 0.55)
    trem = 0.85 + 0.15 * np.sin(2 * np.pi * 0.11 * t)
    return lowpass(out, 1200) * trem / len(chords[0])

L = ocean(11) * swells(21); R = ocean(12) * swells(22)
p = pad()
mixL = 0.55 * L + 0.42 * p; mixR = 0.55 * R + 0.42 * p
# fades and a gentle level: peaks around -6 dBFS
fade_in = np.minimum(1, t / 1.5); fade_out = np.minimum(1, (dur - t) / 2.5)
env = fade_in * fade_out
st = np.stack([mixL, mixR], 1) * env[:, None]
st /= np.max(np.abs(st)) + 1e-9; st *= 0.5
pcm = (st * 32767).astype(np.int16)
with wave.open(out, "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
print("wrote", out, dur, "s")
