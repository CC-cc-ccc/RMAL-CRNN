from pathlib import Path
import json
import numpy as np
import pywt
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, hilbert
import pipeline as rp
BASE = Path(__file__).resolve().parents[2]
SRC = BASE / 'outputs' / 'range' / 'channels'
OUT = SRC / 'features_refined'
WAV = SRC / 'wav'
VALID = [c for c in range(1, 33) if c not in (9, 15)]
BANDS = np.geomspace(80.0, 5000.0, 9)

def _frames(x):
    return rp._frames(x)

def _band_filter(x, sr, lo, hi):
    hi = min(hi, sr / 2 - 100)
    sos = butter(3, [lo, hi], btype='bandpass', fs=sr, output='sos')
    return sosfiltfilt(sos, x).astype(np.float32)

def refined_hht(x, sr=rp.FS):
    envs, ifs = ([], [])
    for lo, hi in zip(BANDS[:-1], BANDS[1:]):
        z = _band_filter(x, sr, lo, hi)
        a = hilbert(z)
        env = np.abs(a)
        ph = np.unwrap(np.angle(a))
        inst = np.diff(ph, prepend=ph[0]) * sr / (2 * np.pi)
        envs.append(env.astype(np.float32))
        ifs.append(np.clip(inst, 0, sr / 2).astype(np.float32))
    ef = np.stack([_frames(v) for v in envs], axis=1)
    ff = np.stack([_frames(v) for v in ifs], axis=1)
    e2 = ef ** 2 + 1e-08
    p = e2 / e2.sum(axis=2, keepdims=True)
    out = np.stack([np.log1p(e2.mean(2)), ef.std(2) / (ef.mean(2) + 1e-06), np.median(ff, axis=2), -(p * np.log2(p + 1e-08)).sum(2)], axis=2)
    return np.nan_to_num(out.transpose(0, 2, 1).reshape(len(ef), 32), nan=0, posinf=0, neginf=0).astype(np.float32)

def refined_wpt(x):
    frames = _frames(x)
    rows = []
    for frame in frames:
        wp = pywt.WaveletPacket(data=np.array(frame, dtype=np.float32, copy=True), wavelet='db4', mode='symmetric', maxlevel=5)
        nodes = wp.get_level(5, order='natural')
        en = np.asarray([np.mean(np.square(n.data), dtype=np.float64) for n in nodes], np.float32)
        rel = en / (en.sum() + 1e-08)
        rows.append(np.log1p(10000.0 * rel))
    return np.nan_to_num(np.asarray(rows, np.float32), nan=0, posinf=0, neginf=0)

def refined_hhsa(x, sr=rp.FS):
    env_frames = []
    for lo, hi in zip(BANDS[:-1], BANDS[1:]):
        env_frames.append(_frames(np.abs(hilbert(_band_filter(x, sr, lo, hi)))))
    rows = []
    freq = np.fft.rfftfreq(rp.NFFT, 1 / sr)
    sel = (freq >= 1) & (freq <= 200)
    for t in range(env_frames[0].shape[0]):
        vals = []
        for ef in env_frames:
            env = ef[t] - ef[t].mean()
            m = np.abs(np.fft.rfft(env * np.hanning(len(env))))
            mm = m[sel]
            ff = freq[sel]
            p = mm / (mm.sum() + 1e-08)
            vals.extend([np.log1p(np.mean(env * env)), np.std(env), float((ff * p).sum()), float(-(p * np.log2(p + 1e-08)).sum())])
        rows.append(vals)
    return np.nan_to_num(np.asarray(rows, np.float32), nan=0, posinf=0, neginf=0)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for c in VALID:
        rows, labels, sources, seconds = ([], [], [], [])
        for name, label, count in (('90101', 1, 200), ('90102', 1, 200), ('90105', 0, 400)):
            wav = WAV / name / f'ch{c:02d}.wav'
            sr, raw = wavfile.read(wav)
            raw = raw.astype(np.float32)
            y = rp.resample_poly(raw, rp.FS, sr).astype(np.float32) if sr != rp.FS else raw
            n = min(count, len(y) // rp.WIN)
            for i in range(n):
                seg = y[i * rp.WIN:(i + 1) * rp.WIN]
                base = rp.extract_features(seg)
                base[:, 0:32] = refined_hht(seg)
                base[:, 96:128] = refined_wpt(seg)
                base[:, 128:160] = refined_hhsa(seg)
                rows.append(base)
                labels.append(label)
                sources.append(name)
                seconds.append(i)
        X = np.asarray(rows, np.float32)
        np.savez_compressed(OUT / f'ch{c:02d}.npz', X=X, y=np.asarray(labels, np.int64), source=np.asarray(sources), second=np.asarray(seconds))
        print(f'refined channel {c:02d}', flush=True)
    (OUT / 'README.json').write_text(json.dumps({'domains_refined': ['HHT', 'WPT', 'HHSA'], 'domains_kept': ['LOFAR', 'FFT', 'MFCC'], 'standardization': 'none in extraction'}, indent=2), encoding='utf-8')
if __name__ == '__main__':
    main()
