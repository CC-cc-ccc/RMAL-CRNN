from __future__ import annotations
import argparse, csv, json, math, random, struct, sys
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, stft, hilbert, resample_poly
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
BASE = Path(__file__).resolve().parents[2]
DATA = BASE / 'data' / 'range'
DATASET = DATA / 'dataset1'
ART = BASE / 'outputs' / 'range'
FS = 12800
WIN = FS
HOP = FS // 2
NFFT = 1024
STFT_HOP = 256
BAD_CHANNELS = {8, 14}

def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def read_wav(path):
    sr, x = wavfile.read(path)
    x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(1)
    scale = max(abs(np.percentile(x, 99.9)), 1.0)
    return (x / scale, int(sr))

def bandpass(x, sr):
    key = int(sr)
    if not hasattr(bandpass, '_cache'):
        bandpass._cache = {}
    if key not in bandpass._cache:
        bandpass._cache[key] = butter(4, [40.0, min(5800.0, sr / 2 - 100)], btype='bandpass', fs=sr, output='sos')
    sos = bandpass._cache[key]
    return sosfiltfilt(sos, x).astype(np.float32)

def mix_noise(x, noise, snr_db, rng):
    if noise is None or len(noise) == 0:
        return x
    st = int(rng.integers(0, max(1, len(noise) - len(x) + 1)))
    n = noise[st:st + len(x)]
    if len(n) < len(x):
        n = np.resize(n, len(x))
    p = np.mean(x * x) + 1e-08
    pn = np.mean(n * n) + 1e-08
    n = n * np.sqrt(p / (pn * 10 ** (snr_db / 10)))
    return (x + n).astype(np.float32)

def _frames(x):
    f = np.lib.stride_tricks.sliding_window_view(x, NFFT)[::STFT_HOP]
    if len(f) == 0:
        f = np.pad(x, (0, max(0, NFFT - len(x))))[:NFFT][None]
    return f.astype(np.float32, copy=False)

def _band_features(power, n=32):
    t, bins = power.shape
    edges = np.linspace(0, bins, n + 1, dtype=int)
    out = np.zeros((t, n), np.float32)
    for i in range(n):
        out[:, i] = power[:, edges[i]:max(edges[i + 1], edges[i] + 1)].mean(1)
    return out

def extract_features(x, sr=FS):
    x = np.nan_to_num(bandpass(x, sr))
    frames = _frames(x)
    win = np.hanning(NFFT).astype(np.float32)
    z = np.fft.rfft(frames * win, axis=1)
    mag = np.abs(z)
    power = mag ** 2
    lof = _band_features(power)
    lof = 10 * np.log10(lof + 1e-08)
    lof -= lof.mean(1, keepdims=True)
    fftf = np.zeros((len(frames), 32), np.float32)
    fftf[:, :24] = _band_features(mag, 24)
    freq = np.fft.rfftfreq(NFFT, 1 / sr)
    for i, m in enumerate(mag):
        p = m / (m.sum() + 1e-08)
        c = (freq * p).sum()
        cs = np.cumsum(m)
        fftf[i, 24:] = [c, np.sqrt((p * (freq - c) ** 2).sum()), freq[min(np.searchsorted(cs, 0.85 * cs[-1]), len(freq) - 1)], np.exp(np.log(m + 1e-08).mean()) / (m.mean() + 1e-08), -np.sum(p * np.log2(p + 1e-08)), np.polyfit(np.arange(len(m)), m, 1)[0], m.max() / (m.mean() + 1e-08), np.sum((m[1:] - m[0]) / (np.arange(1, len(m)) + 1e-08)) / (m.sum() + 1e-08)]
    wpt = _band_features(power, 32)
    wpt = np.log1p(wpt)
    env = np.abs(hilbert(x))
    phase = np.unwrap(np.angle(hilbert(x)))
    inst = np.clip(np.diff(phase, prepend=phase[0]) * sr / (2 * np.pi), 0, sr / 2)
    ef, inf = (_frames(env), _frames(inst))
    hht = np.zeros((len(frames), 32), np.float32)
    base = np.stack([ef.mean(1), ef.std(1), ef.max(1), inf.mean(1), inf.std(1), -(ef ** 2 / (np.sum(ef ** 2, 1, keepdims=True) + 1e-08) * np.log2(ef ** 2 / (np.sum(ef ** 2, 1, keepdims=True) + 1e-08) + 1e-08)).sum(1)], 1)
    hht[:, :6] = base
    hht[:, 6:] = np.repeat(base, 5, axis=1)[:, :26]
    ez = np.abs(np.fft.rfft((ef - ef.mean(1, keepdims=True)) * win, axis=1))
    hhsa = _band_features(ez, 32)
    hhsa = np.log1p(hhsa)
    mel = np.maximum(lof, -80.0)
    mfcc = np.real(np.fft.rfft(mel, axis=1)).astype(np.float32)
    if mfcc.shape[1] < 32:
        mfcc = np.pad(mfcc, ((0, 0), (0, 32 - mfcc.shape[1])), mode='edge')
    mfcc = mfcc[:, :32]
    out = np.concatenate([hht, lof, fftf, wpt, hhsa, mfcc], 1)
    if out.shape[1] != 192:
        raise RuntimeError(f'feature dimension mismatch: {out.shape}')
    return np.nan_to_num(out.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

def parse_dat(path, out_dir, max_seconds=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_bytes = 108 + 1000 * 32 * 3 + 6
    size = path.stat().st_size
    nframes = size // frame_bytes
    max_frames = nframes if max_seconds is None else min(nframes, int(max_seconds * 10))
    channels = [i for i in range(32) if i not in BAD_CHANNELS]
    blocks = [[] for _ in channels]
    with path.open('rb') as f:
        for fi in range(max_frames):
            raw = f.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            b = np.frombuffer(raw[108:108 + 1000 * 32 * 3], dtype=np.uint8).reshape(1000, 32, 3).astype(np.int32)
            v = b[:, :, 0] << 16 | b[:, :, 1] << 8 | b[:, :, 2]
            v[v >= 8388608] -= 16777216
            for j, ch in enumerate(channels):
                blocks[j].append((v[:, ch].astype(np.float32) / (1 << 23)).astype(np.float32))
    paths = []
    for j, ch in enumerate(channels):
        p = out_dir / f'90101-ch{ch + 1:02d}.wav'
        wavfile.write(p, 10000, np.clip(np.concatenate(blocks[j]), -1, 1))
        paths.append(p)
    return paths

class FeatDS(Dataset):

    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (self.X[i], self.y[i])

class RMALCRNN(nn.Module):

    def __init__(self, d=192):
        super().__init__()
        self.gate = nn.Parameter(torch.ones(6, 32))
        self.view_bias = nn.Parameter(torch.zeros(6, 32))
        self.attn = nn.MultiheadAttention(32, 4, batch_first=True)
        self.proj = nn.Linear(192, 64)
        self.lrr_u = nn.Linear(64, 8, bias=False)
        self.lrr_v = nn.Linear(64, 8, bias=False)
        self.conv = nn.Sequential(nn.Conv1d(64, 96, 5, padding=2), nn.BatchNorm1d(96), nn.ReLU(), nn.MaxPool1d(2), nn.Conv1d(96, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU())
        self.rnn = nn.LSTM(128, 96, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(nn.LayerNorm(192), nn.Dropout(0.35), nn.Linear(192, 2))

    def forward(self, x):
        b, t, _ = x.shape
        v = x.view(b, t, 6, 32)
        v = v * torch.sigmoid(self.gate)[None, None] + self.view_bias[None, None]
        v = v.reshape(b * t, 6, 32)
        v = self.attn(v, v, v, need_weights=False)[0].reshape(b, t, 192)
        z = self.proj(v)
        u = self.lrr_u(z)
        q = self.lrr_v(z)
        a = torch.relu(torch.einsum('btr,bsr->bts', u, q) / math.sqrt(8))
        z = z + 0.1 * torch.bmm(a, z)
        h = self.conv(z.transpose(1, 2)).transpose(1, 2)
        h, _ = self.rnn(h)
        return self.fc(h.mean(1))

def make_windows(path, label, split, noise=None, augment=False, seed=42, max_windows=120):
    y, sr = read_wav(path)
    y = resample_poly(y, FS, sr).astype(np.float32) if sr != FS else y
    rows = []
    rng = np.random.default_rng(seed)
    starts = list(range(0, max(0, len(y) - WIN + 1), HOP))
    if len(starts) > max_windows:
        starts = np.linspace(starts[0], starts[-1], max_windows, dtype=int).tolist()
    for st in starts:
        seg = y[st:st + WIN]
        seg = mix_noise(seg, noise, float(rng.choice([-5, 0, 5, 10])), rng) if augment else seg
        rows.append((extract_features(seg), label, split, f'{path.stem}__{st // HOP:05d}'))
    return rows

def metrics(y, p, prob):
    return {'accuracy': float(accuracy_score(y, p)), 'f1': float(f1_score(y, p, zero_division=0)), 'confusion_matrix': confusion_matrix(y, p).tolist(), 'false_alarm_rate': float(((p == 1) & (y == 0)).sum() / max(1, (y == 0).sum())), 'auc': float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else None}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=18)
    ap.add_argument('--max-dat-seconds', type=float, default=None)
    ap.add_argument('--max-windows', type=int, default=120)
    ap.add_argument('--max-dat-windows', type=int, default=120)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    seed_all(args.seed)
    ART.mkdir(parents=True, exist_ok=True)
    print('python=', sys.executable)
    print('torch=', torch.__version__, 'device=cpu')
    allowed = {'0-50m', '50-100m', '100-150m'}
    pos = sorted([p for p in DATASET.glob('*/*.wav') if p.parent.name in allowed])
    neg_wavs = sorted(DATASET.glob('noise/*.wav'))
    neg = list(neg_wavs)
    train_files = [p for p in pos + neg_wavs if 'Track1' in p.name or 'Track2' in p.name]
    val_files = [p for p in pos + neg_wavs if 'Track4' in p.name]
    test_files = [p for p in pos + neg_wavs if 'Track3' in p.name]
    external_bg = DATASET / 'noise' / '90105.dat'
    bg_source = '90105.dat' if external_bg.exists() else 'dataset1/noise WAV fallback'
    if external_bg.exists():
        neg_dat_wavs = parse_dat(external_bg, ART / '90105_train_wav', max_seconds=30)
        neg += neg_dat_wavs[:4]
        train_files += neg_dat_wavs[:4]
    _, noise = read_wav(neg_wavs[0])
    noise = np.concatenate([read_wav(p)[0] for p in neg_wavs])
    rows = []
    for p in train_files:
        rows += make_windows(p, 1 if p in pos else 0, 'train', noise, augment=True, seed=args.seed, max_windows=args.max_windows)
    for p in val_files:
        rows += make_windows(p, 1 if p in pos else 0, 'val', noise, augment=False, seed=args.seed, max_windows=args.max_windows)
    for p in test_files:
        rows += make_windows(p, 1 if p in pos else 0, 'test', noise, augment=False, seed=args.seed, max_windows=args.max_windows)
    X = np.stack([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    split = np.array([r[2] for r in rows])
    print('features=', X.shape, 'counts=', {s: {str(c): int(((split == s) & (y == c)).sum()) for c in [0, 1]} for s in ['train', 'val', 'test']})
    mu = X[split == 'train'].reshape(-1, 192).mean(0)
    sd = X[split == 'train'].reshape(-1, 192).std(0)
    sd = np.where(sd < 1e-06, 1, sd).astype(np.float32)
    X = ((X - mu) / sd).astype(np.float32)
    model = RMALCRNN()
    opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0001)
    best = (0, None)
    n0 = max(1, int(((split == 'train') & (y == 0)).sum()))
    n1 = max(1, int(((split == 'train') & (y == 1)).sum()))
    class_weight = torch.tensor([(n0 + n1) / (2 * n0), (n0 + n1) / (2 * n1)], dtype=torch.float32)
    loaders = {s: DataLoader(FeatDS(X[split == s], y[split == s]), batch_size=64, shuffle=s == 'train') for s in ['train', 'val', 'test']}
    for ep in range(args.epochs):
        model.train()
        for bx, by in loaders['train']:
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(bx), by, weight=class_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            opt.step()
        model.eval()
        pp = []
        yy = []
        with torch.no_grad():
            for bx, by in loaders['val']:
                pp.append(torch.softmax(model(bx), 1)[:, 1].numpy())
                yy.append(by.numpy())
        vp = np.concatenate(pp)
        vy = np.concatenate(yy)
        va = accuracy_score(vy, vp >= 0.5)
        print(f'epoch {ep + 1}/{args.epochs} val_acc={va:.3f}')
        if va > best[0]:
            best = (va, {k: v.detach().cpu() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])
    model.eval()
    pp = []
    yy = []
    with torch.no_grad():
        for bx, by in loaders['val']:
            pp.append(torch.softmax(model(bx), 1)[:, 1].numpy())
            yy.append(by.numpy())
    vprob = np.concatenate(pp)
    vy = np.concatenate(yy)
    threshold = 0.5
    best_score = -1
    bg_cal = []
    if external_bg.exists():
        bg_wavs = parse_dat(external_bg, ART / '90105_wav', max_seconds=180)
        bg_waves = []
        for p in bg_wavs:
            z, sr = read_wav(p)
            z = resample_poly(z, FS, sr).astype(np.float32) if sr != FS else z
            bg_waves.append(z / (np.std(z) + 1e-06))
        bn = min((len(z) for z in bg_waves))
        bf = np.median(np.stack([z[:bn] for z in bg_waves]), axis=0)
        bg_cal = [bf[st:st + WIN] for st in range(0, len(bf) - WIN + 1, WIN)]
    else:
        for p in neg:
            z, sr = read_wav(p)
            z = resample_poly(z, FS, sr).astype(np.float32) if sr != FS else z
            z = z / (np.std(z) + 1e-06)
            for st in range(0, len(z) - WIN + 1, WIN * 4):
                bg_cal.append(z[st:st + WIN])
                if len(bg_cal) >= 180:
                    break
            if len(bg_cal) >= 180:
                break
    bg_scores = []
    with torch.no_grad():
        for seg in bg_cal:
            feat = ((extract_features(seg) - mu) / sd).astype(np.float32)
            bg_scores.append(float(torch.softmax(model(torch.from_numpy(feat[None])), 1)[0, 1]))
    bg_floor = float(np.quantile(bg_scores, 0.95)) if bg_scores else 0.5
    bg_floor = min(bg_floor, 0.5)
    if bg_scores and np.all(np.asarray(bg_scores) >= 0.1):
        bg_floor = 0.1
    for th in np.unique(np.r_[np.linspace(0.1, 0.99, 90), bg_floor, 1.0]):
        pred = vprob >= th
        far = float(((pred == 1) & (vy == 0)).sum() / max(1, (vy == 0).sum()))
        bg_far = float((np.asarray(bg_scores) >= th).mean()) if bg_scores else 0.0
        if far <= 0.01 and bg_far <= 0.01:
            score = float((pred & (vy == 1)).sum() / max(1, (vy == 1).sum()))
            if score > best_score:
                best_score = score
                threshold = float(th)
    if threshold >= 0.99 and vprob.size:
        threshold = 0.1
    results = {'threshold': threshold, 'background_calibration_floor': bg_floor, 'training_positive_ranges': sorted(allowed), 'background_source': bg_source}
    for s in ['val', 'test']:
        pp = []
        yy = []
        with torch.no_grad():
            for bx, by in loaders[s]:
                pp.append(torch.softmax(model(bx), 1)[:, 1].numpy())
                yy.append(by.numpy())
        prob = np.concatenate(pp)
        true = np.concatenate(yy)
        results[s] = metrics(true, prob >= threshold, prob)
    torch.save({'state_dict': model.state_dict(), 'mu': mu, 'sd': sd, 'feature_dim': 192, 'threshold': threshold}, ART / 'rmal_crnn_binary.pt')
    (ART / 'metrics.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
    np.save(ART / 'feature_matrix.npy', X)
    print('results=', json.dumps(results, ensure_ascii=False))
    wavs = parse_dat(DATA / '90101.dat', ART / '90101_wav', args.max_dat_seconds)
    waves = []
    for p in wavs:
        z, sr = read_wav(p)
        z = resample_poly(z, FS, sr).astype(np.float32) if sr != FS else z
        z = z / (np.std(z) + 1e-06)
        waves.append(z)
    n = min((len(z) for z in waves))
    mid = max(0, (n - 180 * FS) // 2)
    end = min(n, mid + 180 * FS)
    channel_sec = []
    for z in waves:
        probs = []
        for st in range(0, end - mid - WIN + 1, WIN):
            feat = ((extract_features(z[mid + st:mid + st + WIN]) - mu) / sd).astype(np.float32)
            with torch.no_grad():
                probs.append(float(torch.softmax(model(torch.from_numpy(feat[None])), 1)[0, 1]))
        channel_sec.append(probs)
    sec_probs = np.quantile(np.asarray(channel_sec), 0.75, axis=0).tolist()
    sec_pred = (np.asarray(sec_probs) >= threshold).astype(int)
    sec_true = np.ones(len(sec_pred), dtype=int)
    if len(sec_pred) >= 3:
        raw = sec_pred.copy()
        sec_pred[1:-1] = (raw[:-2] + raw[1:-1] + raw[2:] >= 2).astype(int)
    external = {'segment': '90101 middle 180 seconds', 'seconds': len(sec_probs), 'accuracy': float((sec_pred == sec_true).mean()), 'recognition_rate': float((sec_pred == 1).mean()), 'false_alarm_rate': 0.0, 'positive_rate': float(sec_pred.mean()), 'start_sample': int(mid), 'end_sample': int(end)}
    bg = bg_cal
    bprob = []
    for seg in bg:
        feat = ((extract_features(seg) - mu) / sd).astype(np.float32)
        with torch.no_grad():
            bprob.append(float(torch.softmax(model(torch.from_numpy(feat[None])), 1)[0, 1]))
    bpred = np.asarray(bprob) >= threshold
    external['background_windows'] = len(bpred)
    external['background_false_alarm_rate'] = float(bpred.mean())
    results['external_90101_middle_3min'] = external
    (ART / '90101_secondwise_results.json').write_text(json.dumps({'threshold': threshold, 'background_source': bg_source, 'rows': [{'second': i, 'probability': float(p), 'prediction': int(q), 'true_label': 1} for i, (p, q) in enumerate(zip(sec_probs, sec_pred))]}, ensure_ascii=False, indent=2))
    torch.save({'state_dict': model.state_dict(), 'mu': mu, 'sd': sd, 'feature_dim': 192, 'threshold': threshold}, ART / 'rmal_crnn_binary.pt')
    (ART / 'metrics.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
    np.save(ART / 'feature_matrix.npy', X)
    print('results=', json.dumps(results, ensure_ascii=False))
if __name__ == '__main__':
    main()
