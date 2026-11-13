from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score
import torch
from torch.utils.data import DataLoader
import pipeline as rp
BASE = Path(__file__).resolve().parents[2]
DATA = BASE / 'data' / 'range'
OUT = BASE / 'outputs' / 'range' / 'channels'
WAV = OUT / 'wav'
FEAT = OUT / 'features'
FS = rp.FS
WIN = rp.WIN
FRAME_BYTES = 108 + 1000 * 32 * 3 + 6
VALID = [c for c in range(1, 33) if c not in (9, 15)]

def decode_segment(path: Path, out_dir: Path, start_frame: int, n_frames: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    chs = [c - 1 for c in VALID]
    blocks = [[] for _ in chs]
    with path.open('rb') as f:
        f.seek(start_frame * FRAME_BYTES)
        for _ in range(n_frames):
            raw = f.read(FRAME_BYTES)
            if len(raw) != FRAME_BYTES:
                break
            b = np.frombuffer(raw[108:108 + 1000 * 32 * 3], dtype=np.uint8).reshape(1000, 32, 3).astype(np.int32)
            v = b[:, :, 0] << 16 | b[:, :, 1] << 8 | b[:, :, 2]
            v[v >= 8388608] -= 16777216
            for j, c in enumerate(chs):
                blocks[j].append((v[:, c].astype(np.float32) / (1 << 23)).astype(np.float32))
    paths = []
    for j, c in enumerate(VALID):
        p = out_dir / f'ch{c:02d}.wav'
        wavfile.write(p, 10000, np.concatenate(blocks[j]).astype(np.float32))
        paths.append(p)
    return paths

def dat_segments():
    specs = {'90101': (DATA / '90101.dat', 76.0, 200.0), '90102': (DATA / '90102.dat', None, 200.0), '90105': (DATA / 'dataset1' / 'noise' / '90105.dat', None, 400.0)}
    all_paths = {}
    protocol = {}
    for name, (path, start, dur) in specs.items():
        if not path.exists():
            raise FileNotFoundError(path)
        nf = path.stat().st_size // FRAME_BYTES
        take = int(round(dur * 10))
        sf = int(round(start * 10)) if start is not None else nf - take if name == '90105' else (nf - take) // 2
        all_paths[name] = decode_segment(path, WAV / name, sf, take)
        protocol[name] = {'path': str(path), 'total_frames': int(nf), 'start_frame': int(sf), 'end_frame': int(sf + take), 'start_sec': sf / 10, 'duration_sec': take / 10, 'sample_rate': 10000, 'channels_removed': [9, 15]}
    return (all_paths, protocol)

def load_segment(p):
    x, sr = rp.read_wav(p)
    x = rp.resample_poly(x, FS, sr).astype(np.float32) if sr != FS else x
    n = len(x) // WIN
    return [rp.extract_features(x[i * WIN:(i + 1) * WIN]) for i in range(n)]

def split_indices(n):
    a = int(n * 0.6)
    b = int(n * 0.8)
    return (np.arange(n) < a, (np.arange(n) >= a) & (np.arange(n) < b), np.arange(n) >= b)

def rank_channels(paths, max_ch=None, max_windows=80):
    FEAT.mkdir(parents=True, exist_ok=True)
    rows = []
    data = {}
    for c in VALID:
        cached = FEAT / f'ch{c:02d}.npz'
        if cached.exists():
            q = np.load(cached, allow_pickle=True)
            X = q['X'].astype(np.float32)
            y = q['y']
            src = q['source']
            sec = q['second']
        else:
            xs = []
            ys = []
            src = []
            sec = []
            for name, label in [('90101', 1), ('90102', 1), ('90105', 0)]:
                pp = [p for p in paths[name] if p.name == f'ch{c:02d}.wav']
                if not pp:
                    raise FileNotFoundError(f'missing decoded channel {c} for {name}')
                f = np.asarray(load_segment(pp[0]), dtype=np.float32)
                if len(f) > max_windows:
                    f = f[:max_windows]
                xs.append(f)
                ys.append(np.full(len(f), label))
                src += [name] * len(f)
                sec += list(range(len(f)))
            X = np.concatenate(xs)
            y = np.concatenate(ys)
            src = np.asarray(src)
            sec = np.asarray(sec)
            np.savez_compressed(cached, X=X, y=y, source=src, second=sec)
        tr = np.zeros(len(y), bool)
        va = np.zeros(len(y), bool)
        te = np.zeros(len(y), bool)
        for name in ['90101', '90102', '90105']:
            m = src == name
            i = np.where(m)[0]
            a, b, d = split_indices(len(i))
            tr[i[a]] = 1
            va[i[b]] = 1
            te[i[d]] = 1

        def flat(z):
            return np.concatenate([z.mean(1), z.std(1)], axis=1)
        A = flat(X)
        mu = A[tr].mean(0)
        sd = np.where(A[tr].std(0) < 1e-06, 1, A[tr].std(0))
        A = (A - mu) / sd
        lr = LogisticRegression(max_iter=300, class_weight='balanced', C=0.5, solver='liblinear').fit(A[tr], y[tr])
        pv = lr.predict_proba(A[va])[:, 1]
        pt = lr.predict_proba(A[te])[:, 1]
        val_auc = float(roc_auc_score(y[va], pv))
        test_auc = float(roc_auc_score(y[te], pt))
        val_acc = float(accuracy_score(y[va], pv >= 0.5))
        test_acc = float(accuracy_score(y[te], pt >= 0.5))
        rows.append({'channel': c, 'val_auc': val_auc, 'test_auc': test_auc, 'val_accuracy': val_acc, 'test_accuracy': test_acc, 'samples': len(y)})
        data[c] = {'X': X, 'y': y, 'source': src, 'train': tr, 'val': va, 'test': te}
        print(f'channel {c:02d}: val_auc={val_auc:.3f} test_auc={test_auc:.3f}', flush=True)
    rows.sort(key=lambda r: (r['val_auc'], r['test_auc']), reverse=True)
    return (rows, data)

def model_probs(model, X):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 128):
            out.extend(torch.softmax(model(torch.from_numpy(X[i:i + 128])), 1)[:, 1].numpy())
    return np.asarray(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=24)
    ap.add_argument('--max-rank-channels', type=int, default=30)
    ap.add_argument('--max-windows', type=int, default=80)
    args = ap.parse_args()
    rp.seed_all(2026)
    OUT.mkdir(parents=True, exist_ok=True)
    paths, protocol = dat_segments()
    json.dump(protocol, open(OUT / 'protocol.json', 'w'), ensure_ascii=False, indent=2)
    ranking, data = rank_channels(paths, args.max_rank_channels, args.max_windows)
    json.dump(ranking, open(OUT / 'channel_ranking.json', 'w'), ensure_ascii=False, indent=2)
    with open(OUT / 'channel_ranking.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(ranking[0]))
        w.writeheader()
        w.writerows(ranking)
    top20 = [r['channel'] for r in ranking[:20]]
    extra = ranking[20]['channel']
    json.dump({'train_channels': top20, 'extra_test_channel': extra, 'excluded_channels': [9, 15]}, open(OUT / 'selected_channels.json', 'w'), ensure_ascii=False, indent=2)
    Xtr = []
    ytr = []
    Xva = []
    yva = []
    Xte = []
    yte = []
    for c in top20:
        d = data[c]
        for dest, mask, xx, yy in [(None, d['train'], Xtr, ytr), (None, d['val'], Xva, yva), (None, d['test'], Xte, yte)]:
            xx.extend(d['X'][mask])
            yy.extend(d['y'][mask])
    Xtr = np.asarray(Xtr, np.float32)
    Xva = np.asarray(Xva, np.float32)
    Xte = np.asarray(Xte, np.float32)
    ytr = np.asarray(ytr)
    yva = np.asarray(yva)
    yte = np.asarray(yte)
    mu = Xtr.reshape(-1, 192).mean(0)
    sd = np.where(Xtr.reshape(-1, 192).std(0) < 1e-06, 1, Xtr.reshape(-1, 192).std(0)).astype(np.float32)
    Xtr = (Xtr - mu) / sd
    Xva = (Xva - mu) / sd
    Xte = (Xte - mu) / sd
    tr = DataLoader(rp.FeatDS(Xtr, ytr), 128, shuffle=True)
    model = rp.RMALCRNN()
    opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0001)
    wt = torch.tensor([1.0, 1.0])
    best = (-1, None)
    for ep in range(args.epochs):
        model.train()
        for bx, by in tr:
            opt.zero_grad()
            logits = model(bx)
            ce = torch.nn.functional.cross_entropy(logits, by, weight=wt)
            p = torch.softmax(logits, 1)[:, 1]
            pt = torch.where(by == 1, p, 1 - p)
            focal = ((1 - pt) ** 2 * torch.nn.functional.cross_entropy(logits, by, reduction='none')).mean()
            (ce + 0.25 * focal).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            opt.step()
        pv = model_probs(model, Xva)
        acc = accuracy_score(yva, pv >= 0.5)
        print(f'epoch {ep + 1}/{args.epochs} val_acc={acc:.3f}', flush=True)
        if acc > best[0]:
            best = (acc, {k: v.detach().cpu() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])
    pv = model_probs(model, Xva)
    threshold = 0.5
    best_rec = -1
    for th in np.linspace(0.05, 0.995, 190):
        q = pv >= th
        far = ((q == 1) & (yva == 0)).sum() / max(1, (yva == 0).sum())
        rec = ((q == 1) & (yva == 1)).sum() / max(1, (yva == 1).sum())
        if far <= 0.01 and rec > best_rec:
            best_rec = rec
            threshold = float(th)
    pred = model_probs(model, Xte) >= threshold
    test = {'accuracy': float(accuracy_score(yte, pred)), 'f1': float(f1_score(yte, pred)), 'auc': float(roc_auc_score(yte, model_probs(model, Xte))), 'false_alarm_rate': float(((pred == 1) & (yte == 0)).sum() / max(1, (yte == 0).sum())), 'recall': float(((pred == 1) & (yte == 1)).sum() / max(1, (yte == 1).sum())), 'confusion_matrix': confusion_matrix(yte, pred).tolist()}
    d = data[extra]
    Xe = ((d['X'] - mu) / sd).astype(np.float32)
    pe = model_probs(model, Xe)
    raw = pe >= threshold
    smooth = np.convolve(raw.astype(float), np.ones(3) / 3, mode='same') >= 0.5
    ye = d['y']
    extra_m = {'channel': extra, 'threshold': threshold, 'raw_accuracy': float(accuracy_score(ye, raw)), 'smoothed_accuracy': float(accuracy_score(ye, smooth)), 'raw_false_alarm_rate': float(((raw == 1) & (ye == 0)).sum() / max(1, (ye == 0).sum())), 'smoothed_false_alarm_rate': float(((smooth == 1) & (ye == 0)).sum() / max(1, (ye == 0).sum())), 'raw_recall': float(((raw == 1) & (ye == 1)).sum() / max(1, (ye == 1).sum())), 'smoothed_recall': float(((smooth == 1) & (ye == 1)).sum() / max(1, (ye == 1).sum())), 'confusion_matrix_smoothed': confusion_matrix(ye, smooth).tolist()}
    torch.save({'state_dict': model.state_dict(), 'mu': mu, 'sd': sd, 'threshold': threshold, 'top20': top20, 'extra_channel': extra}, OUT / 'rmal_crnn_top20.pt')
    np.savez_compressed(OUT / 'model_splits.npz', X_train=Xtr, y_train=ytr, X_val=Xva, y_val=yva, X_test=Xte, y_test=yte)
    json.dump({'threshold': threshold, 'top20_test': test, 'extra_test': extra_m}, open(OUT / 'metrics.json', 'w'), ensure_ascii=False, indent=2)
    json.dump([{'sample': i, 'source': str(d['source'][i]), 'second': int(d['second'][i]), 'label': int(ye[i]), 'probability': float(pe[i]), 'raw_prediction': int(raw[i]), 'smoothed_prediction': int(smooth[i])} for i in range(len(ye))], open(OUT / 'extra_test_predictions.json', 'w'), ensure_ascii=False, indent=2)
    print('selected top20=', top20, 'extra=', extra, flush=True)
    print(json.dumps({'top20_test': test, 'extra_test': extra_m}, ensure_ascii=False), flush=True)
if __name__ == '__main__':
    main()
