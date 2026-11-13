from pathlib import Path
import json, copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import pipeline as rp
BASE = Path(__file__).resolve().parents[2]
ART = BASE / 'outputs' / 'range' / 'teacher'
FEAT = BASE / 'outputs' / 'range' / 'channels' / 'features_refined'
RANK = BASE / 'outputs' / 'range' / 'channels' / 'ranking.json'
ART.mkdir(parents=True, exist_ok=True)

def prob(m, x):
    m.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 128):
            out.append(torch.softmax(m(torch.from_numpy(x[i:i + 128])), 1)[:, 1].numpy())
    return np.concatenate(out)

def met(y, p, pr):
    return {'accuracy': float(accuracy_score(y, p)), 'f1': float(f1_score(y, p, zero_division=0)), 'auc': float(roc_auc_score(y, pr)), 'false_alarm_rate': float(((p == 1) & (y == 0)).sum() / max(1, (y == 0).sum())), 'recall': float(((p == 1) & (y == 1)).sum() / max(1, (y == 1).sum())), 'confusion_matrix': confusion_matrix(y, p).tolist()}

def main():
    rp.seed_all(2026)
    ranking = json.loads(RANK.read_text(encoding='utf-8'))
    top = [r['channel'] for r in ranking[:20]]
    extra = ranking[20]['channel']
    Xtr = []
    ytr = []
    Xv = []
    yv = []
    Xt = []
    yt = []
    for c in top:
        q = np.load(FEAT / f'ch{c:02d}.npz', allow_pickle=True)
        X = q['X'].astype(np.float32)
        y = q['y'].astype(np.int64)
        s = q['source']
        for name in ('90101', '90102', '90105'):
            ii = np.where(s == name)[0]
            n = len(ii)
            a = int(0.6 * n)
            b = int(0.8 * n)
            Xtr.extend(X[ii[:a]])
            ytr.extend(y[ii[:a]])
            Xv.extend(X[ii[a:b]])
            yv.extend(y[ii[a:b]])
            Xt.extend(X[ii[b:]])
            yt.extend(y[ii[b:]])
    Xtr = np.asarray(Xtr, np.float32)
    Xv = np.asarray(Xv, np.float32)
    Xt = np.asarray(Xt, np.float32)
    ytr = np.asarray(ytr, np.int64)
    yv = np.asarray(yv, np.int64)
    yt = np.asarray(yt, np.int64)
    mu = Xtr.reshape(-1, 192).mean(0).astype(np.float32)
    sd = np.maximum(Xtr.reshape(-1, 192).std(0), 1e-06).astype(np.float32)
    Xtr = (Xtr - mu) / sd
    Xv = (Xv - mu) / sd
    Xt = (Xt - mu) / sd
    n0 = max(1, (ytr == 0).sum())
    n1 = max(1, (ytr == 1).sum())
    wt = torch.tensor([(n0 + n1) / (2 * n0), (n0 + n1) / (2 * n1)], dtype=torch.float32)
    m = rp.RMALCRNN()
    opt = torch.optim.AdamW(m.parameters(), lr=0.0015, weight_decay=0.0001)
    dl = DataLoader(rp.FeatDS(Xtr, ytr), 128, shuffle=True)
    best = (-1, None)
    for ep in range(16):
        m.train()
        for bx, by in dl:
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(m(bx), by, weight=wt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 2)
            opt.step()
        pv = prob(m, Xv)
        va = accuracy_score(yv, pv >= 0.5)
        print(f'epoch {ep + 1}/16 val_acc={va:.4f}', flush=True)
        if va > best[0]:
            best = (va, copy.deepcopy(m.state_dict()))
    m.load_state_dict(best[1])
    pv = prob(m, Xv)
    threshold = 0.5
    bestrec = -1
    for th in np.linspace(0.05, 0.995, 190):
        pp = pv >= th
        far = ((pp == 1) & (yv == 0)).sum() / max(1, (yv == 0).sum())
        rec = ((pp == 1) & (yv == 1)).sum() / max(1, (yv == 1).sum())
        if far <= 0.01 and rec > bestrec:
            bestrec = rec
            threshold = float(th)
    ptest = prob(m, Xt)
    qtest = ptest >= threshold
    qe = np.load(FEAT / f'ch{extra:02d}.npz', allow_pickle=True)
    Xe = (qe['X'].astype(np.float32) - mu) / sd
    ye = qe['y'].astype(np.int64)
    pe = prob(m, Xe)
    qe_raw = pe >= threshold
    smooth = np.convolve(qe_raw.astype(float), np.ones(3) / 3, mode='same') >= 0.5
    ext_threshold = 0.98
    ext_raw = pe >= ext_threshold
    ext_smooth = np.convolve(ext_raw.astype(float), np.ones(3) / 3, mode='same') >= 0.5
    result = {'protocol': {'90101': '76s onward, 200s', '90102': 'middle 200s', '90105': 'last 400s', 'channels_removed': [9, 15]}, 'domains': ['HHT', 'LOFAR', 'FFT', 'WPT', 'HHSA', 'MFCC'], 'top20_channels': top, 'extra_test_channel': extra, 'threshold': threshold, 'top20_test': met(yt, qtest, ptest), 'extra_test_raw': met(ye, qe_raw, pe), 'extra_test_smoothed': met(ye, smooth, pe), 'extra_test_conservative': {'threshold': ext_threshold, 'raw': met(ye, ext_raw, pe), 'smoothed_3s': met(ye, ext_smooth, pe)}, 'notes': 'Global train-stat scaling only; no per-domain normalization; RMAL-CRNN architecture unchanged.'}
    np.savez_compressed(ART / 'model_splits.npz', X_train=Xtr, y_train=ytr, X_val=Xv, y_val=yv, X_test=Xt, y_test=yt)
    torch.save({'state_dict': m.state_dict(), 'mu': mu, 'sd': sd, 'threshold': threshold, 'top20': top, 'extra_channel': extra, 'domains': result['domains']}, ART / 'model.pt')
    (ART / 'metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    rows = [{'sample': int(i), 'source': str(qe['source'][i]), 'second': int(qe['second'][i]), 'label': int(ye[i]), 'probability': float(pe[i]), 'raw_prediction': int(qe_raw[i]), 'smoothed_prediction': int(smooth[i]), 'conservative_prediction': int(ext_smooth[i])} for i in range(len(ye))]
    (ART / 'extra_test_predictions.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    sec = [r for r in rows if r['source'] == '90101']
    (ART / '90101_secondwise_results.json').write_text(json.dumps({'description': '90101 76s onward 200 one-second samples on extra channel; FAR is measured separately on 90105 background', 'rows': sec, 'background_false_alarm_rate': result['extra_test_conservative']['smoothed_3s']['false_alarm_rate']}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
if __name__ == '__main__':
    main()
