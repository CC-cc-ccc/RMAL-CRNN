import argparse
from pathlib import Path
import numpy as np
from common import load_teacher_checkpoint
SOURCES = ('90101', '90102', '90105')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features-dir', required=True)
    ap.add_argument('--teacher', default='../teacher/model.pt')
    ap.add_argument('--out', default='artifacts/model_splits.npz')
    args = ap.parse_args()
    ck = load_teacher_checkpoint(args.teacher)
    top20 = [int(x) for x in ck['top20']]
    mu = np.asarray(ck['mu'], np.float32)
    sd = np.maximum(np.asarray(ck['sd'], np.float32), 1e-06)
    root = Path(args.features_dir)
    acc = {k: {'X': [], 'y': []} for k in ('train', 'val', 'test')}
    for ch in top20:
        p = root / f'ch{ch:02d}.npz'
        if not p.exists():
            raise FileNotFoundError(p)
        q = np.load(p, allow_pickle=True)
        X = q['X'].astype(np.float32)
        y = q['y'].astype(np.int64)
        s = q['source']
        if X.ndim != 3 or X.shape[-1] != 192:
            raise RuntimeError(f'unexpected feature shape {X.shape} in {p}')
        for name in SOURCES:
            ii = np.where(s == name)[0]
            n = len(ii)
            a, b = (int(0.6 * n), int(0.8 * n))
            slices = {'train': ii[:a], 'val': ii[a:b], 'test': ii[b:]}
            for split, jj in slices.items():
                acc[split]['X'].append(X[jj])
                acc[split]['y'].append(y[jj])
    out = {}
    for split in ('train', 'val', 'test'):
        X = np.concatenate(acc[split]['X'], axis=0)
        y = np.concatenate(acc[split]['y'], axis=0)
        X = ((X - mu) / sd).astype(np.float32)
        out[f'X_{split}'] = X
        out[f'y_{split}'] = y
        print(split, X.shape, {int(c): int((y == c).sum()) for c in np.unique(y)})
    dst = Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **out)
    print('written:', dst)
if __name__ == '__main__':
    main()
