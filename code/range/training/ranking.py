from pathlib import Path
import json, csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
BASE = Path(__file__).resolve().parents[2]
OUT = BASE / 'outputs' / 'range' / 'channels'
FEAT = OUT / 'features_refined'
VALID = [c for c in range(1, 33) if c not in (9, 15)]
rows = []
for c in VALID:
    q = np.load(FEAT / f'ch{c:02d}.npz', allow_pickle=True)
    X = q['X'].astype(np.float32)
    y = q['y'].astype(int)
    s = q['source']
    A = np.concatenate([X.mean(1), X.std(1)], axis=1)
    tr = np.zeros(len(y), bool)
    va = np.zeros(len(y), bool)
    te = np.zeros(len(y), bool)
    for name in ('90101', '90102', '90105'):
        ii = np.where(s == name)[0]
        n = len(ii)
        a = int(0.6 * n)
        b = int(0.8 * n)
        tr[ii[:a]] = 1
        va[ii[a:b]] = 1
        te[ii[b:]] = 1
    m = LogisticRegression(max_iter=300, class_weight='balanced', C=0.5, solver='liblinear').fit(A[tr], y[tr])
    pv = m.predict_proba(A[va])[:, 1]
    pt = m.predict_proba(A[te])[:, 1]
    rows.append({'channel': c, 'val_accuracy': float(accuracy_score(y[va], pv >= 0.5)), 'val_auc': float(roc_auc_score(y[va], pv)), 'test_accuracy': float(accuracy_score(y[te], pt >= 0.5)), 'test_auc': float(roc_auc_score(y[te], pt)), 'samples': int(len(y))})
rows.sort(key=lambda r: (r['val_auc'], r['test_auc']), reverse=True)
(OUT / 'ranking.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
with (OUT / 'ranking.csv').open('w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
(OUT / 'channels.json').write_text(json.dumps({'train_channels': [r['channel'] for r in rows[:20]], 'extra_test_channel': rows[20]['channel'], 'excluded_channels': [9, 15]}, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'top20': [r['channel'] for r in rows[:20]], 'extra': rows[20]['channel']}, ensure_ascii=False, indent=2))
