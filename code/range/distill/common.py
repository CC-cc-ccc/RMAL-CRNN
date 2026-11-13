import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

def seed_all(seed=2026):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_teacher_checkpoint(path, device='cpu'):
    return torch.load(path, map_location=device, weights_only=False)

def calc_metrics(y, prob, threshold):
    y = np.asarray(y, dtype=np.int64)
    prob = np.asarray(prob, dtype=np.float64)
    pred = prob >= threshold
    out = {'threshold': float(threshold), 'accuracy': float(accuracy_score(y, pred)), 'f1': float(f1_score(y, pred, zero_division=0)), 'false_alarm_rate': float(((pred == 1) & (y == 0)).sum() / max(1, (y == 0).sum())), 'recall': float(((pred == 1) & (y == 1)).sum() / max(1, (y == 1).sum())), 'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1]).tolist()}
    if len(np.unique(y)) == 2:
        out['auc'] = float(roc_auc_score(y, prob))
    else:
        out['auc'] = None
    return out

def select_far_threshold(y, prob, max_far=0.01):
    y = np.asarray(y, dtype=np.int64)
    prob = np.asarray(prob, dtype=np.float64)
    best = None
    for th in np.linspace(0.01, 0.995, 396):
        pred = prob >= th
        far = ((pred == 1) & (y == 0)).sum() / max(1, (y == 0).sum())
        rec = ((pred == 1) & (y == 1)).sum() / max(1, (y == 1).sum())
        if far <= max_far and (best is None or rec > best[0]):
            best = (float(rec), float(th))
    return 0.5 if best is None else best[1]

def softmax_np(logits):
    x = np.asarray(logits, np.float64)
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
