import argparse
from pathlib import Path
import numpy as np
import torch
from models import RMALStudentCore
from common import load_teacher_checkpoint, softmax_np, calc_metrics, write_json

def infer(model, X, batch=256):
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(model(torch.from_numpy(X[i:i + batch]).float()).numpy())
    return np.concatenate(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--student', default='artifacts/student/rmal_student_distilled.pt')
    ap.add_argument('--teacher', default='../teacher/model.pt')
    ap.add_argument('--splits', default='artifacts/model_splits.npz')
    ap.add_argument('--extra-feature', default=None, help='optional refined ch20.npz')
    ap.add_argument('--out', default='artifacts/student/evaluation.json')
    args = ap.parse_args()
    sck = torch.load(args.student, map_location='cpu', weights_only=False)
    m = RMALStudentCore()
    m.load_state_dict(sck['state_dict'])
    th = float(sck['threshold'])
    q = np.load(args.splits)
    report = {}
    for split in ('val', 'test'):
        X = q[f'X_{split}'].astype(np.float32)
        y = q[f'y_{split}'].astype(np.int64)
        p = softmax_np(infer(m, X))[:, 1]
        report[split] = calc_metrics(y, p, th)
    if args.extra_feature:
        tck = load_teacher_checkpoint(args.teacher)
        mu = np.asarray(tck['mu'], np.float32)
        sd = np.maximum(np.asarray(tck['sd'], np.float32), 1e-06)
        e = np.load(args.extra_feature, allow_pickle=True)
        X = (e['X'].astype(np.float32) - mu) / sd
        y = e['y'].astype(np.int64)
        p = softmax_np(infer(m, X))[:, 1]
        report['extra_ch20_threshold_student'] = calc_metrics(y, p, th)
        report['extra_ch20_threshold_0p98'] = calc_metrics(y, p, 0.98)
        raw = (p >= 0.98).astype(float)
        sm = np.convolve(raw, np.ones(3) / 3, mode='same') >= 0.5
        report['extra_ch20_0p98_vote3'] = calc_metrics(y, sm.astype(float), 0.5)
    write_json(args.out, report)
    print(report)
if __name__ == '__main__':
    main()
