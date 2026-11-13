import argparse, copy
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from models import RMALStudentCore, parameter_count
from common import seed_all, softmax_np, calc_metrics, select_far_threshold, load_teacher_checkpoint, write_json

class DistillDS(Dataset):

    def __init__(self, X, y, teacher_logits):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.t = torch.from_numpy(teacher_logits).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (self.X[i], self.y[i], self.t[i])

def infer(model, X, batch_size=256):
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            out.append(model(torch.from_numpy(X[i:i + batch_size]).float()).cpu().numpy())
    return np.concatenate(out, axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', default='artifacts/model_splits.npz')
    ap.add_argument('--teacher-logits', default='artifacts/teacher_logits.npz')
    ap.add_argument('--teacher', default='../teacher/model.pt')
    ap.add_argument('--out-dir', default='artifacts/student')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--weight-decay', type=float, default=0.0001)
    ap.add_argument('--temperature', type=float, default=4.0)
    ap.add_argument('--alpha-hard', type=float, default=0.35, help='hard CE weight; KD weight is 1-alpha-hard')
    ap.add_argument('--max-val-far', type=float, default=0.01)
    ap.add_argument('--seed', type=int, default=2026)
    args = ap.parse_args()
    seed_all(args.seed)
    q = np.load(args.splits)
    tq = np.load(args.teacher_logits)
    Xtr, ytr = (q['X_train'].astype(np.float32), q['y_train'].astype(np.int64))
    Xv, yv = (q['X_val'].astype(np.float32), q['y_val'].astype(np.int64))
    Xt, yt = (q['X_test'].astype(np.float32), q['y_test'].astype(np.int64))
    ttr = tq['logits_train'].astype(np.float32)
    ds = DistillDS(Xtr, ytr, ttr)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = RMALStudentCore()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n0, n1 = (max(1, int((ytr == 0).sum())), max(1, int((ytr == 1).sum())))
    cls_w = torch.tensor([(n0 + n1) / (2 * n0), (n0 + n1) / (2 * n1)], dtype=torch.float32)
    T = float(args.temperature)
    alpha = float(args.alpha_hard)
    best = (-1.0, None, None)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    for ep in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for bx, by, bt in dl:
            opt.zero_grad()
            s = model(bx)
            hard = F.cross_entropy(s, by, weight=cls_w)
            kd = F.kl_div(F.log_softmax(s / T, dim=1), F.softmax(bt / T, dim=1), reduction='batchmean') * (T * T)
            loss = alpha * hard + (1.0 - alpha) * kd
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total += float(loss.item()) * len(by)
        vlogits = infer(model, Xv)
        vprob = softmax_np(vlogits)[:, 1]
        th = select_far_threshold(yv, vprob, args.max_val_far)
        vm = calc_metrics(yv, vprob, th)
        history.append({'epoch': ep, 'loss': total / len(ds), 'val': vm})
        score = vm['recall'] if vm['false_alarm_rate'] <= args.max_val_far else -1.0
        print(f"epoch={ep:02d} loss={total / len(ds):.6f} val_acc={vm['accuracy']:.4f} val_far={vm['false_alarm_rate']:.4f} val_rec={vm['recall']:.4f} th={th:.4f}")
        if score > best[0]:
            best = (score, copy.deepcopy(model.state_dict()), th)
    if best[1] is None:
        raise RuntimeError('no valid student checkpoint selected')
    model.load_state_dict(best[1])
    threshold = float(best[2])
    test_prob = softmax_np(infer(model, Xt))[:, 1]
    test_metrics = calc_metrics(yt, test_prob, threshold)
    teacher_ck = load_teacher_checkpoint(args.teacher)
    ckpt = {'state_dict': model.state_dict(), 'mu': np.asarray(teacher_ck['mu'], np.float32), 'sd': np.asarray(teacher_ck['sd'], np.float32), 'threshold': threshold, 'temperature': T, 'alpha_hard': alpha, 'input_shape': [1, int(Xtr.shape[1]), 192], 'feature_order': ['HHT', 'LOFAR', 'FFT', 'WPT', 'HHSA', 'MFCC'], 'student_params': parameter_count(model)}
    torch.save(ckpt, out_dir / 'rmal_student_distilled.pt')
    report = {'student_params': parameter_count(model), 'teacher_checkpoint_bytes': Path(args.teacher).stat().st_size, 'temperature': T, 'alpha_hard': alpha, 'selected_threshold': threshold, 'internal_test': test_metrics, 'history': history}
    write_json(out_dir / 'student_metrics.json', report)
    print('saved:', out_dir / 'rmal_student_distilled.pt')
    print('test:', test_metrics)
if __name__ == '__main__':
    main()
