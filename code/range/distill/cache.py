import argparse
from pathlib import Path
import numpy as np
import torch
from models import RMALCRNNTeacher
from common import load_teacher_checkpoint

def infer_logits(model, X, batch_size=256):
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            bx = torch.from_numpy(X[i:i + batch_size]).float()
            out.append(model(bx).cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', default='artifacts/model_splits.npz')
    ap.add_argument('--teacher', default='../teacher/model.pt')
    ap.add_argument('--out', default='artifacts/teacher_logits.npz')
    ap.add_argument('--batch-size', type=int, default=256)
    args = ap.parse_args()
    ck = load_teacher_checkpoint(args.teacher)
    model = RMALCRNNTeacher()
    model.load_state_dict(ck['state_dict'], strict=True)
    q = np.load(args.splits)
    data = {}
    for split in ('train', 'val', 'test'):
        X = q[f'X_{split}'].astype(np.float32)
        logits = infer_logits(model, X, args.batch_size)
        data[f'logits_{split}'] = logits
        print(split, logits.shape)
    dst = Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **data)
    print('written:', dst)
if __name__ == '__main__':
    main()
