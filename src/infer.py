"""TTA inference + fold ensemble. Run: python3 src/infer.py --data crash_competition_data"""
import argparse, glob, os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import ClipDataset, eval_transforms
from model import VideoModel


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="crash_competition_data")
    p.add_argument("--models", default="models")
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--crops", type=int, default=4, help="temporal TTA crops")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-clips", type=int, default=0, help="subset for smoke test")
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


@torch.no_grad()
def main():
    a = parse()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpts = sorted(glob.glob(f"{a.models}/fold*.pt"))
    assert ckpts, f"no checkpoints in {a.models}/"
    sample = pd.read_csv(os.path.join(a.data, "sample_submission.csv"))
    if a.max_clips:
        sample = sample.iloc[:a.max_clips]
    ids = sample["clip_id"].tolist()

    acc = np.zeros(len(ids))
    n = 0
    for ck in ckpts:
        c = torch.load(ck, map_location="cpu", weights_only=True)
        for crop in range(a.crops):
            for flip in (False, True):
                ds = ClipDataset(a.data, "test", c["frames"], eval_transforms(hflip=flip),
                                 True, ids=ids, seed=1000 + crop)
                ld = DataLoader(ds, batch_size=a.batch_size, num_workers=a.num_workers)
                m = VideoModel(c["backbone"], pretrained=False).to(device)
                m.head.load_state_dict(c["head"])
                m.eval()
                ps = []
                for x, _ in ld:
                    logits, _ = m(x.to(device))
                    ps += torch.sigmoid(logits).cpu().tolist()
                acc += np.array(ps)
                n += 1
    sample["label"] = (acc / n).clip(1e-6, 1 - 1e-6)
    sample[["clip_id", "label"]].to_csv(a.out, index=False)
    print(f"wrote {a.out} ({len(sample)} rows, {len(ckpts)} folds x {a.crops} crops x 2 flips)")


if __name__ == "__main__":
    main()
