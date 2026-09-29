"""GroupKFold training. Run from repo root: python3 src/train.py --data crash_competition_data"""
import argparse, os, time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

from dataset import ClipDataset, train_transforms, eval_transforms
from model import VideoModel


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="crash_competition_data")
    p.add_argument("--out", default="models")
    p.add_argument("--backbone", default="resnet50", choices=["resnet50", "resnet18"])
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--subset", type=int, default=0, help="use first N clips (smoke test)")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def run_epoch(model, loader, opt, device, smoothing, train):
    model.train(train)
    crit = torch.nn.BCEWithLogitsLoss()
    tot, n = 0.0, 0
    for x, y, _ in loader:
        x, y = x.to(device), y.float().to(device)
        if train:
            opt.zero_grad()
        logits, _ = model(x)
        ys = y * (1 - smoothing) + 0.5 * smoothing
        loss = crit(logits, ys)
        if train:
            loss.backward()
            opt.step()
        tot += loss.item() * len(x)
        n += len(x)
    return tot / max(n, 1)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ps, ys, ids = [], [], []
    for x, y, cid in loader:
        logits, _ = model(x.to(device))
        ps += torch.sigmoid(logits).cpu().tolist()
        ys += y.tolist()
        ids += list(cid)
    return np.array(ps), np.array(ys), ids


def main():
    a = parse()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    # ponytail: single GPU/CPU, no DDP — multi-GPU only if quota forces it
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    os.makedirs(a.out, exist_ok=True)

    labels = pd.read_csv(os.path.join(a.data, "train_labels.csv"))
    if a.subset:
        labels = labels.iloc[:a.subset].reset_index(drop=True)
    groups = labels["group_id"].values

    oof = pd.DataFrame({"clip_id": labels["clip_id"], "label": labels["label"],
                        "pred": np.nan})
    gkf = GroupKFold(n_splits=min(a.folds, len(np.unique(groups))))
    aucs = []
    for fold, (tr, va) in enumerate(gkf.split(labels, labels["label"], groups)):
        tr_ds = ClipDataset(a.data, "train", a.frames, train_transforms(), True,
                            ids=labels.iloc[tr]["clip_id"], seed=a.seed + fold)
        # keep labels aligned: rebuild df with labels
        tr_ds.df = labels.iloc[tr].reset_index(drop=True)
        tr_ds.has_label = True
        va_ds = ClipDataset(a.data, "train", a.frames, eval_transforms(), False,
                            ids=labels.iloc[va]["clip_id"], seed=999)
        va_ds.df = labels.iloc[va].reset_index(drop=True)
        va_ds.has_label = True
        tr_ld = DataLoader(Subset(tr_ds, range(len(tr_ds))), batch_size=a.batch_size,
                           shuffle=True, num_workers=a.num_workers)
        va_ld = DataLoader(va_ds, batch_size=a.batch_size * 2, num_workers=a.num_workers)

        model = VideoModel(a.backbone, not a.no_pretrained).to(device)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=a.lr, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)

        best, bad, best_state = 0.0, 0, None
        for ep in range(a.epochs):
            t0 = time.time()
            tl = run_epoch(model, tr_ld, opt, device, a.label_smoothing, True)
            sched.step()
            pv, yv, _ = predict(model, va_ld, device)
            auc = roc_auc_score(yv, pv)
            print(f"fold {fold} ep {ep}: loss {tl:.4f} val_auc {auc:.4f} ({time.time()-t0:.0f}s)")
            if auc > best:
                best, bad, best_state = auc, 0, {k: v.cpu() for k, v in model.head.state_dict().items()}
            else:
                bad += 1
                if bad >= a.patience:
                    break
        model.head.load_state_dict(best_state)
        torch.save({"head": best_state, "backbone": a.backbone, "frames": a.frames,
                    "fold": fold, "auc": best}, f"{a.out}/fold{fold}.pt")
        pv, yv, vids = predict(model, va_ld, device)
        oof.loc[oof.clip_id.isin(vids), "pred"] = pv
        aucs.append(best)
        print(f"fold {fold} best_auc {best:.4f}")

    # reorder oof to match vids order safely
    oof.to_csv(f"{a.out}/oof.csv", index=False)
    print(f"mean_fold_auc {[round(x, 4) for x in aucs]} -> {np.mean(aucs):.4f}")
    print(f"pooled_oof_auc {roc_auc_score(oof.label, oof.pred):.4f}")


if __name__ == "__main__":
    main()
