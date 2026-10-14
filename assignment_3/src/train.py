"""Training: (a) a prompt-initialised linear probe, (b) optional LoRA fine-tuning.

Default first move is **mode=probe**: with strong frozen features, a linear head
initialised from the *text* prototypes (prompt ensembling as the classifier
initialisation) trained with proper cross-validation takes a couple of minutes
and captures most of the available headroom. Only if image->category top-1 on
validation looks stuck does it pay to move to **mode=lora**, which unfreezes the
vision tower through low-rank adapters while the text encoder - and therefore the
classifier weights - stays frozen.

Both modes use the practices that matter on a 2000-image dataset: cosine LR with
warmup, weight decay, label smoothing, gradient clipping, bf16 autocast,
deterministic seeds, honest k-fold validation, and checkpoint selection on
validation accuracy - never on the leaderboard.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import catalog
import config
import dataset as ds
import features
import imagemap
import prompts


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=None)
    p.add_argument("--mode", choices=["probe", "lora"], default="probe")
    p.add_argument("--backbone", default=config.DEFAULT_BACKBONE, choices=sorted(config.BACKBONES))
    p.add_argument(
        "--views", nargs="+", default=list(config.DEFAULT_VIEWS), choices=list(config.EVAL_VIEWS)
    )
    p.add_argument("--out", default="models")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=20, help="probe epochs / lora epochs")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3, help="probe: 1e-3, lora: 1e-4")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.1, help="fraction of steps used for warmup")
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.999, help="probe only, 0 disables")
    p.add_argument("--logit-scale", type=float, default=100.0)
    p.add_argument(
        "--nce-weight", type=float, default=0.5, help="lora: weight of the CLIP contrastive term"
    )
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=32.0)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# shared training utilities
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_scheduler(opt, warmup_steps: int, total_steps: int):
    """Linear warmup into cosine decay - the standard recipe for short fine-tunes."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


class ModelEMA:
    """Exponential moving average of the weights.

    On 2000 images the raw weights bounce around between epochs; the EMA copy is
    consistently a little better and is what gets validated and shipped.
    """

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for key, value in model.state_dict().items():
            shadow = self.shadow[key]
            if value.dtype.is_floating_point:
                shadow.mul_(self.decay).add_(value.detach().float(), alpha=1.0 - self.decay)
            else:
                shadow.copy_(value.detach().float())

    def state_dict(self, model: nn.Module) -> dict:
        ref = model.state_dict()
        return {
            key: (self.shadow[key].to(ref[key].dtype) if key in self.shadow else ref[key])
            for key in ref
        }


def amp_context(precision: str, device: torch.device):
    """bf16 on CUDA when available, otherwise a no-op autocast."""
    if device.type != "cuda" or precision == "fp32":
        return torch.autocast("cpu", enabled=False)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def grad_scaler(precision: str, device: torch.device):
    """fp16 needs a scaler; bf16 does not."""
    enabled = device.type == "cuda" and precision == "fp16"
    return torch.amp.GradScaler("cuda", enabled=enabled)


# --------------------------------------------------------------------------- #
# mode 1: prompt-initialised linear probe on frozen features
# --------------------------------------------------------------------------- #
class PromptHead(nn.Module):
    """Linear classifier initialised from the category prompt prototypes.

    Starting at the zero-shot solution means the model never spends an epoch
    recovering from a random init; it can only improve on zero-shot by what it
    learns from the 2000 training images.
    """

    def __init__(self, prototypes: np.ndarray, logit_scale: float = 100.0):
        super().__init__()
        weight = torch.from_numpy(np.asarray(prototypes, dtype=np.float32))
        self.weight = nn.Parameter(weight.clone())
        self.bias = nn.Parameter(torch.zeros(weight.shape[0]))
        self.register_buffer("scale", torch.tensor(float(logit_scale)))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.scale * F.normalize(feats, dim=-1) @ F.normalize(self.weight, dim=-1).T


def stratified_folds(labels: np.ndarray, n_folds: int, seed: int):
    """Stratified k-fold by class (every class has ~20 training images)."""
    from sklearn.model_selection import StratifiedKFold

    dummy = np.zeros(len(labels))
    return list(StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(dummy, labels))


def top1(logits: np.ndarray, labels: np.ndarray) -> float:
    return float((logits.argmax(axis=1) == labels).mean())


def train_probe(args) -> dict:
    """k-fold prompt-initialised probe; saves one head per fold to ensemble."""
    set_seed(args.seed)
    root = config.resolve_data_dir(args.data)
    device = features.pick_device(args.device)
    out_dir = Path(args.out) / "probe"
    out_dir.mkdir(parents=True, exist_ok=True)

    cats = catalog.load_categories(root)
    train = catalog.load_images(root, "train")
    feats, _ = features.encode_images(
        args.backbone,
        train["image_id"].tolist(),
        train["path"].tolist(),
        views=tuple(args.views),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        precision=args.precision,
    )
    labels = imagemap.labels_from_frame(train, cats["category_id"].tolist())
    proto, _ = prompts.category_prototype_matrix(args.backbone, cats, device=device, precision=args.precision)

    x = torch.from_numpy(feats).float()
    y = torch.from_numpy(labels).long()

    history, fold_scores = [], []
    for fold, (train_idx, val_idx) in enumerate(stratified_folds(labels, args.folds, args.seed)):
        head = PromptHead(proto, logit_scale=args.logit_scale).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        xtr, ytr = x[train_idx].to(device), y[train_idx].to(device)
        xva = x[val_idx].to(device)
        yva = y[val_idx].cpu().numpy()

        steps_per_epoch = max(1, math.ceil(len(train_idx) / args.batch_size / args.grad_accum))
        total_steps = steps_per_epoch * args.epochs
        sched = make_scheduler(opt, int(args.warmup * total_steps), total_steps)
        ema = ModelEMA(head, args.ema_decay) if args.ema_decay > 0 else None

        best = {
            "acc": -1.0,
            "epoch": -1,
            "state": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
        }
        for epoch in range(args.epochs):
            head.train()
            perm = torch.randperm(len(xtr), device=device).tolist()
            opt.zero_grad(set_to_none=True)
            running, n_batches = 0.0, 0
            for step, start in enumerate(range(0, len(perm), args.batch_size), start=1):
                idx = perm[start : start + args.batch_size]
                logits = head(xtr[idx])
                loss = F.cross_entropy(
                    logits, ytr[idx], label_smoothing=args.label_smoothing
                )
                loss.backward()
                running += float(loss)
                n_batches += 1
                if step % args.grad_accum == 0 or start + args.batch_size >= len(perm):
                    if args.clip_grad > 0:
                        nn.utils.clip_grad_norm_(head.parameters(), args.clip_grad)
                    opt.step()
                    sched.step()
                    opt.zero_grad(set_to_none=True)
                    if ema is not None:
                        ema.update(head)

            state = ema.state_dict(head) if ema is not None else head.state_dict()
            working = {k: v.detach().clone() for k, v in head.state_dict().items()}
            head.load_state_dict(state)
            head.eval()
            with torch.no_grad():
                val_logits = head(xva).float().cpu().numpy()
            head.load_state_dict(working)

            acc = top1(val_logits, yva)
            mean_loss = running / max(n_batches, 1)
            history.append(
                {"fold": fold, "epoch": epoch, "loss": mean_loss, "val_acc": acc}
            )
            print(f"[probe] fold {fold} epoch {epoch:02d} loss {mean_loss:.4f} val_acc {acc:.4f}")
            if acc > best["acc"]:
                best = {
                    "acc": acc,
                    "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in state.items()},
                }

        np.savez(
            out_dir / f"probe_fold{fold}.npz",
            weight=best["state"]["weight"].numpy(),
            bias=best["state"]["bias"].numpy(),
            val_acc=best["acc"],
            backbone=args.backbone,
            scale=args.logit_scale,
        )
        fold_scores.append(best["acc"])
        print(f"[probe] fold {fold} best val_acc {best['acc']:.4f} (epoch {best['epoch']})")

    np.savez(
        out_dir / "zero_shot.npz",
        weight=proto,
        bias=np.zeros(len(cats), dtype=np.float32),
        val_acc=float("nan"),
        backbone=args.backbone,
        scale=args.logit_scale,
    )
    report = {
        "mode": "probe",
        "backbone": args.backbone,
        "views": list(args.views),
        "fold_val_acc": fold_scores,
        "mean_val_acc": float(np.mean(fold_scores)),
        "heads": [str(out_dir / f"probe_fold{f}.npz") for f in range(args.folds)],
        "zero_shot_head": str(out_dir / "zero_shot.npz"),
    }
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"[probe] mean fold accuracy {report['mean_val_acc']:.4f} -> {out_dir}")
    return report


# --------------------------------------------------------------------------- #
# mode 2: LoRA fine-tuning of the vision tower against frozen text prototypes
# --------------------------------------------------------------------------- #
# ``out_proj`` is deliberately absent: nn.MultiheadAttention reaches into
# ``self.out_proj.weight`` on its fast path, so it cannot be wrapped.
LORA_TARGETS = {"c_fc", "c_proj", "fc1", "fc2", "proj", "linear", "dense"}


class LoRALinear(nn.Module):
    """``y = W x + b + (alpha / r) * B A x`` with the pretrained ``W`` frozen."""

    @property
    def weight(self):
        """Delegate to the frozen base layer (MultiheadAttention reads this)."""
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for param in self.base.parameters():
            param.requires_grad_(False)
        self.rank = int(rank)
        self.scaling = float(alpha) / max(rank, 1)
        self.lora_a = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        return out + (self.drop(x) @ self.lora_a.t() @ self.lora_b.t()) * self.scaling


def apply_lora(module: nn.Module, rank: int, alpha: float, dropout: float) -> int:
    """Replace eligible ``nn.Linear`` layers in place; returns how many."""
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in LORA_TARGETS:
            setattr(module, name, LoRALinear(child, rank, alpha, dropout))
            replaced += 1
        else:
            replaced += apply_lora(child, rank, alpha, dropout)
    return replaced


def lora_state_dict(model: nn.Module) -> dict:
    """Only the adapter tensors (a few MB) instead of the whole backbone."""
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if "lora_" in key
    }


def load_lora(model: nn.Module, path: str) -> int:
    """Load adapter tensors produced by :func:`lora_state_dict` into ``model``."""
    blob = torch.load(path, map_location="cpu", weights_only=True)
    missing = model.load_state_dict(blob, strict=False)
    unexpected = [k for k in missing.unexpected_keys if "lora_" in k]
    if unexpected:
        raise ValueError(f"adapter {path} does not match the model: {unexpected[:4]}")
    return sum(1 for k in blob if "lora_" in k)


class ClipImageClassifier(nn.Module):
    """Vision tower + frozen category prototypes as the classifier head."""

    def __init__(self, model: nn.Module, prototypes: torch.Tensor, logit_scale: float):
        super().__init__()
        self.model = model
        self.register_buffer("prototypes", F.normalize(prototypes.float(), dim=-1))
        self.scale = float(logit_scale)

    def encode_image(self, pixels: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.model.encode_image(pixels).float(), dim=-1)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.scale * self.encode_image(pixels) @ self.prototypes.t()


class LabeledImages(torch.utils.data.Dataset):
    """Training images with (light) augmentation and their class index."""

    def __init__(self, paths: list[str], labels: list[int], transform):
        self.paths, self.labels, self.transform = paths, labels, transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        from PIL import Image

        with Image.open(self.paths[idx]) as img:
            tensor = self.transform(img.convert("RGB"))
        return tensor, self.labels[idx], idx


@torch.no_grad()
def validation_image_top1(
    classifier: ClipImageClassifier,
    root: str,
    transform,
    device: torch.device,
    args,
) -> float:
    """Image->category top-1 on the validation gallery (the number that matters)."""
    cats = catalog.load_categories(root)
    gallery = catalog.load_images(root, "validation")
    ground_truth = catalog.load_ground_truth(root)
    labels = catalog.validation_image_labels(gallery, ground_truth, cats)
    mask = labels >= 0

    loader = torch.utils.data.DataLoader(
        ds.ImagePathDataset(gallery["path"].tolist(), transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    classifier.eval()
    preds = []
    for pixels, _ in loader:
        pixels = pixels.to(device, non_blocking=True)
        with amp_context(args.precision, device):
            preds.append(classifier(pixels).float().cpu().numpy())
    classifier.train()
    pred = np.concatenate(preds, axis=0).argmax(axis=1)
    return float((pred[mask] == labels[mask]).mean())


def train_lora(args) -> dict:
    """Low-rank fine-tuning of the vision tower with the text side frozen."""
    import open_clip

    set_seed(args.seed)
    root = config.resolve_data_dir(args.data)
    device = features.pick_device(args.device)
    out_dir = Path(args.out) / "lora"
    out_dir.mkdir(parents=True, exist_ok=True)

    cats = catalog.load_categories(root)
    train = catalog.load_images(root, "train")
    labels = imagemap.labels_from_frame(train, cats["category_id"].tolist())

    arch, pretrained = config.backbone_spec(args.backbone)
    model, _, preprocess = open_clip.create_model_and_transforms(
        arch, pretrained=pretrained, device=device
    )
    size = ds.infer_image_size(model, preprocess)
    mean, std = ds.extract_norm(preprocess)
    tokenizer = open_clip.get_tokenizer(arch)

    for param in model.parameters():
        param.requires_grad_(False)
    n_replaced = apply_lora(model.visual, args.lora_rank, args.lora_alpha, args.lora_dropout)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(
        f"[lora] adapted {n_replaced} linear layers, "
        f"{sum(p.numel() for p in trainable) / 1e6:.2f}M trainable parameters"
    )

    proto, _ = prompts.category_prototype_matrix(args.backbone, cats, device=device, precision=args.precision)
    classifier = ClipImageClassifier(model, torch.from_numpy(proto), args.logit_scale).to(device)

    # Frozen text embeddings of the 5 training descriptions per dish, used as
    # in-batch negatives for the contrastive term.
    train_queries = catalog.load_queries(root, "train")
    with torch.no_grad():
        tokens = tokenizer(prompts.query_texts(train_queries)).to(device)
        text_embeds = []
        for start in range(0, tokens.shape[0], 256):
            text_embeds.append(F.normalize(model.encode_text(tokens[start : start + 256]).float(), dim=-1))
        text_embeds = torch.cat(text_embeds, dim=0)
    query_cat = train_queries["category_id"].tolist()
    cat_index = {cid: i for i, cid in enumerate(cats["category_id"])}
    text_pools = {
        cls: torch.tensor(
            [i for i, c in enumerate(query_cat) if cat_index.get(c) == cls],
            device=device,
        )
        for cls in range(len(cats))
    }
    text_pools = {cls: pool for cls, pool in text_pools.items() if pool.numel() > 0}

    train_tf = ds.build_train_transform(size, mean, std)
    eval_tf = ds.build_eval_views(size, mean, std, ("center", "full"))["center"]
    loader = torch.utils.data.DataLoader(
        LabeledImages(train["path"].tolist(), labels.tolist(), train_tf),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(loader) // args.grad_accum) * args.epochs
    sched = make_scheduler(opt, int(args.warmup * total_steps), total_steps)
    scaler = grad_scaler(args.precision, device)

    history, best = [], {"acc": -1.0, "epoch": -1}
    for epoch in range(args.epochs):
        classifier.train()
        running, n_batches = 0.0, 0
        opt.zero_grad(set_to_none=True)
        steps = len(loader)
        for step, (pixels, targets, _) in enumerate(loader, start=1):
            pixels = pixels.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with amp_context(args.precision, device):
                image_embeds = classifier.encode_image(pixels)
                loss = F.cross_entropy(
                    args.logit_scale * image_embeds @ classifier.prototypes.t(),
                    targets,
                    label_smoothing=args.label_smoothing,
                )
                if args.nce_weight > 0:
                    pick = torch.empty(targets.shape[0], dtype=torch.long, device=device)
                    for cls, pool in text_pools.items():
                        sel = targets == cls
                        count = int(sel.sum())
                        if count:
                            pick[sel] = pool[torch.randint(pool.numel(), (count,), device=device)]
                    chosen = text_embeds[pick]
                    scale = args.logit_scale
                    logits_per_image = scale * image_embeds @ chosen.t()
                    logits_per_text = logits_per_image.t()
                    diag = torch.arange(image_embeds.shape[0], device=device)
                    nce = 0.5 * (
                        F.cross_entropy(logits_per_image, diag, label_smoothing=args.label_smoothing)
                        + F.cross_entropy(logits_per_text, diag, label_smoothing=args.label_smoothing)
                    )
                    loss = loss + args.nce_weight * nce

            scaler.scale(loss / args.grad_accum).backward()
            running += float(loss)
            n_batches += 1
            if step % args.grad_accum == 0 or step == steps:
                if args.clip_grad > 0:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(trainable, args.clip_grad)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()

        mean_loss = running / max(n_batches, 1)
        acc = validation_image_top1(classifier, root, eval_tf, device, args)
        history.append({"epoch": epoch, "loss": mean_loss, "val_image_top1": acc})
        print(f"[lora] epoch {epoch:02d} loss {mean_loss:.4f} val_image_top1 {acc:.4f}")
        if acc > best["acc"]:
            best = {"acc": acc, "epoch": epoch}
            torch.save(
                {
                    "state_dict": lora_state_dict(classifier.model),
                    "backbone": args.backbone,
                    "lora_rank": args.lora_rank,
                    "lora_alpha": args.lora_alpha,
                    "epoch": epoch,
                    "val_image_top1": acc,
                },
                out_dir / "best.pt",
            )

    report = {
        "mode": "lora",
        "backbone": args.backbone,
        "best_val_image_top1": best["acc"],
        "best_epoch": best["epoch"],
        "checkpoint": str(out_dir / "best.pt"),
        "history": history,
    }
    (out_dir / "history.json").write_text(json.dumps(report, indent=2))
    print(
        f"[lora] best val image top-1 {best['acc']:.4f} at epoch {best['epoch']} "
        f"-> {out_dir / 'best.pt'}"
    )
    return report


def main() -> None:
    args = parse()
    started = time.time()
    report = train_lora(args) if args.mode == "lora" else train_probe(args)
    report["seconds"] = round(time.time() - started, 1)
    out_path = Path(args.out) / f"{args.mode}_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=float))
    print(f"[{args.mode}] done in {report['seconds']}s -> {out_path}")


if __name__ == "__main__":
    main()