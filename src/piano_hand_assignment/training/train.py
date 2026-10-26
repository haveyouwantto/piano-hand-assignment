"""Training script for the hand-assignment taggers.

Example:
    python -m piano_hand_assignment.training.train \
        --data data/dataset --arch lstm --epochs 40 --batch-size 128 --lr 8e-4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .. import model as M
from ..features import (NOTE_DROPOUT, ONSET_JITTER, TEMPO_JITTER, build_features)
from ..metrics import piece_metrics, summarize


def load_pieces(data_dir: str | Path) -> tuple[list[dict], dict]:
    data_dir = Path(data_dir)
    blob = np.load(data_dir / "pieces.npz")
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    keys = sorted(blob.files, key=lambda k: int(k[1:]))
    pieces = []
    for key in keys:
        arr = blob[key]
        pieces.append({
            "onset": arr[0].astype(np.float64),
            "pitch": arr[1].astype(np.int64),
            "dur": arr[2].astype(np.float64),
            "label": arr[3].astype(np.int64),
        })
    return pieces, meta


class WindowDataset(Dataset):
    """Fixed-length windows over each piece, with light augmentation on train."""

    def __init__(self, pieces: list[dict], win: int = 384, stride: int = 192,
                 augment: bool = False, seed: int = 0):
        self.pieces = pieces
        self.win, self.stride = win, stride
        self.augment = augment
        self.index: list[tuple[int, int, int]] = []
        for pi, piece in enumerate(pieces):
            n = len(piece["pitch"])
            if n <= win:
                self.index.append((pi, 0, n))
            else:
                start = 0
                while start < n:
                    end = min(start + win, n)
                    self.index.append((pi, start, end))
                    start += stride
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        pi, start, end = self.index[i]
        piece = self.pieces[pi]
        rng = self.rng if self.augment else None
        tempo = 1.0
        if self.augment:
            tempo = 1.0 + self.rng.uniform(-TEMPO_JITTER, TEMPO_JITTER)
        feats, keep = build_features(
            piece["onset"][start:end], piece["pitch"][start:end], piece["dur"][start:end],
            rng=rng, jitter=ONSET_JITTER if self.augment else 0.0, tempo=tempo,
            note_dropout=NOTE_DROPOUT if self.augment else 0.0)
        labels = piece["label"][start:end][keep]
        return torch.from_numpy(feats[keep]), torch.from_numpy(labels), pi, start


def collate(batch):
    feats = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    pids = [b[2] for b in batch]
    starts = [b[3] for b in batch]
    length = max(f.size(0) for f in feats)
    x = torch.zeros(len(feats), length, feats[0].size(1), dtype=torch.long)
    y = torch.full((len(feats), length), -100, dtype=torch.long)
    mask = torch.zeros(len(feats), length, dtype=torch.bool)
    for i, (feat, lab) in enumerate(zip(feats, labels)):
        n = feat.size(0)
        x[i, :n] = feat
        y[i, :n] = lab
        mask[i, :n] = True
    return x, y, mask, pids, starts


def split_ids(meta: dict, split: str) -> list[int]:
    return [entry["idx"] for entry in meta["splits"][split]]


def evaluate(net: nn.Module, loader: DataLoader, pieces: list[dict],
             ids: list[int], device: str) -> dict:
    net.eval()
    buckets: dict[int, list] = {}
    with torch.no_grad():
        for x, y, mask, pids, starts in loader:
            x, mask = x.to(device), mask.to(device)
            logits = net(x, mask).float().cpu().numpy()
            labels = y.numpy()
            for i, pid in enumerate(pids):
                n = int(mask[i].sum())
                buckets.setdefault(int(pid), []).append((int(starts[i]), logits[i, :n],
                                                         labels[i, :n]))
    per_piece = []
    for local, original in enumerate(ids):
        if local not in buckets:
            continue
        chunks = sorted(buckets[local], key=lambda c: c[0])
        logits = np.concatenate([c[1] for c in chunks])
        labels = np.concatenate([c[2] for c in chunks])
        valid = labels >= 0
        pred, labels = logits[valid].argmax(1), labels[valid]
        pitch = pieces[original]["pitch"][:len(labels)]
        per_piece.append(piece_metrics(pred, labels, pitch))
    return summarize(per_piece)


def build_loaders(pieces, meta, args):
    train_pieces = [pieces[i] for i in split_ids(meta, "train")]
    val_pieces = [pieces[i] for i in split_ids(meta, "val")]
    test_pieces = [pieces[i] for i in split_ids(meta, "test")]
    train_ds = WindowDataset(train_pieces, args.win, args.stride, augment=True, seed=1)
    val_ds = WindowDataset(val_pieces, args.win, args.win)
    test_ds = WindowDataset(test_pieces, args.win, args.win)
    return (DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                       collate_fn=collate, drop_last=True),
            DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate),
            DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate),
            split_ids(meta, "val"), split_ids(meta, "test"))


def train_one(arch: str, pieces, meta, args, device: str) -> dict:
    train_loader, val_loader, test_loader, val_ids, test_ids = build_loaders(pieces, meta, args)
    torch.manual_seed(args.seed)
    net = M.build(arch).to(device)
    if args.init:
        ckpt = torch.load(args.init, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["state"])
        print(f"  initialised from {args.init}")

    optimiser = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader)) * args.epochs
    warmup = max(1, int(total_steps * 0.05))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimiser,
        lambda s: s / warmup if s < warmup
        else 0.5 * (1 + np.cos(np.pi * (s - warmup) / max(1, total_steps - warmup))))
    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.02)

    best_acc, best_state, best_epoch = -1.0, None, -1
    history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        net.train()
        running, seen = 0.0, 0
        for x, y, mask, _, _ in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            logits = net(x, mask)
            loss = criterion(logits.reshape(-1, 2), y.reshape(-1))
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimiser.step()
            schedule.step()
            running += float(loss.item()) * x.size(0)
            seen += x.size(0)
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val = evaluate(net, val_loader, pieces, val_ids, device)
            history.append({"epoch": epoch, "loss": running / max(seen, 1),
                            "val_acc": val["acc"], "val_switch_f1": val["sw_f1"]})
            mark = ""
            if val["acc"] > best_acc:
                best_acc = val["acc"]
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                best_epoch = epoch
                mark = " *"
            print(f"  [{arch}] epoch {epoch:3d}  loss {running / max(seen, 1):.4f}  "
                  f"val_acc {val['acc']:.4f}  val_switch_f1 {val['sw_f1']:.4f}"
                  f"  ({time.time() - started:.0f}s){mark}")

    if best_state is not None:
        net.load_state_dict(best_state)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{arch}.pt"
    torch.save({"arch": arch, "state": net.state_dict(), "args": vars(args)}, ckpt_path)

    result = {
        "arch": arch,
        "params": M.n_params(net),
        "best_epoch": best_epoch,
        "val": evaluate(net, val_loader, pieces, val_ids, device),
        "test": evaluate(net, test_loader, pieces, test_ids, device),
        "checkpoint": str(ckpt_path),
        "history": history,
        "seconds": round(time.time() - started, 1),
    }
    print(f"  [{arch}] test_acc {result['test']['acc']:.4f}  "
          f"switch_f1 {result['test']['sw_f1']:.4f}  checkpoint {ckpt_path}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Train hand-assignment taggers")
    ap.add_argument("--data", default="data/dataset", help="directory with pieces.npz + meta.json")
    ap.add_argument("--arch", default="both", choices=["lstm", "transformer", "both"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--win", type=int, default=384)
    ap.add_argument("--stride", type=int, default=192)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--init", default="", help="optional checkpoint to start from")
    ap.add_argument("--output-dir", default="checkpoints")
    ap.add_argument("--out", default="runs/result.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pieces, meta = load_pieces(args.data)
    print(f"device {device} | pieces {len(pieces)} | "
          f"train {len(meta['splits']['train'])} val {len(meta['splits']['val'])} "
          f"test {len(meta['splits']['test'])}")

    archs = ["lstm", "transformer"] if args.arch == "both" else [args.arch]
    results = [train_one(arch, pieces, meta, args, device) for arch in archs]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=1, default=float),
                        encoding="utf-8")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
