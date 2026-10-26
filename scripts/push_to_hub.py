"""Upload the trained checkpoints to the Hugging Face Hub.

    hf auth login
    python scripts/push_to_hub.py \
        --lstm checkpoints/lstm.pt \
        --transformer checkpoints/transformer.pt \
        --repo-id hywt232/piano-hand-assignment
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo

MODEL_CARD = """---
license: mit
library_name: pytorch
tags: [midi, piano, hand-separation, music-information-retrieval]
---

# piano-hand-assignment

Weights for [piano-hand-assignment](https://github.com/haveyouwantto/piano-hand-assignment):
assign every note of a piano MIDI file to the right or the left hand.

## Files

| File | Architecture | Parameters |
| --- | --- | --- |
| `lstm.pt` | 2-layer bidirectional LSTM, d=192 | 1.58M |
| `transformer.pt` | 4-layer Transformer encoder, d=192, 4 heads | 1.75M |

Both are PyTorch checkpoints saved as `{"arch": ..., "state": state_dict, "args": ...}`.
Inference averages the two probability distributions.

## Usage

```python
from piano_hand_assignment import HandAssigner

assigner = HandAssigner()                 # downloads these files on first run
assigner.assign_file("input.mid", "output.mid")
```

## Training data

30,382 two-hand piano scores from PDMX (MuseScore-derived MusicXML), 20.4M notes.
Hand labels come from the staff number in the score.

## Evaluation

| Test set | Accuracy | Hand-switch F1 |
| --- | --- | --- |
| Held-out PDMX split (2.0M notes) | 94.30 | 0.918 |
| Yamaha Disklavier pieces (unseen domain) | 95.91 | 0.927 |
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lstm", default="checkpoints/lstm.pt")
    ap.add_argument("--transformer", default="checkpoints/transformer.pt")
    ap.add_argument("--repo-id", default="hywt232/piano-hand-assignment")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    files = {Path(args.lstm): "lstm.pt", Path(args.transformer): "transformer.pt"}
    for path in files:
        if not path.exists():
            raise SystemExit(f"missing checkpoint: {path}")

    api = HfApi()
    create_repo(args.repo_id, repo_type="model", exist_ok=True, private=args.private)
    for path, name in files.items():
        api.upload_file(path_or_fileobj=str(path), path_in_repo=name,
                        repo_id=args.repo_id, repo_type="model")
        print(f"uploaded {path} -> {args.repo_id}/{name}")

    card = Path("README_hf.md")
    card.write_text(MODEL_CARD, encoding="utf-8")
    api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md",
                    repo_id=args.repo_id, repo_type="model")
    print(f"model card -> https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
