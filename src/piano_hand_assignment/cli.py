"""Command line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .predict import HandAssigner, iter_midi_files


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="pha-assign",
        description="Assign piano notes to the right or left hand and write "
                    "a MIDI file with two note tracks.")
    ap.add_argument("input", help="MIDI file or a directory of MIDI files")
    ap.add_argument("-o", "--output", default="out",
                    help="output file (single input) or directory (default: out)")
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    ap.add_argument("--checkpoint-dir", default=None,
                    help="use local checkpoints instead of downloading from the Hub")
    ap.add_argument("--repo-id", default=None, help="override the Hugging Face repo id")
    ap.add_argument("--archs", default="lstm,transformer",
                    help="comma separated ensemble members")
    ap.add_argument("--window", type=int, default=1024, help="notes per forward window")
    ap.add_argument("--merge-window", type=float, default=0.05,
                    help="chord merging window in quarter notes")
    ap.add_argument("--no-split-channels", dest="split_channels", action="store_false",
                    help="keep both hands on their original MIDI channels")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    assigner = HandAssigner(device=args.device,
                            repo_id=args.repo_id,
                            checkpoint_dir=args.checkpoint_dir,
                            archs=[a.strip() for a in args.archs.split(",") if a.strip()],
                            window=args.window,
                            merge_window=args.merge_window,
                            split_channels=args.split_channels)

    files = list(iter_midi_files(args.input))
    if not files:
        print(f"no MIDI files found in {args.input}", file=sys.stderr)
        return 2

    single = Path(args.input).is_file()
    out_root = Path(args.output)
    for src in files:
        dst = out_root if single and out_root.suffix else out_root / f"{src.stem}_hands.mid"
        stats = assigner.assign_file(src, dst)
        channels = f"ch {stats['right_channels']} / {stats['left_channels']}"
        print(f"{src.name:40s} {stats['notes']:6d} notes  "
              f"LH {stats['left_ratio'] * 100:5.1f}%  "
              f"mean conf {stats['mean_confidence']:.3f}  {channels}  -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
