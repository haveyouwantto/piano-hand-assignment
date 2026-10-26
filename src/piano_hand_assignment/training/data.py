"""Training data preparation.

The model is trained on piano scores whose hands are known from the score
itself: in a grand-staff part, ``<staff>1</staff>`` notes belong to the right
hand and ``<staff>2</staff>`` notes to the left hand.

The pipeline is:

    MusicXML (.mxl / .musicxml)  ->  pieces.npz + meta.json  ->  train.py

``pieces.npz`` stores one array per piece named ``p0``, ``p1``, ... with rows
``[onset, pitch, dur, label]``; onsets and durations are in quarter notes and
``label`` is 0 for the right hand and 1 for the left hand.
"""

from __future__ import annotations

import argparse
import json
import random
import zipfile
from pathlib import Path

import numpy as np
from lxml import etree

STEP_SEMI = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
MIN_NOTES = 40


def _local(tag) -> str:
    return tag.split("}")[-1] if isinstance(tag, str) else ""


def read_score_bytes(path: str | Path) -> bytes:
    """Read a plain MusicXML file or the inner XML of a compressed ``.mxl``."""
    path = Path(path)
    if path.suffix.lower() == ".mxl":
        with zipfile.ZipFile(path) as z:
            target = None
            if "META-INF/container.xml" in z.namelist():
                container = etree.fromstring(z.read("META-INF/container.xml"))
                for node in container.iter():
                    if _local(node.tag) == "rootfile":
                        target = node.get("full-path")
                        break
            if target is None:
                candidates = [n for n in z.namelist()
                              if n.lower().endswith((".xml", ".musicxml"))
                              and "META-INF" not in n]
                if not candidates:
                    raise ValueError("no MusicXML inside the archive")
                target = candidates[0]
            return z.read(target)
    return path.read_bytes()


def parse_partwise(root) -> list[tuple[str, int, float, int, float, int, int]]:
    """Return ``(part_id, staff, onset, pitch, dur, velocity, voice)`` tuples."""
    out = []
    for part in root.iter():
        if _local(part.tag) != "part":
            continue
        part_id = part.get("id") or ""
        divisions = 1
        position = 0.0
        chord_onset = None
        for measure in part:
            if _local(measure.tag) != "measure":
                continue
            for element in measure:
                tag = _local(element.tag)
                if tag == "attributes":
                    for sub in element:
                        if _local(sub.tag) == "divisions":
                            try:
                                divisions = max(1, int(sub.text))
                            except (TypeError, ValueError):
                                divisions = divisions
                    continue
                if tag != "note":
                    continue
                is_chord = is_rest = False
                step = octave = None
                alter = 0.0
                dur = 0
                staff = 1
                velocity = 64
                voice = 1
                for sub in element:
                    name = _local(sub.tag)
                    if name == "chord":
                        is_chord = True
                    elif name == "rest":
                        is_rest = True
                    elif name == "duration":
                        try:
                            dur = int(sub.text)
                        except (TypeError, ValueError):
                            dur = 0
                    elif name == "staff":
                        try:
                            staff = int(sub.text)
                        except (TypeError, ValueError):
                            staff = 1
                    elif name == "velocity":
                        try:
                            velocity = int(sub.text)
                        except (TypeError, ValueError):
                            velocity = 64
                    elif name == "voice":
                        try:
                            voice = int(sub.text)
                        except (TypeError, ValueError):
                            voice = 1
                    elif name == "pitch":
                        for p in sub:
                            pname = _local(p.tag)
                            if pname == "step":
                                step = (p.text or "").strip()
                            elif pname == "octave":
                                try:
                                    octave = int(p.text)
                                except (TypeError, ValueError):
                                    octave = None
                            elif pname == "alter":
                                try:
                                    alter = float(p.text)
                                except (TypeError, ValueError):
                                    alter = 0.0
                onset = chord_onset if is_chord else position
                dur_q = dur / divisions
                if not is_chord:
                    chord_onset = position
                    position += dur_q
                if is_rest or step is None or octave is None or step not in STEP_SEMI:
                    continue
                pitch = 12 * (octave + 1) + STEP_SEMI[step] + int(round(alter))
                out.append((part_id, staff, onset, pitch, max(dur_q, 1e-3), velocity, voice))
    return out


def score_to_piece(path: str | Path, min_notes: int = MIN_NOTES):
    """Convert one score into ``(onset, pitch, dur, label)`` arrays or ``None``."""
    root = etree.fromstring(read_score_bytes(path))
    if _local(root.tag) != "score-partwise":
        return None
    entries = parse_partwise(root)
    if not entries:
        return None

    parts = sorted({pid for pid, *_ in entries})
    staves = {staff for _, staff, *_ in entries}
    if len(parts) == 1 and staves <= {1, 2}:
        part_hand = None                       # grand staff: staff number is the hand
    elif len(parts) == 2:
        part_hand = {parts[0]: 0, parts[1]: 1}  # two piano parts: upper / lower
    else:
        return None

    rows = []
    for pid, staff, onset, pitch, dur, velocity, voice in entries:
        label = staff - 1 if part_hand is None else part_hand[pid]
        if label not in (0, 1):
            continue
        rows.append((onset, pitch, dur, label))
    if len(rows) < min_notes:
        return None

    rows.sort(key=lambda r: (r[0], r[1]))
    arr = np.array(rows, dtype=np.float32).T
    labels = arr[3].astype(np.int64)
    if len(np.unique(labels)) < 2:               # a single hand is not a useful example
        return None
    arr[3] = labels
    return arr


def build_dataset(inputs: list[str | Path], out_dir: str | Path,
                  val_frac: float = 0.1, test_frac: float = 0.1,
                  min_notes: int = MIN_NOTES, seed: int = 0) -> dict:
    """Convert a list of scores (files or directories) into ``pieces.npz``."""
    files: list[Path] = []
    for item in inputs:
        item = Path(item)
        if item.is_dir():
            files += sorted(p for p in item.rglob("*")
                            if p.suffix.lower() in (".mxl", ".musicxml", ".xml"))
        else:
            files.append(item)

    pieces, names, skipped = [], [], []
    for path in files:
        try:
            arr = score_to_piece(path, min_notes=min_notes)
        except Exception as exc:
            skipped.append((str(path), f"{type(exc).__name__}: {exc}"))
            continue
        if arr is None:
            skipped.append((str(path), "not a two-hand grand-staff score"))
            continue
        pieces.append(arr)
        names.append(path.name)

    if not pieces:
        raise SystemExit("no usable scores found")

    order = list(range(len(pieces)))
    random.Random(seed).shuffle(order)
    n = len(order)
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))
    splits = {
        "test": order[:n_test],
        "val": order[n_test:n_test + n_val],
        "train": order[n_test + n_val:],
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "pieces.npz",
                        **{f"p{i}": p for i, p in enumerate(pieces)})
    meta = {
        "n_pieces": n,
        "pieces": names,
        "splits": {k: [{"idx": i, "name": names[i]} for i in v]
                   for k, v in splits.items()},
        "skipped": skipped,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    return {"pieces": n, "splits": {k: len(v) for k, v in splits.items()},
            "skipped": len(skipped)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a training set from MusicXML scores")
    ap.add_argument("inputs", nargs="+", help="score files or directories")
    ap.add_argument("-o", "--out", default="data/dataset")
    ap.add_argument("--min-notes", type=int, default=MIN_NOTES)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    args = ap.parse_args()
    stats = build_dataset(args.inputs, args.out, args.val_frac, args.test_frac,
                          args.min_notes)
    print(json.dumps(stats, ensure_ascii=False))
