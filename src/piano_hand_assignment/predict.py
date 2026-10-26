"""Hand assignment inference.

The public entry points are :class:`HandAssigner` (stateful, reusable) and the
module-level :func:`assign_hands` helper, which keeps a single cached assigner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from . import model as M
from .features import build_features
from .hub import ensure_checkpoints
from .midi_io import (Note, collect_notes, load_midi, merge_chord_onsets,
                      save_bytes, write_with_hands)

DEFAULT_ARCHS = ("lstm", "transformer")
DEFAULT_WINDOW = 1024
DEFAULT_MERGE_WINDOW = 0.05     # quarter notes


def _resolve_device(device: str = "auto") -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class HandAssigner:
    """Averaged ensemble of the released checkpoints."""

    def __init__(self, device: str = "auto", repo_id: str | None = None,
                 checkpoint_dir: str | Path | None = None,
                 archs: Sequence[str] = DEFAULT_ARCHS,
                 window: int = DEFAULT_WINDOW,
                 merge_window: float = DEFAULT_MERGE_WINDOW,
                 batch_size: int = 8):
        self.device = _resolve_device(device)
        self.window = window
        self.merge_window = merge_window
        self.batch_size = batch_size
        paths = ensure_checkpoints(repo_id=repo_id, checkpoint_dir=checkpoint_dir,
                                   archs=tuple(archs))
        self.archs = tuple(archs)
        self.nets = []
        for arch in self.archs:
            ckpt = torch.load(paths[arch], map_location=self.device, weights_only=False)
            net = M.build(ckpt.get("arch", arch)).to(self.device)
            net.load_state_dict(ckpt["state"])
            net.eval()
            self.nets.append(net)

    # ------------------------------------------------------------------ notes
    def note_onsets(self, notes: Sequence[Note], ticks_per_beat: int) -> list[float]:
        raw = [n.start / float(ticks_per_beat) for n in notes]
        return merge_chord_onsets(raw, window=self.merge_window)

    def predict_probs(self, feats: np.ndarray) -> np.ndarray:
        """Windowed forward pass; returns averaged probabilities ``[N, 2]``."""
        n = len(feats)
        probs = np.zeros((n, 2), dtype=np.float64)
        if n == 0:
            return probs
        spans = [(s, min(s + self.window, n)) for s in range(0, n, self.window)]
        with torch.no_grad():
            for i in range(0, len(spans), self.batch_size):
                chunk = spans[i:i + self.batch_size]
                length = max(e - s for s, e in chunk)
                x = torch.zeros(len(chunk), length, feats.shape[1], dtype=torch.long)
                mask = torch.zeros(len(chunk), length, dtype=torch.bool)
                for j, (s, e) in enumerate(chunk):
                    x[j, :e - s] = torch.from_numpy(feats[s:e])
                    mask[j, :e - s] = True
                x, mask = x.to(self.device), mask.to(self.device)
                batch_probs = None
                for net in self.nets:
                    logits = net(x, mask).float()
                    p = torch.softmax(logits, dim=-1)
                    batch_probs = p if batch_probs is None else batch_probs + p
                batch_probs = (batch_probs / len(self.nets)).cpu().numpy()
                for j, (s, e) in enumerate(chunk):
                    probs[s:e] = batch_probs[j, :e - s]
        return probs

    def assign_notes(self, notes: Sequence[Note], ticks_per_beat: int) -> tuple[np.ndarray, np.ndarray]:
        if not notes:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
        onset = self.note_onsets(notes, ticks_per_beat)
        pitch = [n.pitch for n in notes]
        dur = [max(n.end - n.start, 1) / float(ticks_per_beat) for n in notes]
        feats, _ = build_features(onset, pitch, dur)
        probs = self.predict_probs(feats)
        return probs.argmax(1).astype(np.int64), probs.max(1)

    # ------------------------------------------------------------------- files
    def assign_bytes(self, data: bytes) -> bytes:
        midi = load_midi(bytes(data))
        notes = collect_notes(midi)
        labels, _ = self.assign_notes(notes, midi.ticks_per_beat)
        return save_bytes(write_with_hands(midi, notes, labels))

    def assign_file(self, src: str | Path, dst: str | Path) -> dict:
        midi = load_midi(str(src))
        notes = collect_notes(midi)
        labels, conf = self.assign_notes(notes, midi.ticks_per_beat)
        out = write_with_hands(midi, notes, labels)
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        out.save(str(dst))
        left = int((labels == 1).sum())
        return {
            "notes": len(notes),
            "left_hand": left,
            "right_hand": len(notes) - left,
            "left_ratio": float(left / len(notes)) if notes else 0.0,
            "mean_confidence": float(conf.mean()) if len(conf) else 0.0,
            "low_confidence_ratio": float((conf < 0.7).mean()) if len(conf) else 0.0,
        }


_DEFAULT_ASSIGNER: HandAssigner | None = None


def get_assigner(**kwargs) -> HandAssigner:
    global _DEFAULT_ASSIGNER
    if _DEFAULT_ASSIGNER is None:
        _DEFAULT_ASSIGNER = HandAssigner(**kwargs)
    return _DEFAULT_ASSIGNER


def assign_hands(source: str | Path | bytes, destination: str | Path | None = None):
    """Assign hands to every piano note in a MIDI file.

    Args:
        source: path to a ``.mid``/``.midi`` file, or raw MIDI bytes.
        destination: output path; when omitted, bytes are returned.

    Returns:
        The output path, or the encoded MIDI bytes.
    """
    assigner = get_assigner()
    if isinstance(source, (bytes, bytearray)):
        data = assigner.assign_bytes(bytes(source))
        if destination is None:
            return data
        Path(destination).write_bytes(data)
        return destination
    if destination is None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.mid"
            assigner.assign_file(source, out)
            return out.read_bytes()
    assigner.assign_file(source, destination)
    return destination


def iter_midi_files(root: str | Path) -> Iterable[Path]:
    """Yield MIDI files once each, even on case-insensitive file systems."""
    root = Path(root)
    if root.is_file():
        yield root
        return
    seen: set[Path] = set()
    for pattern in ("*.mid", "*.midi", "*.MID", "*.MIDI"):
        for path in sorted(root.glob(pattern)):
            key = path.resolve()
            if key in seen:
                continue
            seen.add(key)
            yield path
