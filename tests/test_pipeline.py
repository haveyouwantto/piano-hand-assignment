"""End-to-end smoke test with randomly initialised checkpoints (no download)."""

from __future__ import annotations

from pathlib import Path

import mido
import torch

from piano_hand_assignment import model as M
from piano_hand_assignment.midi_io import collect_notes, load_midi, snapshot
from piano_hand_assignment.predict import HandAssigner


def write_tiny_checkpoints(tmp: Path) -> Path:
    for arch in ("lstm", "transformer"):
        net = M.build(arch)
        torch.save({"arch": arch, "state": net.state_dict()}, tmp / f"{arch}.pt")
    return tmp


def make_midi(path: Path) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    tick = 0
    for i in range(64):
        for pitch in (48 + (i % 5), 72 + (i % 7)):
            track.append(mido.Message("note_on", note=pitch, velocity=80, channel=0, time=0))
        track.append(mido.Message("note_off", note=48 + (i % 5), velocity=0, channel=0, time=240))
        track.append(mido.Message("note_off", note=72 + (i % 7), velocity=0, channel=0, time=0))
        tick += 240
    midi.tracks.append(track)
    midi.save(str(path))


def test_assign_file(tmp_path: Path):
    ckpt_dir = write_tiny_checkpoints(tmp_path)
    src = tmp_path / "in.mid"
    make_midi(src)
    assigner = HandAssigner(device="cpu", checkpoint_dir=ckpt_dir, window=32, batch_size=4)
    dst = tmp_path / "out.mid"
    stats = assigner.assign_file(src, dst)

    assert stats["notes"] == len(collect_notes(load_midi(src)))
    before = snapshot(load_midi(src))
    after = snapshot(load_midi(dst))
    original_tracks = len(load_midi(src).tracks)
    assert before["notes"] == after["notes"]
    assert before["non_note"] == [e for e in after["non_note"] if e[0] < original_tracks]


def test_assign_bytes(tmp_path: Path):
    ckpt_dir = write_tiny_checkpoints(tmp_path)
    src = tmp_path / "in.mid"
    make_midi(src)
    assigner = HandAssigner(device="cpu", checkpoint_dir=ckpt_dir, window=64)
    data = assigner.assign_bytes(src.read_bytes())
    assert data[:4] == b"MThd"
    reloaded = load_midi(data)
    assert len(reloaded.tracks) == len(load_midi(src).tracks) + 2
