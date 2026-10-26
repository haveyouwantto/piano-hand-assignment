"""The only thing the tool is allowed to change is which track a note lives in."""

from __future__ import annotations

import mido

from piano_hand_assignment.midi_io import (collect_notes, load_midi, save_bytes,
                                           snapshot, write_with_hands)


def build_sample() -> mido.MidiFile:
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("track_name", name="conductor", time=0))
    meta.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    meta.append(mido.MetaMessage("time_signature", numerator=3, denominator=4, time=0))
    meta.append(mido.MetaMessage("key_signature", key="Eb", time=0))
    midi.tracks.append(meta)

    piano = mido.MidiTrack()
    piano.append(mido.MetaMessage("track_name", name="Piano", time=0))
    piano.append(mido.Message("program_change", program=0, channel=0, time=0))
    piano.append(mido.Message("control_change", control=64, value=127, channel=0, time=0))
    piano.append(mido.Message("note_on", note=60, velocity=90, channel=0, time=0))
    piano.append(mido.Message("note_on", note=48, velocity=80, channel=0, time=0))
    piano.append(mido.Message("note_off", note=60, velocity=0, channel=0, time=480))
    piano.append(mido.Message("pitchwheel", pitch=1200, channel=0, time=0))
    piano.append(mido.Message("note_off", note=48, velocity=0, channel=0, time=240))
    piano.append(mido.Message("note_on", note=72, velocity=100, channel=0, time=0))
    piano.append(mido.Message("note_off", note=72, velocity=0, channel=0, time=480))
    midi.tracks.append(piano)

    drums = mido.MidiTrack()
    drums.append(mido.MetaMessage("track_name", name="Drums", time=0))
    drums.append(mido.Message("note_on", note=36, velocity=100, channel=9, time=0))
    drums.append(mido.Message("note_off", note=36, velocity=0, channel=9, time=240))
    midi.tracks.append(drums)
    return midi


def test_notes_are_moved_but_never_altered():
    midi = build_sample()
    notes = collect_notes(midi)
    assert len(notes) == 3

    out = write_with_hands(midi, notes, [0, 1, 0])
    before, after = snapshot(midi), snapshot(out)
    original_tracks = len(midi.tracks)
    kept = [e for e in after["non_note"] if e[0] < original_tracks]

    assert before["notes"] == after["notes"]       # pitch, velocity, timing, channel
    assert before["non_note"] == kept              # every other event, untouched
    assert out.ticks_per_beat == midi.ticks_per_beat
    assert len(out.tracks) == original_tracks + 2  # only the two hand tracks are new


def test_drum_notes_are_not_touched():
    midi = build_sample()
    notes = collect_notes(midi)
    out = write_with_hands(midi, notes, [0, 1, 0])

    def drum_events(track_file):
        return [(t, m.type, m.note) for _, t, m in _events(track_file)
                if not m.is_meta and getattr(m, "channel", None) == 9]

    assert drum_events(out) == drum_events(midi) == [(0, "note_on", 36), (240, "note_off", 36)]


def _events(midi):
    from piano_hand_assignment.midi_io import iter_abs_events
    return list(iter_abs_events(midi))


def test_roundtrip_through_bytes():
    midi = build_sample()
    notes = collect_notes(midi)
    out = write_with_hands(midi, notes, [1, 1, 1])
    reloaded = load_midi(save_bytes(out))
    assert snapshot(reloaded)["notes"] == snapshot(midi)["notes"]
    assert len(reloaded.tracks) == len(midi.tracks) + 2
