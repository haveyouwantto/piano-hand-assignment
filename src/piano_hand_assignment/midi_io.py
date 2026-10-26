"""MIDI input/output.

Reading: notes are collected from every non-drum track together with their exact
tick positions, channels and velocities.

Writing: note events are moved into two new tracks (Right Hand / Left Hand). The
left hand can also be moved to an unused MIDI channel, together with the channel
state it needs, so that software which only distinguishes channels shows the two
hands separately. Every other event — tempo, time signature, key signature,
controllers, pitch bend, program changes, markers, lyrics, other instrument
tracks — is copied unchanged to the same tick in the same track.
"""

from __future__ import annotations

import io
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Iterator

import mido

DRUM_CHANNEL = 9
DEFAULT_MERGE_WINDOW = 0.05   # quarter notes; groups notes struck nearly together


@dataclass
class Note:
    start: int          # ticks
    end: int            # ticks
    pitch: int
    velocity: int
    channel: int
    track: int


def _message_payload(msg) -> tuple:
    """Message fields excluding the delta time; the absolute tick is compared instead."""
    return tuple(sorted((k, v) for k, v in msg.dict().items() if k != "time"))


def iter_abs_events(midi: mido.MidiFile) -> Iterator[tuple[int, int, object]]:
    """Yield ``(track_index, abs_tick, message)`` for every message."""
    for ti, track in enumerate(midi.tracks):
        tick = 0
        for msg in track:
            tick += msg.time
            yield ti, tick, msg


def load_midi(source: str | bytes) -> mido.MidiFile:
    if isinstance(source, (bytes, bytearray)):
        return mido.MidiFile(file=io.BytesIO(source))
    return mido.MidiFile(source)


def collect_notes(midi: mido.MidiFile) -> list[Note]:
    """Pair note-on/off messages into notes, skipping drum channels."""
    pending: dict[tuple[int, int, int], list[tuple[int, int]]] = {}
    notes: list[Note] = []
    for ti, tick, msg in iter_abs_events(midi):
        if msg.is_meta or msg.type not in ("note_on", "note_off"):
            continue
        if msg.channel == DRUM_CHANNEL:
            continue
        key = (ti, msg.channel, msg.note)
        if msg.type == "note_on" and msg.velocity > 0:
            pending.setdefault(key, []).append((tick, msg.velocity))
            continue
        stack = pending.get(key)
        if not stack:
            continue                      # unmatched note-off, leave it alone
        start, velocity = stack.pop(0)     # FIFO keeps repeated pitches ordered
        if tick <= start:
            continue
        notes.append(Note(start=start, end=tick, pitch=msg.note, velocity=velocity,
                          channel=msg.channel, track=ti))
    notes.sort(key=lambda n: (n.start, n.pitch, n.channel))
    return notes


def _consumed_keys(notes: list[Note]) -> Counter:
    """Multiset of the exact note events that were turned into :class:`Note`s.

    Only these events are removed from the original tracks; orphan note-on or
    note-off messages in malformed files stay where they were.
    """
    keys: Counter = Counter()
    for n in notes:
        keys[(n.track, n.channel, n.pitch, n.start, "on")] += 1
        keys[(n.track, n.channel, n.pitch, n.end, "off")] += 1
    return keys


def merge_chord_onsets(onset_quarters: list[float] | "list[float]",
                       window: float = DEFAULT_MERGE_WINDOW) -> list[float]:
    """Snap onsets that fall inside the same short window to a common value.

    Real performances spread chords over tens of milliseconds; the model expects
    simultaneous notes to share an onset so that chord features are meaningful.
    """
    if not onset_quarters:
        return []
    order = sorted(range(len(onset_quarters)), key=lambda i: onset_quarters[i])
    result = [0.0] * len(onset_quarters)
    cluster_start = onset_quarters[order[0]]
    prev = cluster_start
    for idx in order:
        t = onset_quarters[idx]
        if t - prev > window:
            cluster_start = t
        result[idx] = cluster_start
        prev = t
    return result


CHANNEL_STATE_TYPES = ("program_change", "control_change", "pitchwheel",
                       "channel_pressure", "polytouch")


def used_channels(midi: mido.MidiFile) -> set[int]:
    """Channels that carry any channel message anywhere in the file."""
    channels = set()
    for _, _, msg in iter_abs_events(midi):
        if not msg.is_meta and getattr(msg, "channel", None) is not None:
            channels.add(msg.channel)
    return channels


def free_channel(midi: mido.MidiFile) -> int | None:
    """First channel that the file does not use, ignoring the drum channel."""
    occupied = used_channels(midi)
    for channel in range(16):
        if channel != DRUM_CHANNEL and channel not in occupied:
            return channel
    return None


def write_with_hands(midi: mido.MidiFile, notes: list[Note], labels: Iterable[int],
                     right_name: str = "Right Hand",
                     left_name: str = "Left Hand",
                     split_channels: bool = True) -> mido.MidiFile:
    """Write a MIDI file whose notes are split onto two tracks.

    When ``split_channels`` is true the left-hand notes are additionally moved to
    a channel the file does not use yet, and the channel state they need
    (program, bank, volume, pan, sustain, pitch bend, pressure) is copied from
    their source channel to that channel at the same ticks. Software that can
    only distinguish channels then shows the two hands separately.
    """
    labels = list(labels)
    if len(labels) != len(notes):
        raise ValueError("labels and notes must have the same length")

    target_channel = free_channel(midi) if split_channels else None
    if split_channels and target_channel is None:
        warnings.warn("every MIDI channel is in use; the two hands keep their "
                      "original channels and only the tracks differ",
                      RuntimeWarning, stacklevel=2)

    consumed = _consumed_keys(notes)
    out = mido.MidiFile(ticks_per_beat=midi.ticks_per_beat,
                        type=1 if midi.type == 0 else midi.type)

    # 1) copy every original track, dropping the note events that were paired up
    for ti, track in enumerate(midi.tracks):
        new_track = mido.MidiTrack()
        tick = 0
        for msg in track:
            tick += msg.time
            if not msg.is_meta and msg.type in ("note_on", "note_off"):
                kind = "on" if (msg.type == "note_on" and msg.velocity > 0) else "off"
                key = (ti, msg.channel, msg.note, tick, kind)
                if consumed.get(key, 0) > 0:
                    consumed[key] -= 1
                    continue
            if msg.is_meta and msg.type == "end_of_track":
                continue
            new_track.append(msg.copy(time=tick))
        out.tracks.append(_finalize(new_track))

    # channel messages the left hand needs on its own channel
    hand_state: list[tuple[int, int, mido.Message]] = []
    if target_channel is not None:
        sources = {note.channel for note, label in zip(notes, labels) if label == 1}
        for _, tick, msg in iter_abs_events(midi):
            if msg.is_meta or msg.type not in CHANNEL_STATE_TYPES:
                continue
            if msg.channel in sources:
                hand_state.append((tick, 0, msg.copy(channel=target_channel)))

    # 2) append the two hand tracks
    for name, wanted in ((right_name, 0), (left_name, 1)):
        track = mido.MidiTrack()
        track.append(mido.MetaMessage("track_name", name=name, time=0))
        events: list[tuple[int, int, mido.Message]] = []
        if wanted == 1:
            events += hand_state
        for note, label in zip(notes, labels):
            if label != wanted:
                continue
            channel = note.channel
            if wanted == 1 and target_channel is not None:
                channel = target_channel
            events.append((note.start, 1, mido.Message(
                "note_on", note=note.pitch, velocity=note.velocity,
                channel=channel, time=0)))
            events.append((note.end, 2, mido.Message(
                "note_off", note=note.pitch, velocity=0,
                channel=channel, time=0)))
        events.sort(key=lambda e: (e[0], e[1]))
        for tick, _, msg in events:
            track.append(msg.copy(time=tick))
        out.tracks.append(_finalize(track))
    return out


def track_channels(midi: mido.MidiFile, name: str) -> list[int]:
    """Channels used by the notes of the named track."""
    for track in midi.tracks:
        if (track.name or "") != name:
            continue
        return sorted({msg.channel for msg in track
                       if msg.type == "note_on" and msg.velocity > 0 and not msg.is_meta})
    return []


def _finalize(track: mido.MidiTrack) -> mido.MidiTrack:
    """Convert absolute tick times held in ``.time`` into delta times."""
    fixed = mido.MidiTrack()
    prev = 0
    for msg in track:
        tick = msg.time
        fixed.append(msg.copy(time=tick - prev))
        prev = tick
    fixed.append(mido.MetaMessage("end_of_track", time=0))
    return fixed


def snapshot(midi: mido.MidiFile) -> dict:
    """Compact, comparable summary of a MIDI file used by the round-trip test."""
    non_note = []
    for ti, tick, msg in iter_abs_events(midi):
        if msg.is_meta:
            if msg.type == "end_of_track":
                continue
            non_note.append((ti, tick, msg.type, _message_payload(msg)))
        elif msg.type not in ("note_on", "note_off"):
            non_note.append((ti, tick, msg.type, msg.channel, _message_payload(msg)))
    notes = sorted((n.start, n.end, n.pitch, n.velocity, n.channel)
                   for n in collect_notes(midi))
    return {"tracks": len(midi.tracks), "ticks_per_beat": midi.ticks_per_beat,
            "non_note": non_note, "notes": notes}


def save_bytes(midi: mido.MidiFile) -> bytes:
    buf = io.BytesIO()
    midi.save(file=buf)
    return buf.getvalue()
