"""Per-note input features.

Every feature is derived from the note stream alone (onset, pitch, duration),
so the same code path is used at training time and at inference time. All
values are small integers suitable for embedding tables.
"""

from __future__ import annotations

import numpy as np

PITCH_LO, PITCH_HI = 21, 108
N_PITCH = PITCH_HI - PITCH_LO + 1  # 88 keys, A0..C8

NB_REL = 49    # pitch minus local rolling mean, -24..24
NB_IOI = 32    # log-scaled onset gap to the previous note
NB_DUR = 32    # log-scaled note duration in quarter notes
NB_POS = 8     # index inside the chord, ordered by pitch
NB_SIZE = 8    # number of notes in the chord, minus one
NB_DIFF = 49   # pitch step from the previous note, -24..24
NB_STD = 8     # local pitch spread

FEAT_DIMS = [N_PITCH, NB_REL, NB_IOI, NB_DUR, NB_POS, NB_SIZE, NB_DIFF, NB_STD]
N_FEATS = len(FEAT_DIMS)

ONSET_JITTER = 0.02     # quarter notes, training-time augmentation
TEMPO_JITTER = 0.04     # relative tempo scaling
NOTE_DROPOUT = 0.01     # random note removal


def _rolling(x: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Rolling mean and standard deviation with a window of width ``w``."""
    n = len(x)
    cs = np.cumsum(np.insert(x.astype(np.float64), 0, 0.0))
    cs2 = np.cumsum(np.insert(x.astype(np.float64) ** 2, 0, 0.0))
    half = w // 2
    idx = np.arange(n)
    lo = np.clip(idx - half, 0, n)
    hi = np.clip(idx + half + 1, 0, n)
    cnt = np.maximum(hi - lo, 1)
    m = (cs[hi] - cs[lo]) / cnt
    v = np.maximum((cs2[hi] - cs2[lo]) / cnt - m * m, 0.0)
    return m, np.sqrt(v)


def _bucket_log(x: np.ndarray, nb: int, xmax: float) -> np.ndarray:
    v = np.log1p(np.maximum(x, 0.0))
    idx = (v / xmax * (nb - 1)).astype(np.int64)
    return np.clip(idx, 0, nb - 1)


def build_features(onset, pitch, dur, *, rng: np.random.RandomState | None = None,
                   jitter: float = 0.0, tempo: float = 1.0,
                   note_dropout: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Build the feature matrix for one note sequence.

    Args:
        onset: note onsets in quarter notes, non-decreasing.
        pitch: MIDI pitch per note.
        dur: note duration in quarter notes.
        rng: only needed for training-time augmentation.
        jitter: onset jitter in quarter notes, applied per chord.
        tempo: tempo scaling factor applied to onsets.
        note_dropout: probability of dropping a note (simulates missed notes).

    Returns:
        (feats, keep) where ``feats`` is ``int64[N, 8]`` and ``keep`` marks the
        notes that survived augmentation (all ``True`` when not augmenting).
    """
    n = len(pitch)
    if n == 0:
        return np.zeros((0, N_FEATS), dtype=np.int64), np.zeros(0, dtype=bool)

    onset = np.asarray(onset, dtype=np.float64)
    pitch = np.asarray(pitch, dtype=np.int64)
    dur = np.asarray(dur, dtype=np.float64)
    o = onset * tempo

    if jitter > 0 and rng is not None:
        # Jitter whole chords together so intra-chord alignment is preserved
        uniq, inv = np.unique(o, return_inverse=True)
        off = rng.uniform(-jitter, jitter, size=len(uniq))
        u2 = np.maximum.accumulate(uniq + off)   # keep the order stable
        o = u2[inv]

    ioi = np.empty(n)
    ioi[0] = 0.5
    if n > 1:
        ioi[1:] = np.maximum(o[1:] - o[:-1], 0.0)

    uq, inv, cnt = np.unique(o, return_inverse=True, return_counts=True)
    order_in_group = np.zeros(n, dtype=np.int64)
    for g in range(len(uq)):
        idxs = np.where(inv == g)[0]
        idxs = idxs[np.argsort(pitch[idxs])]
        order_in_group[idxs] = np.arange(len(idxs))

    roll_m, roll_s = _rolling(pitch, 33)
    rel = np.clip(pitch - roll_m, -24, 24).astype(np.int64) + 24
    std_b = (np.clip(roll_s, 0, 24) / 24 * (NB_STD - 1)).astype(np.int64)

    diff = np.zeros(n, dtype=np.int64)
    if n > 1:
        diff[1:] = np.clip(pitch[1:] - pitch[:-1], -24, 24)
    diff = diff + 24

    feats = np.stack([
        np.clip(pitch - PITCH_LO, 0, N_PITCH - 1),
        rel,
        _bucket_log(ioi, NB_IOI, 3.0),
        _bucket_log(dur, NB_DUR, 3.0),
        np.clip(order_in_group, 0, NB_POS - 1),
        np.clip(cnt[inv] - 1, 0, NB_SIZE - 1),
        diff,
        std_b,
    ], axis=1).astype(np.int64)

    keep = np.ones(n, dtype=bool)
    if note_dropout > 0 and rng is not None:
        keep = rng.random(n) > note_dropout
        keep[0] = True
    return feats, keep
