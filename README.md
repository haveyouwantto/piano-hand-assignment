# piano-hand-assignment

**Split a piano MIDI file into right-hand and left-hand tracks.** One MIDI in, one MIDI out.

The tool is for piano MIDI that has no hand information — transcriptions, exports from DAWs, merged single-track files. It reads every note, decides which hand plays it, and writes the same file back with two note tracks called `Right Hand` and `Left Hand`. Notes, velocities, timing, channels, tempo map, controllers, pitch bends, markers and all other tracks are left exactly as they were.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Model on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-model-orange)](https://huggingface.co/anime-song/piano-hand-assignment)

---

## Quick start

```bash
git clone https://github.com/haveyouwantto/piano-hand-assignment.git
cd piano-hand-assignment
uv sync          # or: pip install -e .
uv run pha-assign input.mid -o output.mid
```

Weights are downloaded from Hugging Face on first run and cached in `~/.cache/piano_hand_assignment`.

Directories work too:

```bash
pha-assign path/to/midis/ -o out/
```

## Python

```python
from piano_hand_assignment import HandAssigner

assigner = HandAssigner()                       # loads the ensemble once
assigner.assign_file("input.mid", "output.mid") # path in, path out
data = assigner.assign_bytes(open("input.mid", "rb").read())   # bytes in, bytes out
```

`assign_file` also returns statistics for the piece: note count, left-hand ratio, mean confidence and the share of low-confidence notes.

## HTTP API

```bash
uv run pha-serve        # uvicorn on 0.0.0.0:8000
```

```bash
curl -X POST http://localhost:8000/assign \
     -F "file=@input.mid" \
     -o output.mid
```

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | status, loaded architectures, device |
| `POST` | `/assign` | multipart field `file` → MIDI file with hand tracks |

`POST /assign` answers with `audio/midi` and a `Content-Disposition` header carrying the original file name.

---

## What the output looks like

Given a file with one piano part, the output keeps the original tracks (minus their note events) and appends:

```
[tracks from the input]        ← tempo, time signatures, controllers, other instruments, unchanged
[track] Right Hand             ← notes the right hand plays
[track] Left Hand              ← notes the left hand plays
```

Details of the contract:

- Every note keeps its pitch, velocity, start tick, end tick and MIDI channel.
- Every non-note event keeps its type, tick and payload; drum tracks are not touched at all.
- The two hand tracks are appended, so track indices of the original tracks stay valid.
- A type-0 file (single track) is written as type 1, since two note tracks are required.
- Notes are not re-quantised, merged, shortened or re-voiced. The only thing that changes is which track a note lives on.

The test suite checks this contract note by note and event by event (`tests/test_midi_preservation.py`).

## How it works

Each note is described by eight discrete features, all computed from the note stream alone: MIDI pitch, pitch relative to the local rolling mean, onset gap to the previous note, duration, position inside its chord, chord size, pitch step from the previous note, and local pitch spread. Onsets within a short window are merged first, so chords played with a few milliseconds of spread still group together.

Two sequence taggers read the whole piece and emit a probability for each note:

| Model | Layers | Parameters |
| --- | --- | --- |
| Bidirectional LSTM | 2 × 192 | 1.58M |
| Transformer encoder | 4 × 192, 4 heads | 1.75M |

Their probabilities are averaged; the argmax of the average decides the hand. Long files are processed in windows of 1024 notes.

### Training data

30,382 two-hand piano scores taken from [PDMX](https://github.com/piano-music-dataset/pdmx) (MuseScore-derived MusicXML), 20.4M notes. Labels come from the score itself: in a grand-staff part, `<staff>1</staff>` notes are the right hand and `<staff>2</staff>` notes are the left hand. Pieces are deduplicated by content and split 24,306 / 3,038 / 3,038 into train / validation / test.

### Accuracy

| Test set | Accuracy | Hand-switch F1 | Hard cases |
| --- | --- | --- | --- |
| Held-out PDMX split (3,038 pieces, 2.0M notes) | 94.30 | 0.918 | 93.34 |
| Yamaha Disklavier pieces (50 pieces, unseen domain) | 95.91 | 0.927 | 91.87 |

"Accuracy" is the share of notes assigned to the correct hand. "Hand-switch F1" scores the predicted boundaries between hands against the true boundaries. "Hard cases" restricts the score to pieces where the two hands sit within 12 semitones of each other.

---

## Training your own model

Only MusicXML scores that encode hands are needed — a grand-staff part, or two separate piano parts. The label comes from the staff number.

```bash
# 1. build the dataset (pieces.npz + meta.json)
python -m piano_hand_assignment.training.data path/to/scores/ -o data/dataset

# 2. train
python -m piano_hand_assignment.training.train \
    --data data/dataset --arch both \
    --epochs 40 --batch-size 128 --lr 8e-4 \
    --output-dir checkpoints
```

`--arch` accepts `lstm`, `transformer` or `both`. Checkpoints are written as `checkpoints/lstm.pt` and `checkpoints/transformer.pt`, in the same format the inference code reads.

Dataset format, one array per piece named `p0`, `p1`, ...:

| Row | Content |
| --- | --- |
| 0 | onset in quarter notes |
| 1 | MIDI pitch |
| 2 | duration in quarter notes |
| 3 | label, 0 = right hand, 1 = left hand |

`meta.json` holds the file names and the train/val/test split.

Point inference at local weights with `--checkpoint-dir checkpoints`, or upload them to the Hub and let the default download take over.

## Configuration

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `PHA_HF_REPO` | `anime-song/piano-hand-assignment` | model repository |
| `PHA_CACHE_DIR` | `~/.cache/piano_hand_assignment` | download cache |

`HandAssigner(checkpoint_dir=...)` and `pha-assign --checkpoint-dir ...` bypass the Hub entirely. `--archs lstm` runs a single model if you prefer the smaller footprint.

## Limitations

- Passages where the hands cross or overlap for long stretches are the main source of errors; the model has no hand-position model to appeal to.
- Simultaneous notes at the same pitch can only carry one hand label.
- Notes on MIDI channel 10 (drums) are ignored by design and stay where they are.
- The model expects a piano part; on other instruments the assignment is meaningless.

## Acknowledgements

The model was trained on [PDMX](https://github.com/piano-music-dataset/pdmx), a large dataset of MuseScore piano scores with staff annotations. Evaluation used Yamaha Disklavier performances.

## License

MIT — see [LICENSE](LICENSE).
