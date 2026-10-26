"""HTTP API: upload a MIDI file, download the same file with hands separated."""

from __future__ import annotations

from urllib.parse import quote

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from .predict import HandAssigner

app = FastAPI(
    title="Piano Hand Assignment",
    version="0.1.0",
    description="Split the piano part of a MIDI file into right- and left-hand tracks.",
)
_assigner: HandAssigner | None = None


def get_assigner() -> HandAssigner:
    global _assigner
    if _assigner is None:
        _assigner = HandAssigner()
    return _assigner


@app.get("/health")
def health() -> JSONResponse:
    assigner = get_assigner()
    return JSONResponse({"status": "ok",
                         "architectures": list(assigner.archs),
                         "device": str(assigner.device)})


@app.post("/assign", responses={200: {"content": {"audio/midi": {}}}})
async def assign(file: UploadFile = File(...)) -> Response:
    """Assign hands to a MIDI file.

    Send the MIDI as multipart form data under the field ``file``. The response
    body is a Standard MIDI File: note events are moved into ``Right Hand`` and
    ``Left Hand`` tracks, everything else is left untouched.
    """
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    try:
        out = get_assigner().assign_bytes(data)
    except Exception as exc:                      # malformed MIDI, unsupported input
        raise HTTPException(status_code=422, detail=f"could not process MIDI: {exc}") from exc
    name = (file.filename or "output.mid").rsplit(".", 1)[0]
    out_name = f"{name}_hands.mid"
    ascii_name = name.encode("ascii", "ignore").decode() or "output"
    headers = {
        "Content-Disposition":
            f'attachment; filename="{ascii_name}_hands.mid"; '
            f"filename*=UTF-8''{quote(out_name)}",
    }
    return Response(content=out, media_type="audio/midi", headers=headers)


def main() -> None:
    import uvicorn
    uvicorn.run("piano_hand_assignment.api:app", host="0.0.0.0", port=8000)
