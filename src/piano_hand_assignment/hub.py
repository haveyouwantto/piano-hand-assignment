"""Model weights are pulled from the Hugging Face Hub on first use."""

from __future__ import annotations

import os
from pathlib import Path

HF_REPO_ID = os.environ.get("PHA_HF_REPO", "anime-song/piano-hand-assignment")
CHECKPOINT_FILES = {"lstm": "lstm.pt", "transformer": "transformer.pt"}


def cache_dir() -> Path:
    root = os.environ.get("PHA_CACHE_DIR")
    if root:
        return Path(root).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(base) / "piano_hand_assignment"


def ensure_checkpoints(repo_id: str | None = None,
                       checkpoint_dir: str | Path | None = None,
                       archs: tuple[str, ...] = ("lstm", "transformer")) -> dict[str, Path]:
    """Return local paths for the requested checkpoints, downloading if needed."""
    repo_id = repo_id or HF_REPO_ID
    if checkpoint_dir is not None:
        root = Path(checkpoint_dir).expanduser()
        paths = {a: root / CHECKPOINT_FILES[a] for a in archs}
        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(f"checkpoint(s) not found: {', '.join(missing)}")
        return paths

    paths: dict[str, Path] = {}
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:      # fall back to plain HTTPS download
        hf_hub_download = None

    for arch in archs:
        filename = CHECKPOINT_FILES[arch]
        if hf_hub_download is not None:
            paths[arch] = Path(hf_hub_download(
                repo_id=repo_id, filename=filename, cache_dir=str(cache_dir())))
        else:
            import torch
            target = cache_dir() / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}?download=true"
                torch.hub.download_url_to_file(url, str(target))
            paths[arch] = target
    return paths
