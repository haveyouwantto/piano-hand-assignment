"""Model weights are pulled from the Hub on first use.

The Hugging Face repository is the primary source; the GitHub release carries
the same two files as a fallback for environments where the Hub is blocked.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

HF_REPO_ID = os.environ.get("PHA_HF_REPO", "hywt232/piano-hand-assignment")
GITHUB_BASE = os.environ.get(
    "PHA_GITHUB_BASE",
    "https://github.com/haveyouwantto/piano-hand-assignment/releases/latest/download")
CHECKPOINT_FILES = {"lstm": "lstm.pt", "transformer": "transformer.pt"}


def cache_dir() -> Path:
    root = os.environ.get("PHA_CACHE_DIR")
    if root:
        return Path(root).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(base) / "piano_hand_assignment"


def _download(url: str, target: Path) -> None:
    """Stream a file to disk. Uses requests when available so that HTTP/SOCKS
    proxies from the environment are honoured."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    try:
        import requests
    except ImportError:
        requests = None
    if requests is not None:
        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with open(tmp, "wb") as out:
                for block in response.iter_content(1 << 20):
                    out.write(block)
    else:
        with urllib.request.urlopen(url, timeout=60) as response, open(tmp, "wb") as out:
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                out.write(block)
    tmp.replace(target)


def _from_hub(repo_id: str, filename: str) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        hf_hub_download = None
    if hf_hub_download is not None:
        return Path(hf_hub_download(repo_id=repo_id, filename=filename,
                                    cache_dir=str(cache_dir())))
    target = cache_dir() / filename
    if not target.exists():
        _download(f"https://huggingface.co/{repo_id}/resolve/main/{filename}", target)
    return target


def _from_github(filename: str) -> Path:
    target = cache_dir() / filename
    if not target.exists() or target.stat().st_size == 0:
        _download(f"{GITHUB_BASE}/{filename}", target)
    return target


def ensure_checkpoints(repo_id: str | None = None,
                       checkpoint_dir: str | Path | None = None,
                       archs: tuple[str, ...] = ("lstm", "transformer")) -> dict[str, Path]:
    """Return local paths for the requested checkpoints, downloading if needed."""
    if checkpoint_dir is not None:
        root = Path(checkpoint_dir).expanduser()
        paths = {a: root / CHECKPOINT_FILES[a] for a in archs}
        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(f"checkpoint(s) not found: {', '.join(missing)}")
        return paths

    repo_id = repo_id or HF_REPO_ID
    paths: dict[str, Path] = {}
    for arch in archs:
        filename = CHECKPOINT_FILES[arch]
        try:
            paths[arch] = _from_hub(repo_id, filename)
        except Exception as exc:                      # offline, blocked or gated
            print(f"[piano-hand-assignment] Hub download failed ({type(exc).__name__}), "
                  f"falling back to the GitHub release")
            paths[arch] = _from_github(filename)
    return paths
