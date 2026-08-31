"""AutoDL job preflight. Exit nonzero if 1K/kit cannot run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.config import Settings
from agent.env.datasets import files, find_scale_dir, present, resolve_data_dir
from agent.env.probe import hardware, snapshot

SEARCH_ROOTS = (
    Path("/root/autodl-tmp"),
    Path("/root/autodl-fs"),
    Path("/root/autodl-nas"),
    Path.home() / "autodl-tmp",
)


def discover_scale(scale: str, settings: Settings) -> Path | None:
    found = find_scale_dir(settings.data_dir, scale, settings)
    if found is not None:
        return found
    for root in SEARCH_ROOTS:
        if not root.is_dir():
            continue
        for child in root.iterdir():
            cand = child / "data" if child.is_dir() else None
            if cand is not None and present(cand, scale):
                return cand
    return None


def discover_kit(settings: Settings) -> Path | None:
    kit = Path(settings.kit_dir)
    if (kit / "evaluate.py").is_file():
        return kit
    for root in SEARCH_ROOTS:
        if not root.is_dir():
            continue
        direct = root / "kuairand-starter-kit" / "evaluate.py"
        if direct.is_file():
            return direct.parent
        for child in root.iterdir():
            ev = child / "evaluate.py"
            if child.is_dir() and ev.is_file() and "kuairand" in child.name.lower():
                return child
    return None


def check_ready(settings: Settings, torch_mod=None) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    scale = str(getattr(settings, "data_scale", "") or "") or "auto"
    hw = hardware(torch_mod)
    try:
        data_dir = resolve_data_dir(settings, {"data_scale": scale} if scale in {"pure", "1k", "27k"} else {})
    except FileNotFoundError as exc:
        data_dir = Path(settings.data_dir)
        errors.append(str(exc))
    if scale in {"1k", "27k"}:
        found = discover_scale(scale, settings)
        if found is None:
            errors.append(f"{scale} logs not found (need {files(scale)['video_basic']})")
        else:
            data_dir = found
        if not hw.get("torch"):
            errors.append("model_family=torch needs PyTorch on this 1K/27K job")
        elif not hw.get("cuda"):
            warnings.append("CUDA not visible; 1K torch will run on CPU and may miss the wall-clock")
        if hw.get("vram_gb") is not None and float(hw["vram_gb"]) < 12 and scale == "1k":
            warnings.append(f"VRAM {hw['vram_gb']}GB is tight for 1K; 24GB is the comfortable floor")
    kit = discover_kit(settings)
    if kit is None:
        errors.append("starter kit evaluate.py not found")
    spec = files("1k" if scale == "auto" else scale) if scale in {"pure", "1k", "27k"} else files("pure")
    if scale in {"pure", "1k", "27k"}:
        spec = files(scale)
        for key in ("train_log", "rest_log", "video_basic"):
            path = Path(data_dir) / spec[key]
            if not path.is_file():
                errors.append(f"missing {path.name}")
    probe = snapshot(settings, torch_mod=torch_mod)
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "data_dir": str(data_dir),
        "kit_dir": str(kit) if kit is not None else str(settings.kit_dir),
        "data_scale": scale,
        "probe": probe,
    }


def render_ready(rec: dict[str, Any]) -> str:
    lines = [
        f"autodl_ready ok={int(bool(rec.get('ok')))} scale={rec.get('data_scale')}",
        f"data_dir={rec.get('data_dir')}",
        f"kit_dir={rec.get('kit_dir')}",
    ]
    for warn in rec.get("warnings") or []:
        lines.append(f"warning: {warn}")
    for err in rec.get("errors") or []:
        lines.append(f"error: {err}")
    return "\n".join(lines) + "\n"
