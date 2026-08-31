"""Environment facts: CUDA, VRAM, torch, dataset scales. Not a trial agenda."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.env.datasets import catalog


def hardware(torch_mod=None) -> dict[str, Any]:
    torch_ok = False
    version = None
    cuda = False
    name = None
    vram = None
    if torch_mod is False:
        torch_mod = None
    elif torch_mod is None:
        try:
            import torch as torch_mod
        except ImportError:
            torch_mod = None
    if torch_mod is not None:
        torch_ok = True
        version = getattr(torch_mod, "__version__", None)
        cuda_mod = getattr(torch_mod, "cuda", None)
        if cuda_mod is not None and callable(getattr(cuda_mod, "is_available", None)):
            cuda = bool(cuda_mod.is_available())
            if cuda:
                getter = getattr(cuda_mod, "get_device_name", None)
                name = getter(0) if callable(getter) else None
                props_fn = getattr(cuda_mod, "get_device_properties", None)
                if callable(props_fn):
                    props = props_fn(0)
                    total = int(getattr(props, "total_memory", 0) or 0)
                    vram = round(total / (1024**3), 2)
    lgb = False
    try:
        import lightgbm  # noqa: F401

        lgb = True
    except ImportError:
        pass
    return {
        "torch": torch_ok,
        "torch_version": version,
        "cuda": cuda,
        "cuda_name": name,
        "vram_gb": vram,
        "lightgbm": lgb,
    }


def snapshot(settings=None, torch_mod=None) -> dict[str, Any]:
    if settings is None:
        from agent.config import load_settings

        settings = load_settings()
    hw = hardware(torch_mod)
    datasets = catalog(settings)
    families = ["fm"]
    if hw["lightgbm"]:
        families.append("gbm")
    if hw["torch"]:
        families.append("torch")
    scales = [name for name, rec in datasets.items() if rec.get("present")]
    return {
        "cuda": hw["cuda"],
        "cuda_name": hw["cuda_name"],
        "vram_gb": hw["vram_gb"],
        "torch": hw["torch"],
        "torch_version": hw["torch_version"],
        "lightgbm": hw["lightgbm"],
        "datasets": datasets,
        "legal_families": families,
        "legal_scales": scales,
        "contest_hidden_test": "pure",
        "ids_reindexed": True,
    }


def render_facts(snap: dict[str, Any] | None = None, settings=None) -> list[str]:
    snap = snap or snapshot(settings)
    vram = snap.get("vram_gb")
    vram_s = "na" if vram is None else f"{vram}"
    families = ",".join(snap.get("legal_families") or [])
    scales = ",".join(snap.get("legal_scales") or [])
    lines = [
        f"env: cuda={int(bool(snap.get('cuda')))} vram_gb={vram_s} "
        f"torch={int(bool(snap.get('torch')))} lightgbm={int(bool(snap.get('lightgbm')))}",
        f"legal_families={families}",
        f"legal_scales={scales}",
        "datasets:",
    ]
    for name in ("pure", "1k", "27k"):
        rec = (snap.get("datasets") or {}).get(name) or {}
        if rec.get("present"):
            lines.append(
                f"  {name} present published_rows={rec.get('published_rows')} "
                f"log_bytes={rec.get('log_bytes')}"
            )
        else:
            lines.append(f"  {name} absent")
    lines.append(
        "env_facts: IDs are re-indexed across Pure/1K/27K so they are not the same "
        "task; do not compare 1K primary to Pure FM 0.6016; contest hidden test is Pure."
    )
    return lines


def write_probe(path: Path, snap: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(snap, indent=2), encoding="utf-8")
