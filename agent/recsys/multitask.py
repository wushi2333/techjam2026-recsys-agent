from __future__ import annotations

from dataclasses import dataclass

from agent.config import Settings


@dataclass(frozen=True)
class MultiTaskSpec:
    enabled: bool
    main_task: str
    auxiliary: tuple[str, ...]

    def describe(self) -> str:
        aux = ", ".join(self.auxiliary)
        return f"main={self.main_task}; aux=[{aux}]; enabled={self.enabled}"


def spec_from(settings: Settings) -> MultiTaskSpec:
    # Kit scores long_view. Click is an auxiliary signal, not the primary label.
    return MultiTaskSpec(
        enabled=settings.mtl_enabled,
        main_task=settings.mtl_main,
        auxiliary=settings.mtl_aux,
    )
