"""Config loader — all hyperparameters come from weights.yaml only."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).parent.parent.parent  # fimi-cyber-poc/


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class Config:
    def __init__(
        self,
        weights_path: Path | None = None,
        scenarios_path: Path | None = None,
    ) -> None:
        wp = weights_path or (_ROOT / "config" / "weights.yaml")
        sp = scenarios_path or (_ROOT / "config" / "synthetic_scenarios.yaml")
        self._w: dict[str, Any] = _load_yaml(wp)
        self._s: dict[str, Any] = _load_yaml(sp)

    # ── top-level shortcuts ────────────────────────────────────────────────
    @property
    def seed(self) -> int:
        return int(self._w["seed"])

    @property
    def embedding(self) -> dict[str, Any]:
        return self._w["embedding"]

    @property
    def narrative(self) -> dict[str, Any]:
        return self._w["narrative"]

    @property
    def ioc_score(self) -> dict[str, Any]:
        return self._w["ioc_score"]

    @property
    def components(self) -> dict[str, Any]:
        return self._w["components"]

    @property
    def fcls(self) -> dict[str, Any]:
        return self._w["fcls"]

    @property
    def priority(self) -> dict[str, Any]:
        return self._w["priority"]

    @property
    def eval(self) -> dict[str, Any]:
        return self._w["eval"]

    @property
    def synthetic(self) -> dict[str, Any]:
        return self._s

    @property
    def data_dir(self) -> Path:
        return _ROOT / "data"

    @property
    def results_dir(self) -> Path:
        return _ROOT / "results"

    @property
    def root(self) -> Path:
        return _ROOT


def load_config(
    weights_path: Path | None = None,
    scenarios_path: Path | None = None,
) -> Config:
    return Config(weights_path, scenarios_path)
