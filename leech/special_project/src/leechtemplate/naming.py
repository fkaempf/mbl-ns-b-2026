"""Controlled vocabulary of identified cell names.

Loads the seed list from cell_names.yaml. Names not in the list are allowed by the
annotation tool but flagged, so typos and genuinely new identifications are visible
rather than silently accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class CellName:
    name: str
    typical_aspect: str = "either"
    note: str = ""


class NameList:
    """The known-cell vocabulary loaded from a YAML file."""

    def __init__(self, cells: list[CellName]):
        self.cells = cells
        self._by_name = {c.name: c for c in cells}

    @classmethod
    def from_yaml(cls, path: str | Path) -> "NameList":
        data = yaml.safe_load(Path(path).read_text()) or {}
        entries = data.get("cells", [])
        cells = [
            CellName(
                name=e["name"],
                typical_aspect=e.get("typical_aspect", "either"),
                note=e.get("note", ""),
            )
            for e in entries
        ]
        return cls(cells)

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.cells]

    def is_known(self, name: str) -> bool:
        return name in self._by_name

    def get(self, name: str) -> CellName | None:
        return self._by_name.get(name)
