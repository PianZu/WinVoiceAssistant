"""
macros.py

User phrases that expand into several commands. Edit macros.txt
(one phrase per line). If that file is missing, macros.example.txt is used.

    arbeitsmodus = open slack | open chrome | open spotify
"""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_MACROS: dict[str, list[str]] = {}
_PREFIX = re.compile(r"^(?:open|start|starte|starten|mach(?:e)?|activate|run)\s+")


def load(path: Path | None = None) -> dict[str, list[str]]:
    """Load macros. With no path, macros.txt wins over macros.example.txt."""
    global _MACROS
    if path is None:
        user = _ROOT / "macros.txt"
        example = _ROOT / "macros.example.txt"
        path = user if user.exists() else example
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    _MACROS = _parse(text)
    print(f"[macros] {len(_MACROS)} from {path.name}" if path.exists() else "[macros] none")
    return _MACROS


def match(text: str) -> list[str] | None:
    """Commands for this utterance, or None when it is not a macro name."""
    key = re.sub(r"\s+", " ", text.strip().lower())
    key = _PREFIX.sub("", key)
    commands = _MACROS.get(key)
    if not commands:
        return None
    return list(commands)


def _parse(text: str) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, rest = line.partition("=")
        key = re.sub(r"\s+", " ", key.strip().lower())
        commands = [part.strip() for part in rest.split("|") if part.strip()]
        if key and commands:
            found[key] = commands
    return found


load()
