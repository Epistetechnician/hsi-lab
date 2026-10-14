"""`hsi watch`: read-only terminal view of any run (stages, mechanism, data, receipt, events)."""

from __future__ import annotations

from pathlib import Path

from hsi.watch.model import RunView, discover, load
from hsi.watch.panels import PANELS, TABS, Line, list_rows


def render_text(view: RunView, tabs: list[str], width: int = 100) -> str:
    out: list[str] = []
    for t in tabs:
        out.append(f"━━ {t} " + "━" * max(0, width - len(t) - 4))
        out += [text[:width] for text, _ in PANELS[t](view, width)]
        out.append("")
    return "\n".join(out)


def watch(path: Path, once: bool, tab: str, width: int, log: Path | None) -> int:
    is_run = (path / "packet.frozen.json").exists()
    if once or not _is_tty():
        if is_run:
            tabs = TABS if tab == "all" else [tab]
            print(render_text(load(path), tabs, width))
        else:
            print("\n".join(t[:width] for t, _ in list_rows([load(p) for p in discover(path)])))
        return 0
    from hsi.watch.tui import run_tui

    run_tui(path.parent if is_run else path, path if is_run else None, log)
    return 0


def _is_tty() -> bool:
    import sys

    return sys.stdout.isatty() and sys.stdin.isatty()


__all__ = ["PANELS", "TABS", "Line", "RunView", "discover", "load", "render_text", "watch"]
