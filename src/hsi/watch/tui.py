"""Curses front end for `hsi watch`: run list -> run detail with tabs; auto-refreshes."""

from __future__ import annotations

import contextlib
import curses
import locale
import os
import time
from pathlib import Path

from hsi.watch.model import RunView, discover, load
from hsi.watch.panels import PANELS, TABS, Line, list_rows

REFRESH_S = 1.0


def _colors() -> dict[str, int]:
    curses.start_color()
    curses.use_default_colors()
    for i, c in enumerate(
        [
            curses.COLOR_GREEN,
            curses.COLOR_YELLOW,
            curses.COLOR_RED,
            curses.COLOR_CYAN,
            curses.COLOR_MAGENTA,
        ],
        start=1,
    ):
        curses.init_pair(i, c, -1)
    return {
        "": 0,
        "ok": curses.color_pair(1),
        "warn": curses.color_pair(2),
        "bad": curses.color_pair(3),
        "hdr": curses.color_pair(4) | curses.A_BOLD,
        "acc": curses.color_pair(5) | curses.A_BOLD,
        "dim": curses.A_DIM,
    }


def _put(win: curses.window, y: int, x: int, text: str, attr: int, w: int) -> None:
    with contextlib.suppress(curses.error):
        win.addnstr(y, x, text, max(0, w - x - 1), attr)


def _draw(win: curses.window, lines: list[Line], top: int, col: dict[str, int], y0: int) -> int:
    h, w = win.getmaxyx()
    for i, (text, st) in enumerate(lines[top : top + h - y0 - 1]):
        _put(win, y0 + i, 0, text, col.get(st, 0), w)
    return len(lines)


def _loop(stdscr: curses.window, root: Path, run: Path | None, log: Path | None) -> None:
    locale.setlocale(locale.LC_ALL, "")
    curses.curs_set(0)
    col = _colors()
    stdscr.timeout(int(REFRESH_S * 1000))
    sel, tab, top, cur = 0, 0, 0, run
    last = 0.0
    views: list[RunView] = []
    view: RunView | None = None
    while True:
        h, w = stdscr.getmaxyx()
        now = time.time()
        if now - last >= REFRESH_S:
            if cur is None:
                views = [load(p) for p in discover(root)]
            else:
                view = load(cur)
            last = now
        stdscr.erase()
        if h < 8 or w < 40:
            _put(stdscr, 0, 0, "terminal too small", 0, w)
        elif cur is None:
            _put(stdscr, 0, 0, f"hsi watch · {root} · {len(views)} runs", col["hdr"], w)
            lines = list_rows(views)
            for i, (text, st) in enumerate(lines):
                attr = col.get(st, 0) | (curses.A_REVERSE if i - 1 == sel and i > 0 else 0)
                _put(stdscr, 2 + i, 0, text, attr, w)
            _put(stdscr, h - 1, 0, "↑/↓ select · Enter open · r refresh · q quit", col["dim"], w)
        elif view is not None:
            tabbar = "  ".join(
                f"[{i + 1}]{t}" if i == tab else f" {i + 1} {t}" for i, t in enumerate(TABS)
            )
            _put(stdscr, 0, 0, f"hsi watch · {view.title} · {view.digest16}", col["hdr"], w)
            _put(stdscr, 1, 0, tabbar, col["acc"], w)
            lines = PANELS[TABS[tab]](view, w)
            if log is not None and TABS[tab] == "events":
                try:
                    tail = log.read_text("utf-8", errors="replace").splitlines()[-12:]
                    lines = [
                        *lines,
                        ("", ""),
                        (f"LOG {log.name}", "hdr"),
                        *[(x, "dim") for x in tail],
                    ]
                except OSError:
                    pass
            n = _draw(stdscr, lines, top, col, 3)
            top = max(0, min(top, n - (h - 4)))
            foot = (
                "1-5/←→ tabs · j/k scroll · r refresh · "
                + ("b back · " if run is None else "")
                + "q quit"
            )
            _put(stdscr, h - 1, 0, foot, col["dim"], w)
        stdscr.refresh()
        k = stdscr.getch()
        if k == ord("q"):
            return
        if k == curses.KEY_RESIZE or k == -1:
            continue
        if k == ord("r"):
            last = 0.0
        elif cur is None:
            if k in (curses.KEY_DOWN, ord("j")):
                sel = min(len(views) - 1, sel + 1)
            elif k in (curses.KEY_UP, ord("k")):
                sel = max(0, sel - 1)
            elif k in (10, 13, curses.KEY_ENTER) and views:
                cur, last, tab, top = views[sel].path, 0.0, 0, 0
        else:
            if k in (ord("b"), curses.KEY_BACKSPACE) and run is None:
                cur, last = None, 0.0
            elif ord("1") <= k <= ord("5"):
                tab, top = k - ord("1"), 0
            elif k == curses.KEY_RIGHT:
                tab, top = (tab + 1) % len(TABS), 0
            elif k == curses.KEY_LEFT:
                tab, top = (tab - 1) % len(TABS), 0
            elif k in (curses.KEY_DOWN, ord("j")):
                top += 1
            elif k in (curses.KEY_UP, ord("k")):
                top = max(0, top - 1)
            elif k == curses.KEY_NPAGE:
                top += h - 6
            elif k == curses.KEY_PPAGE:
                top = max(0, top - (h - 6))


def run_tui(root: Path, run: Path | None = None, log: Path | None = None) -> None:
    os.environ.setdefault("ESCDELAY", "25")  # snappy arrow keys
    curses.wrapper(_loop, root, run, log)
