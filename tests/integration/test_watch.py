import ast
import json
import os
import pty
import select
import struct
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hsi.cli import main, template
from hsi.watch import PANELS, TABS, discover, load, render_text
from hsi.watch.panels import bar, ci_bar, eta, spark

REPO = Path(__file__).resolve().parents[2]
CLI = "import sys; from hsi.cli import main; sys.exit(main(sys.argv[1:]))"


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("watch")
    os.environ["HSI_RUN_ROOT"] = str(tmp / "runs")
    os.environ["HSI_CUSTODY_ROOT"] = str(tmp / "custody")
    t = template("E1")
    t["params"] = {"families": ["rule_induction"], "trace_len": 40, "pilot_tasks": 4, "min_n": 8,
                   "max_n": 8}
    (tmp / "p.json").write_text(json.dumps(t))
    assert main(["packet", "freeze", str(tmp / "p.json")]) == 0
    f = str(tmp / "p.frozen.json")
    for cmd in ("pilot", "lock", "run"):
        assert main([cmd, f]) == 0
    rd = next((tmp / "runs").iterdir())
    assert main(["eval", str(rd)]) == 0
    assert main(["verify-receipt", str(rd / "eval" / "receipt.json")]) == 0
    return rd


def test_producers_write_finished_progress(run_dir: Path) -> None:
    prog = {p.stem: json.loads(p.read_text()) for p in (run_dir / "progress").glob("*.json")}
    assert {"pilot-variance", "run", "eval"} <= set(prog)
    assert all(p["finished"] is not None and p["done"] == p["total"] > 0 for p in prog.values())
    assert "score" not in json.dumps(prog) and "success" not in json.dumps(prog)  # counts only


def test_load_and_all_panels_render(run_dir: Path) -> None:
    v = load(run_dir)
    assert [s.state for s in v.stages] == ["done"] * 6 + ["pending"]
    assert v.experiment == "E1" and v.verified and v.receipt is not None
    text = render_text(v, TABS, 100)
    assert "unavailable" not in text and "RECEIPT · " in text and "independent recompute: OK" in text
    assert all(len(line) <= 100 for line in text.splitlines())
    assert "text" in text and "typed_event" in text


def test_tolerates_partial_and_corrupt_artifacts(run_dir: Path, tmp_path: Path) -> None:
    import shutil
    d = tmp_path / "copy"
    shutil.copytree(run_dir, d)
    (d / "pilot.json").write_text('{"n_per_fam')                          # half-written json
    (d / "run" / "episodes.jsonl").write_text('{"episode_id": "x"\n{bad\n')  # garbage lines
    (d / "eval" / "receipt.json").write_text("[]")                         # wrong shape
    (d / "progress" / "run.json").write_text("nope")
    v = load(d)
    assert v.pilot is None and v.receipt is None
    for t in TABS:
        PANELS[t](v, 100)  # must not raise


def test_discover_and_list(run_dir: Path) -> None:
    assert discover(run_dir.parent) == [run_dir]
    r = subprocess.run([sys.executable, "-c", CLI, "watch", str(run_dir.parent), "--once"],
                       capture_output=True, text=True, check=False, cwd=REPO)
    assert r.returncode == 0 and run_dir.name in r.stdout and "E1 v1" in r.stdout


def test_helpers() -> None:
    assert bar(0.5, 10) == "█████░░░░░" and bar(2, 4) == "████" and bar(-1, 4) == "░░░░"
    assert spark([0, 0.5, 1]) == "▁▄█"
    b = ci_bar(0.1, 0.05, 0.15, 0.05, width=41)
    assert len(b) == 41 and b.index("●") > b.index("|") and "[" in b and "]" in b
    assert eta(50, 100, started=0, now=100) == "eta 1m40s" and eta(0, 10, 0, 5) == ""


def test_watch_never_touches_custody() -> None:
    for py in (REPO / "src" / "hsi" / "watch").glob("*.py"):
        assert "custody_root" not in py.read_text()
        tree = ast.parse(py.read_text())
        mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        assert not {m for m in mods if m.startswith(("hsi.pipeline", "hsi.evaluation"))}


@pytest.mark.skipif(os.name != "posix", reason="needs a pty")
def test_interactive_tui_in_a_pty(run_dir: Path) -> None:
    import fcntl
    import termios

    pid, fd = pty.fork()
    if pid == 0:  # child
        os.environ.update(TERM="xterm-256color", HSI_RUN_ROOT=str(run_dir.parent))
        os.execv(sys.executable, [sys.executable, "-c", CLI, "watch", str(run_dir)])
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 120, 0, 0))
    out = b""

    def pump(secs: float) -> None:
        nonlocal out
        end = time.time() + secs
        while time.time() < end:
            if select.select([fd], [], [], 0.1)[0]:
                try:
                    out += os.read(fd, 65536)
                except OSError:
                    return

    pump(1.5)
    for key in (b"2", b"4", b"j", b"k", b"\x1bOC", b"\x1bOD", b"5"):
        try:
            os.write(fd, key)
        except OSError:
            pytest.fail("TUI exited early: " + out.decode("utf-8", errors="replace")[-1500:])
        pump(0.4)
    os.write(fd, b"q")
    pump(1.0)
    _, status = os.waitpid(pid, 0)
    text = out.decode("utf-8", errors="replace")
    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0, text[-500:]
    for needle in ("hsi watch", "MECHANISM", "RECEIPT", "EVENT STREAM"):
        assert needle in text
