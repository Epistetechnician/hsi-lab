import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "hsi"


def imports(pkg: str) -> set[str]:
    out: set[str] = set()
    for py in (SRC / pkg).rglob("*.py"):
        for n in ast.walk(ast.parse(py.read_text())):
            if isinstance(n, ast.ImportFrom) and n.module:
                out.add(n.module)
                out |= {f"{n.module}.{a.name}" for a in n.names}
            elif isinstance(n, ast.Import):
                out |= {a.name for a in n.names}
    return out


def test_generator_is_blind() -> None:
    bad = ("hsi.verification", "hsi.environment", "hsi.evaluation", "hsi.pipeline", "hsi.paths")
    assert not [i for i in imports("synthesis") if i.startswith(bad)]


def test_verifier_never_sees_generator() -> None:
    assert not [i for i in imports("verification") if i.startswith(("hsi.synthesis", "hsi.paths"))]


def test_only_evaluator_reads_custody() -> None:
    users = [p.name for p in SRC.rglob("*.py") if "custody_root" in p.read_text()]
    assert sorted(users) == ["locked.py", "paths.py", "pipeline.py"]


def test_relations_and_collective_never_touch_evaluator_or_custody() -> None:
    bad = ("hsi.evaluation", "hsi.pipeline", "hsi.paths", "hsi.verification", "hsi.synthesis")
    for pkg in ("relations", "collective"):
        assert not [i for i in imports(pkg) if i.startswith(bad)], pkg
