# hsi-lab

Local-first harness testing whether verified synthetic experience, experience codecs,
small agent collectives, and an adaptive relationship layer improve held-out adaptation
at matched cost. Every experiment (E1–E7) is a frozen packet with an independently
recomputed receipt; negative results are first-class.

PRD: docs/PRD.md

## Status (M0–M6)
Packet CLI, rule-world environment, digest-chained ledger, scripted baselines (`hsi baselines`). Other commands exit 5
(not implemented).

```
uv sync
uv run pytest
uv run hsi packet new E1 packets/e1-v1.json
```

## Quality gates

```
uv run ruff check src tests validator       # rules are explicit in pyproject.toml
uv run ruff format --check src tests validator
uv run mypy src
uv run pytest                               # model-free, ~25 s
HSI_LLM_TESTS=1 uv run --extra mlx pytest tests/integration/test_llm_backend.py   # needs the checkpoint
```

## Watching a run

`hsi watch` opens a read-only terminal view of any run (pipeline stages, mechanism per
experiment, data lineage, receipt with interval charts, live events). See docs/watch.md.

## Non-claims
No general-intelligence, understanding, novelty, or production claims. Run roots and
custody roots live outside this repo.
