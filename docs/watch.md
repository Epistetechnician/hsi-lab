# `hsi watch` — terminal mechanism view

A read-only, auto-refreshing terminal view of any run, for every experiment (E1–E7, including the
LLM variants). It reads only the run root (`HSI_RUN_ROOT`, default `~/hsi-runs`), never the
custody root, and shows progress as counts only (no partial held-out scores), so watching cannot
leak or bias a locked evaluation.

```
hsi watch                      # list all runs; ↑/↓ select, Enter open
hsi watch ~/hsi-runs/<id>      # open one run directly
hsi watch <run> --once         # text snapshot (also what you get when stdout is not a TTY)
hsi watch <run> --once --tab receipt --width 120
hsi watch <run> --log chain.out    # also tail a log file in the events tab
```

Keys: `1`–`5` or `←/→` switch tabs · `j/k` or `↑/↓` scroll · `PgUp/PgDn` · `r` refresh · `b` back to
the list · `q` quit (ESC does not quit).

## Tabs

1. **overview** — hypothesis and threshold, the 7-stage pipeline (`packet pilot lock run eval verify
   decide`) with the active stage, live progress bars with ETA, running `hsi` processes, and the
   result line (decision, guardrails, independent verification, clean-env rerun).
2. **mechanism** — what the experiment is actually doing, chosen from the artifacts present:
   arms with roles (treatment/baseline/ref); LLM backbone, pinned weights digest, per-adapter
   table (loss, generator diversity/validity, synthetic/real counts) and generator health by depth
   (E4/E6); verifier decisions by reason (E2); message counts and bytes against the cap (E3/E5);
   link open/drop churn (E5); relation events by op and arm (E7); dev-split rollout success per arm
   and family (all).
3. **data** — ledger sizes, training-data composition by origin/eligibility, a check that no
   held-out split was touched in training, all digests (packet, lock, run, model, data, receipt),
   resources.
4. **receipt** — decision, an ASCII interval chart per comparison against zero and the threshold,
   guardrail values, eval-time generation quality, success by arm, an on-demand independent
   recomputation of the receipt with the standalone validator, and the clean-environment rerun.
5. **events** — newest records across the run's ledgers and logs, plus an optional `--log` tail.

## How progress works

Producers call `hsi.progress.Progress(run_dir, stage)` and `update(done, total, note)`; one small
JSON file per stage is replaced atomically under `<run>/progress/` (`pilot-prepare`,
`pilot-variance`, `run`, `eval`). Writes are best-effort and cannot fail a run. Runs that predate
this (or stages without it) fall back to inferring progress from files, e.g. adapters trained.

## Adding a new experiment's mechanism panel

Panels are pure functions `RunView -> list[(text, style)]` in `src/hsi/watch/panels.py`. To
surface a new artifact, write `_my_panel(v)` using `v.rows("run/<file>.jsonl")` (tolerant of partial
lines) and add it to the tuple in `mechanism()`. Because panels are pure they are testable without
a terminal; `tests/integration/test_watch.py` includes a real pseudo-terminal smoke test.
