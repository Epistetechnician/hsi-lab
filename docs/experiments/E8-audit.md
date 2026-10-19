# E8 pre-run audit

`E8-audit.json` (rows carry the generated text): greedy on all non-broken tasks and 8 samples at T=0.8 per
tune task from the frozen Qwen2.5-1.5B-Instruct 4-bit. Reproduce with `scripts/code_audit.py`. Summary and the
design decisions it set are in ADR 0004.
