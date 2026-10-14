# 0002 — Remote GPU / PyTorch backend (scoping; nothing built or provisioned)

Status: **proposed**, awaiting owner approval. The owner has said `gman` (givemeanode) may be used
for remote GPUs; this record turns that into a bounded plan and says what it would and would not
change. As of writing, `gman` is signed in with a prepaid balance and no top-up rule; the existing
H100 nodes in the workspace belong to other projects and are not touched.

## Why

Local MLX on one 36 GB Mac now limits the science, not the harness: a 7B adapter takes ~25 min to
train, a 5,000-prompt eval ~1 h, and two GPU jobs cannot overlap without a Metal out-of-memory
failure (seen once). Things we cannot do locally: larger generators (14B–32B, Apache-2.0 only),
several seeds per packet, and full-precision (non-4-bit) bases. The PRD already lists PyTorch for
portability and NeMo RL "if E4 outgrows one machine"; this is that trigger, narrowed.

It also buys something we cannot get on this Mac: a **cross-hardware, cross-stack replication**
(MLX/Metal vs PyTorch/CUDA), which is the independent-machine half of assurance level A4.

## What this changes in the PRD

v1 non-goals say "no paid compute, no Linux/CUDA". This would be an explicit, recorded exception
for training and evaluation of LLM-backed packets only. Everything model-free stays local and
unchanged. Hosted *inference endpoints* remain out of scope: `gman` rents compute we run our own
code on, which is the allowed shape.

## Constraints that shape the design

1. **Numerics differ across backends.** MLX/Metal and CUDA will not produce bit-identical logits,
   text, or adapters. The pinned-adapter digests and the "same machine, byte-identical" claim
   therefore apply *per runtime*. Receipts must record the runtime (`hsi-llm-v1/mlx` vs
   `hsi-llm-v1/torch-cuda`), and cross-backend agreement is reported as a **replication of
   conclusions** (decision and interval overlap), never as an A3 aggregate match.
2. **Custody.** The locked evaluator reads the lock secret (salt). Remote training, pilots and the
   dev-split `run` need no secret. A remote *held-out eval* would, so it must use `gman secret`
   (value never enters chat or logs) on a node dedicated to that run, with the node parked/deleted
   afterwards, and the secret rotated by re-locking only if the owner wants. Default: keep held-out
   eval for the original runtime local, and use remote for replication packets with fresh locks.
3. **No real or licensed data exists in this project**, so egress is code, synthetic worlds, and
   public checkpoints. Only Apache-2.0/MIT checkpoints (Qwen2.5-14B/32B qualify; 3B and 72B do not).
4. **Cost is bounded by balance**, billed per minute; parked nodes stop billing. No auto top-up.

## Plan (each step is a gate; stop at any failed gate)

| Step | Deliverable | Gate |
| --- | --- | --- |
| B0 | Owner approves this record, sets a spend ceiling (USD) and a per-run cap | recorded here |
| B1 | `TorchLLM` implementing the existing `LLM` contract (score, generate, LoRA train/save/load, digest, exact adapter-off) on transformers + PEFT, pinned HF revisions | parity tests: adapter identity at init and exact off-switch (bit-exact); scores agree with MLX within a stated tolerance on a fixed prompt set; argmax agrees on ≥ X% |
| B2 | Reproducible Linux environment (locked CUDA torch), runtime tag in packets/receipts | clean build from lockfile on a CPU node, then a 1-GPU smoke node |
| B3 | `hsi remote` wrapper: sync repo at a commit, run a stage detached, stream `progress/*.json` back so `hsi watch` works unchanged, pull run artifacts, park the node | one stage end-to-end on a small node; node parked and idle cost confirmed |
| B4 | Replication packets: E4-llm and E6-ground re-run on the torch runtime with fresh locks | decisions/intervals compared with the MLX receipts in a new doc |
| B5 | Scale-up packets (e.g. Qwen2.5-14B/32B generator, multiple seeds) | only if B4 shows the harness is sound |

Estimated effort: B1–B3 are the real work (a few focused sessions); B4–B5 reuse existing packets.
The first cost-bearing action is B3's smoke run, a few GPU-minutes.

## Risks

- CUDA nondeterminism may break bit-exact adapter reruns; mitigate with deterministic flags and
  by treating torch results as replications, not reproductions.
- Silent drift between PEFT LoRA and the MLX LoRA parameterization (scale, init, which layers).
  The parity gate in B1 exists for this.
- Spend overrun from a forgotten running node; mitigate with `gman` park-on-finish in the wrapper
  and a usage check at the end of every session.
- Scope creep into multi-node training; explicitly out of scope here.

## Decisions needed from the owner

1. Approve the exception to "no paid compute / no CUDA" for LLM-backed packets only.
2. Spend ceiling and per-run cap.
3. Default custody policy for remote held-out eval (recommended: do not; use fresh locks).
4. Whether B4 replication should precede or follow any new E6/E4 science.
