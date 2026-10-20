from __future__ import annotations

from typing import Any

import pytest

from hsi.cli import E8_PARAMS, template
from hsi.contracts.packet import Packet
from hsi.contracts.records import Split
from hsi.environment import SEED_BASE
from hsi.evaluation.e8 import buckets, task_for
from hsi.pipeline import holdout_offsets
from hsi.verification.codetests import (
    BROKEN,
    extract_code,
    passes,
    split_ids,
    tasks,
    user_prompt,
)


def packet() -> Packet:
    return Packet.model_validate(template("E8"))


def test_data_is_pinned_and_reference_solutions_pass_the_strongest_tier() -> None:
    T = tasks()
    assert len(T) == 378
    ok = [i for i in sorted(T)[:12] if i not in BROKEN]
    assert all(passes(T[i].code, T[i], "plus") for i in ok)


def test_tiers_order_by_strength() -> None:
    t = tasks()[2]  # similar_elements: shared elements of two tuples
    wrong = "def similar_elements(a, b):\n    return tuple(a)"
    assert passes(wrong, t, "none")
    assert not passes(wrong, t, "ex")
    assert not passes(None, t, "none")
    # passes the single shown example but is wrong in general
    hacky = "def similar_elements(a, b):\n    return (4, 5) if 3 in a else tuple(set(a) & set(b))"
    assert passes(hacky, t, "ex") and not passes(hacky, t, "plus")


def test_sandbox_blocks_network_and_times_out() -> None:
    t = tasks()[2]
    net = (
        "import socket\ndef similar_elements(a,b):\n    return 1\n"
        "socket.create_connection(('1.1.1.1',80),1)"
    )
    loop = "def similar_elements(a,b):\n    while True: pass"
    assert not passes(net, t, "ex") and not passes(loop, t, "ex")


def test_extract_code_and_prompt() -> None:
    assert extract_code("x\n```python\ndef f():\n    return 1\n```\n") == "def f():\n    return 1"
    assert extract_code("no code here") is None
    assert tasks()[2].test_list[0] in user_prompt(tasks()[2])


def test_split_is_disjoint_and_buckets_are_equal() -> None:
    tune, held = split_ids(0.4)
    assert not set(tune) & set(held) and not BROKEN & (set(tune) | set(held))
    pk = packet()
    hb, tb = buckets(pk, "held"), buckets(pk, "tune")
    assert {len(v) for v in hb.values()} == {E8_PARAMS["holdout_mod"]}
    assert not {i for v in hb.values() for i in v} & {i for v in tb.values() for i in v}


def test_holdout_offsets_are_injective_over_the_task_pool() -> None:
    offs = holdout_offsets("s", 75, 75)
    assert sorted(o % 75 for o in offs) == list(range(75))
    assert holdout_offsets("s", 5) == holdout_offsets("s", 5, 0)  # old locks unchanged
    pk = packet()
    for fam in pk.params["families"]:
        base = SEED_BASE[Split.HELDOUT]
        seen = {task_for(pk, fam, Split.HELDOUT, base + o).task_id for o in offs}
        assert len(seen) == 75


def test_template_pre_registers_the_primary() -> None:
    pk = packet()
    assert pk.params["treatment"] == "d2_plus" and pk.params["baselines"] == ["d2_ex"]
    assert pk.threshold == 0.03
    q: dict[str, Any] = pk.params
    assert {c["tier"] for c in q["chains"]} >= {"none", "ex", "base", "plus"}
    assert all(a in pk.arms for a in (q["treatment"], *q["baselines"]))
    for c in q["secondary"]:
        assert c["treatment"] in pk.arms and c["baseline"] in pk.arms
    with pytest.raises(ValueError):
        Packet.model_validate({**template("E8"), "experiment": "E9"})


class FakeLLM:
    """Returns the reference solution half the time, a wrong stub otherwise."""

    runtime = "fake"

    def __init__(self) -> None:
        self.trained: list[int] = []
        self.on = False
        self.digests = 0

    def chat(self, user: str) -> str:
        return user

    def complete(self, prompt: str, max_tokens: int = 0, temp: float = 0.0, seed: int = 0) -> str:
        t = next(t for t in tasks().values() if prompt.startswith(t.prompt))
        name = t.test_list[0].split("(")[0].split()[-1]
        good = seed % 2 == 0
        code = t.code if good else f"def {name}(*a):\n    return None"
        return f"```python\n{code}\n```"

    def complete_many(
        self, prompts: list[str], max_tokens: int = 0, temp: float = 0.0, seeds: Any = None
    ) -> list[str]:
        return [self.complete(p, max_tokens, temp, s) for p, s in zip(prompts, seeds, strict=True)]

    def set_adapter(self, on: bool) -> None:
        self.on = on

    def reset_lora(self, seed: int) -> None:
        pass

    def train_lora(self, data: list[tuple[str, str]], *a: Any, **k: Any) -> float:
        self.trained.append(len(data))
        assert all(x[1].startswith("```python") for x in data) and k.get("lead") == ""
        return 0.1

    def save_adapter(self, path: Any) -> str:
        self.digests += 1
        return f"d{self.digests}"

    def load_adapter(self, path: Any) -> None:
        pass


def test_chain_keeps_only_tier_passing_solutions_and_fills_the_quota(tmp_path: Any) -> None:
    from hsi.evaluation.e8 import _chain
    from hsi.progress import Progress
    from hsi.verification import DecisionLog

    pk = Packet.model_validate(template("E8"))
    q = {**pk.params, "k": 2, "quota": 20, "n_adapters": 2, "depth": 1, "workers": 2}
    pk = pk.model_copy(update={"params": q})
    llm = FakeLLM()
    (tmp_path / "learning").mkdir()
    cands: dict[str, Any] = {}
    for spec in ({"name": "none", "tier": "none", "seed_offset": 1},
                 {"name": "plus", "tier": "plus", "seed_offset": 2}):  # fmt: skip
        _chain(pk, tmp_path, llm, DecisionLog(tmp_path / "l.jsonl"), cands, spec,
               Progress(tmp_path, "t"), lambda: None)  # fmt: skip
    assert llm.trained == [20, 20]
    assert cands["d1_plus"]["kept_precision"] == 1.0 and cands["d1_plus"]["shortfall"] == 0
    assert cands["d1_none"]["kept_precision"] < 1.0  # unfiltered keeps the wrong stubs too
    assert cands["d1_none"]["n_synthetic"] == cands["d1_plus"]["n_synthetic"]  # pool-matched


def _v3(monkeypatch: pytest.MonkeyPatch, pins: dict[str, str]) -> Packet:
    import hsi.cli as cli

    monkeypatch.setattr(cli, "_e8_v2_pins", lambda: pins)
    return Packet.model_validate(template("E8", "e8-he"))


def test_v3_scores_humaneval_and_pins_the_v2_adapters(monkeypatch: pytest.MonkeyPatch) -> None:
    pins = {"d1_plus_r0": "x"}
    pk = _v3(monkeypatch, pins)
    assert pk.version == 3 and pk.params["eval_pool"] == "humaneval"
    assert pk.params["pinned_adapters"] == pins and pk.params["reuse_from"] == "d55611f54cc09b94"
    held, tune = buckets(pk, "held"), buckets(pk, "tune")
    assert {len(v) for v in held.values()} == {54} and all(
        isinstance(i, str) and i.startswith("HumanEval/") for v in held.values() for i in v
    )
    assert all(isinstance(i, int) for v in tune.values() for i in v)  # pilot/run use MBPP+ tune
    offs = holdout_offsets("s", 54, 54)
    seen = {task_for(pk, "he_b", Split.HELDOUT, SEED_BASE[Split.HELDOUT] + o).task_id for o in offs}
    assert len(seen) == 54


def test_reuse_refuses_adapters_that_differ_from_the_pins(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from hsi.errors import DigestError
    from hsi.evaluation.llm_stack import try_reuse_adapters

    monkeypatch.setenv("HSI_RUN_ROOT", str(tmp_path))
    src = tmp_path / "d55611f54cc09b94"
    (src / "learning").mkdir(parents=True)
    (src / "learning" / "d1_plus_r0.safetensors").write_bytes(b"w")
    (src / "learning" / "training.jsonl").write_text("")
    from hsi.contracts.packet import canonical_digest

    body = {"candidates": {"d1_plus_r0": {"digest": "real"}}}
    man = {**body, "manifest_digest": canonical_digest(body)}
    (src / "learning" / "manifest.json").write_text(json.dumps(man))
    from hsi.evaluation.llm_stack import TRAIN_KEYS

    pk = _v3(monkeypatch, {"d1_plus_r0": "tampered"})
    train = {k: pk.params.get(k) for k in TRAIN_KEYS}
    (src / "packet.frozen.json").write_text(
        json.dumps({"packet": {"seeds": pk.seeds, "params": train}})
    )
    with pytest.raises(DigestError):
        try_reuse_adapters(pk, tmp_path / "new")
    pk = _v3(monkeypatch, {"d1_plus_r0": "real"})
    assert try_reuse_adapters(pk, tmp_path / "new")
    assert (tmp_path / "new" / "learning" / "d1_plus_r0.safetensors").exists()


class RefLLM(FakeLLM):
    """Batched fake: answers every prompt with the task's reference solution."""

    def complete_many(self, prompts: list[str], *a: Any, **k: Any) -> list[str]:
        from hsi.verification import humaneval as he

        out = []
        for p in prompts:
            ht = next((t for t in he.tasks().values() if t.prompt.strip() in p), None)
            if ht is not None:
                out.append("```python\n" + ht.prompt + ht.solution + "\n```")
            else:
                t = next(t for t in tasks().values() if t.prompt in p)
                out.append(f"```python\n{t.code}\n```")
        self.calls = getattr(self, "calls", 0) + 1
        return out


def test_v4_scores_both_benchmarks_with_batched_prefetch(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hsi.evaluation.e8 as e8
    import hsi.evaluation.llm_stack as ls

    pk = Packet.model_validate(template("E8", "e8-7b"))
    assert pk.params["holdout_mod"] == 81 and pk.params["gen_batch"] == 64
    llm = RefLLM()
    monkeypatch.setattr(e8, "_llm", lambda packet: llm)
    monkeypatch.setitem(ls._STATE, "run_dir", tmp_path)
    e8._STATE.pop("gen", None)
    spec = e8.E8Spec()
    base = SEED_BASE[Split.HELDOUT]
    for fam, n in (("he_b", 6), ("mbpp_c", 6)):
        rs = [spec.run_task("frozen_base", fam, Split.HELDOUT, base + o, pk) for o in range(n)]
        assert all(r.success and r.stats["valid"] == 1.0 for r in rs), fam
    assert llm.calls == 2 * 2  # 81 tasks per family / batch 64 = 2 batches, each family once
    # tune split (pilot/run) stays on the unbatched MBPP+ path
    r = spec.run_task("frozen_base", "he_a", Split.TUNE, SEED_BASE[Split.TUNE] + 1, pk)
    assert "pass_ex" in r.stats
