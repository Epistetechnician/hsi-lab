from hsi.verification.codetests import extract_code
from hsi.verification.humaneval import passes, tasks, user_prompt


def test_reference_solutions_pass_and_wrong_code_fails() -> None:
    T = tasks()
    assert len(T) == 164
    ids = ["HumanEval/0", "HumanEval/1", "HumanEval/38", "HumanEval/100"]
    for i in ids:
        t = T[i]
        ref = t.prompt + t.solution  # the reference completes the stub
        assert passes(ref, t), i
        assert not passes(f"def {t.entry_point}(*a, **k):\n    return None", t), i
        assert not passes(None, t)
    assert "def has_close_elements" in user_prompt(T["HumanEval/0"])
    reply = "```python\n" + T["HumanEval/0"].prompt + T["HumanEval/0"].solution + "\n```"
    assert passes(extract_code(reply), T["HumanEval/0"])
