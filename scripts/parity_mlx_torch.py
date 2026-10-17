"""B1 parity: compare MLX (4-bit) and PyTorch (full precision) hypothesis scores on the same
prompts. Different quantisation/numerics, so this reports agreement, not equality."""

import json
import random
import sys

from hsi.agents.learner import HYPOTHESES
from hsi.environment.world import _apply
from hsi.learning.llm import LLM, describe, render_prompt
from hsi.learning.llm_torch import TorchLLM


def main(n_prompts: int = 40, n_hyp: int = 300) -> dict:
    rng = random.Random("parity")
    a, b = LLM(), TorchLLM()
    top1 = 0
    corr = []
    for _ in range(n_prompts):
        h = HYPOTHESES[rng.randrange(len(HYPOTHESES))]
        pairs = []
        for _ in range(3):
            x = tuple(rng.randrange(7) for _ in range(5))
            st = list(x)
            for p in h:
                _apply(p, st)
            pairs.append((x, tuple(st)))
        idx = rng.sample(range(len(HYPOTHESES)), n_hyp)
        texts = [describe(HYPOTHESES[i]) for i in idx]
        pr = render_prompt(pairs)
        sa, sb = a.score(pr, texts), b.score(pr, texts)
        top1 += max(range(n_hyp), key=lambda i: sa[i]) == max(range(n_hyp), key=lambda i: sb[i])
        ma, mb = sum(sa) / n_hyp, sum(sb) / n_hyp
        cov = sum((x - ma) * (y - mb) for x, y in zip(sa, sb, strict=True))
        va = sum((x - ma) ** 2 for x in sa) ** 0.5
        vb = sum((y - mb) ** 2 for y in sb) ** 0.5
        corr.append(cov / (va * vb))
    return {
        "prompts": n_prompts,
        "hypotheses_per_prompt": n_hyp,
        "top1_agreement": top1 / n_prompts,
        "mean_pearson": sum(corr) / len(corr),
        "min_pearson": min(corr),
    }


if __name__ == "__main__":
    print(json.dumps(main(*(int(x) for x in sys.argv[1:])), indent=2))
