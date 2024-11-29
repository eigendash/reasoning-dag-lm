"""Teacher-forced accuracy of the saved DAG model, split by what is being predicted.

    python scripts/diagnose.py

Free-running decoding mixes everything together: one wrong critic mark early on
changes the rest of the trace. Here the model reads the gold trace and we ask, at
each position, whether its argmax matches the gold next token. That separates the
bookkeeping tokens (record syntax, ids, edges) from the decisions the trace has to
get right: the numbers copied into a proposition, the critic's mark, the critic's
one-word comment, and the final answer.
"""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dot_lm.model import CharLM, Config  # noqa: E402
from dot_lm.tasks import dot_text, sample_problem  # noqa: E402
from dot_lm.tokens import STOI, tokenize  # noqa: E402

ART = Path(__file__).resolve().parents[1] / "artifacts"
ORDER = ("syntax", "prop number", "critic mark", "critic word", "answer")


def split_lines(tokens):
    spans, start = [], 0
    while start < len(tokens):
        end = start
        while end < len(tokens) and tokens[end] != "\n":
            end += 1
        spans.append((start, end))
        start = end + 1
    return spans


def categorise(tokens):
    """Label every token position of a gold trace with the kind of prediction it is."""
    cats = ["syntax"] * len(tokens)
    spans = split_lines(tokens)
    text = ["".join(tokens[a:b]) for a, b in spans]
    for i, (a, b) in enumerate(spans):
        if "@status" in text[i]:
            cats[a + text[i].index("mark=") + 5] = "critic mark"
            cats[spans[i + 1][0]] = "critic word"
        elif "@prop" in text[i]:
            for j in range(a + text[i].index("{"), b):
                if tokens[j].isdigit():
                    cats[j] = "prop number"
    cats[spans[-2][0]] = "answer"  # the line before <eos>
    return cats


def main():
    model = CharLM(Config())
    model.load_weights(str(ART / "dot.safetensors"))
    rng = np.random.default_rng(777)
    hits, counts = dict.fromkeys(ORDER, 0), dict.fromkeys(ORDER, 0)
    validated = 0
    for k in (2, 3):
        for _ in range(60):
            tokens = tokenize(dot_text(sample_problem(rng, k), rng))
            ids = mx.array([[STOI[t] for t in tokens]])
            pred = np.array(mx.argmax(model(ids[:, :-1]), axis=-1))[0]
            gold = np.array(ids)[0, 1:]
            for cat, ok, g in zip(categorise(tokens)[1:], pred == gold, np.array(ids)[0, 1:]):
                hits[cat] += int(ok)
                counts[cat] += 1
                validated += cat == "critic mark" and g == STOI["v"]
    total = sum(counts.values())
    print("| position type | share of tokens | teacher-forced accuracy |")
    print("|---|---:|---:|")
    for cat in ORDER:
        print(f"| {cat} | {counts[cat] / total:.3f} | {hits[cat] / counts[cat]:.3f} |")
    share = validated / counts["critic mark"]
    print(f"\nmarks that are 'validated' in the gold traces: {share:.3f} (majority-class rate {max(share, 1 - share):.3f})")


if __name__ == "__main__":
    main()
