"""Train one small LM per trace format and compare them on the point-in-region task.

    python scripts/run_experiment.py
    python scripts/run_experiment.py --steps 4000 --eval-n 128

Models are trained on problems with 2 or 3 inequalities and evaluated on 2, 3 and
4. Four inequalities were never seen in training, so that column measures
length generalisation. Results are written to artifacts/results.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dot_lm.decode import generate  # noqa: E402
from dot_lm.model import CharLM, Config  # noqa: E402
from dot_lm.tasks import full_text, last_line, prompt_text, sample_problem, score_dot  # noqa: E402
from dot_lm.tokens import EOS, PAD, STOI, detokenize, encode, tokenize  # noqa: E402
from dot_lm.validator import Violation, validate  # noqa: E402

ART = Path(__file__).resolve().parents[1] / "artifacts"
FORMATS = ("direct", "chain", "dot")
TRAIN_CONDS = (2, 3)


def batch_of(rng, fmt, size):
    seqs = []
    for _ in range(size):
        p = sample_problem(rng, TRAIN_CONDS[int(rng.integers(0, len(TRAIN_CONDS)))])
        seqs.append(encode(full_text(p, fmt, rng)))
    width = max(map(len, seqs))
    ids = np.full((size, width), STOI[PAD], dtype=np.int32)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = s
    return mx.array(ids)


def loss_fn(model, ids):
    logits = model(ids[:, :-1])
    targets = ids[:, 1:]
    mask = (targets != STOI[PAD]).astype(mx.float32)
    return (nn.losses.cross_entropy(logits, targets, reduction="none") * mask).sum() / mask.sum()


def train(fmt, args):
    mx.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = CharLM(Config())
    mx.eval(model.parameters())
    schedule = optim.join_schedules(
        [optim.linear_schedule(0, args.lr, 100), optim.cosine_decay(args.lr, args.steps - 100)], [100]
    )
    opt = optim.AdamW(learning_rate=schedule, weight_decay=0.01)
    step = nn.value_and_grad(model, loss_fn)
    start = time.perf_counter()
    for i in range(1, args.steps + 1):
        loss, grads = step(model, batch_of(rng, fmt, args.batch))
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss)
        if i % 250 == 0 or i == 1:
            print(f"  [{fmt}] step {i:5d}  loss {float(loss):.4f}", flush=True)
    print(f"  [{fmt}] trained in {time.perf_counter() - start:.0f}s", flush=True)
    ART.mkdir(exist_ok=True)
    model.save_weights(str(ART / f"{fmt}.safetensors"))
    return model


def load(fmt):
    model = CharLM(Config())
    model.load_weights(str(ART / f"{fmt}.safetensors"))
    mx.eval(model.parameters())
    return model


def evaluate(model, fmt, problems, constrained=False):
    prompts = [tokenize(prompt_text(p, fmt)) for p in problems]
    budget = {"direct": 8, "chain": 160, "dot": 2600}[fmt]
    completions = generate(model, prompts, budget, constrained=constrained)
    stats = dict(answer=0, valid=0, faithful=0, sound=0, finished=0, tokens=0)
    for p, prompt, out in zip(problems, prompts, completions):
        stats["finished"] += out[-1:] == [EOS]
        stats["tokens"] += len(out)
        expected = "yes" if p.answer() else "no"
        if fmt != "dot":
            stats["answer"] += last_line(detokenize(out)) == expected
            continue
        try:
            diagram = validate(prompt + out)
        except Violation:
            continue
        stats["valid"] += 1
        result = score_dot(p, diagram)
        stats["faithful"] += result["faithful"]
        stats["sound"] += result["sound"]
        stats["answer"] += result["correct"]
    n = len(problems)
    return {k: (v / n if k != "tokens" else v / n) for k, v in stats.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-n", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reuse", action="store_true", help="load saved weights instead of training")
    args = ap.parse_args()

    models = {f: (load(f) if args.reuse else train(f, args)) for f in FORMATS}

    results = {}
    for n_conds in (2, 3, 4):
        rng = np.random.default_rng(1000 + n_conds)
        problems = [sample_problem(rng, n_conds) for _ in range(args.eval_n)]
        yes_rate = float(np.mean([p.answer() for p in problems]))
        row = {"yes_rate": yes_rate}
        for fmt in FORMATS:
            row[fmt] = evaluate(models[fmt], fmt, problems)
        row["dot+mask"] = evaluate(models["dot"], "dot", problems, constrained=True)
        results[f"{n_conds}_conditions"] = row
        print(f"evaluated {n_conds} conditions", flush=True)

    (ART / "results.json").write_text(json.dumps(results, indent=2))

    print("\nanswer accuracy (majority-class rate in brackets)")
    print("| conditions | direct | chain | dot | dot + validator mask |")
    print("|---:|---:|---:|---:|---:|")
    for key, row in results.items():
        base = max(row["yes_rate"], 1 - row["yes_rate"])
        cells = " | ".join(f"{row[f]['answer']:.2f}" for f in ("direct", "chain", "dot", "dot+mask"))
        print(f"| {key.split('_')[0]} [{base:.2f}] | {cells} |")

    print("\ndot traces: share that the validator accepts / critic marks all correct / validated claims all true / finished")
    print("| conditions | decoding | accepted | faithful | sound | finished | mean tokens |")
    print("|---:|---|---:|---:|---:|---:|---:|")
    for key, row in results.items():
        for name in ("dot", "dot+mask"):
            r = row[name]
            print(
                f"| {key.split('_')[0]} | {'masked' if name == 'dot+mask' else 'free'} | {r['valid']:.2f} | "
                f"{r['faithful']:.2f} | {r['sound']:.2f} | {r['finished']:.2f} | {r['tokens']:.0f} |"
            )


if __name__ == "__main__":
    main()
