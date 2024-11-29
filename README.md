# reasoning-dag-lm

Typed reasoning-DAG traces, an online validator for them, and a small language
model trained to write them, all in MLX.

This is an independent, toy-scale implementation of the serialization and
validation part of *On the Diagram of Thought* (arXiv:2409.10038). It is not a
reproduction of that paper's results, and it does not touch the category-theoretic
half of the paper. The numbers below come from a 0.87M-parameter character model on
a synthetic task and should be read as a test of the machinery, not as evidence
about large models.

## What the paper proposes, as far as it is used here

A single autoregressive model is trained to emit its reasoning as a serialized
directed acyclic graph. Nodes carry one of four roles (problem, proposer, critic,
summarizer), and four role tokens are added to the vocabulary so the model can switch
between them. Structure is written out in `@`-prefixed records: `@node` opens a node
with a fresh increasing id, `@edge` points from an earlier node to the current one
with a kind (`use`, `critique`, `refine`), `@status` lets a critic mark a proposal
`validated` or `invalidated`, and `@prop` carries a machine-checkable statement.
Because every edge ends at the node being written and starts at an earlier one, the
provenance graph is acyclic by construction. A deterministic validator tracks the
records as they are produced and can mask the decoder to well-formed continuations.
Training is ordinary next-token cross-entropy over whole traces.

## What is implemented

`dot_lm/validator.py` is a finite-control validator that works one token at a time.
`allowed()` returns the tokens that keep a trace well formed, `advance()` consumes
one, and `validate()` accepts or rejects a finished trace and returns the extracted
diagram (nodes with final states, typed edges, status records, and for each
summarizer the validated proposals it drew on). The decoder mask and the offline
checker share one code path. The rules it enforces are the ones the paper states:
fresh increasing ids; edges end at the current node; the role table for edge kinds
(critique goes proposer to critic, refine goes proposer or critic to proposer, use
goes problem or proposer to proposer or summarizer); a critic writes one critique
edge and then one status for that target; a status can only be written for a proposal
that is still active, so the first writer wins; a summarizer can only draw `use`
edges from validated proposals or the problem; free text never starts with `@`.

`dot_lm/solver.py` plays the part of the solver check the paper allows for typed
blocks. A `@prop` is a JSON comparison over integers with optional `add`, `sub`
and `mul`, for example `{"eq":[{"add":[3,4]},7]}`. The validator refuses to end a
`@prop` line unless it parses and type-checks.

`dot_lm/tasks.py` generates a point-in-region problem such as
`(04,06) in x>02, y>05, x<10?` together with gold traces in three formats: the full
DAG, a linear chain of checks, and the bare answer. In the DAG each inequality is
first claimed as written, a critic judges the claim, and a false claim is
invalidated and then repaired with its negation through a `refine` edge. With some
probability a proposer instead states a true but irrelevant fact, which the critic
invalidates as off-topic, as in the worked example in the paper's appendix. The
summarizer cites only validated proposals. A task-level scorer separately checks
that every critic mark is what a correct critic would write (faithful), that every
validated proposition is actually true (sound), and that the final answer is right.

`dot_lm/model.py` and `dot_lm/decode.py` hold a four-layer decoder-only transformer
(RoPE, RMSNorm, SwiGLU, 869,504 parameters over a 64-symbol vocabulary), an in-place
KV cache, and greedy batched decoding with an optional validator mask per sequence.

## Choices the paper leaves open

The paper's grammar lets record kinds appear in any order inside a node block. I fix
an order per role (a proposer writes its edges, then its `@prop`, then one line of
text; a critic writes one `critique` edge, then `@status`, then one line of text),
which keeps the machine small and makes the next record type predictable. A role
token is written immediately before each `@node` line, and the validator requires the
token and the `role=` field to agree. Every proposal is critiqued before the next
role token is allowed, and a summary is allowed only when no proposal is open and at
least one is validated. The paper's length-prefixed escapes, `@entails`, `@eq` and the
optional `just=` field are not implemented. Numbers in the human-readable lines are
zero-padded to two digits so that prompt length depends only on the number of
conditions; numbers inside `@prop` are plain integers because JSON does not allow
leading zeros.

## Experiment

One model per trace format, identical architecture and budget: 2,500 AdamW steps
at batch size 8 (20,000 training problems each), learning rate 1e-3 with 100
warm-up steps and cosine decay, loss over every token of the trace. Training
problems have two or three inequalities. Evaluation uses 64 problems per
condition count, identical across formats, with greedy decoding; four inequalities
were never seen in training. Each answer-accuracy cell is therefore out of 64 and
has a standard error of roughly six points.

Answer accuracy (the share of the more frequent answer in the test set is in
brackets):

| inequalities | direct | chain | DAG | DAG with validator mask |
|---:|---:|---:|---:|---:|
| 2 [0.55] | 0.64 | 1.00 | 0.61 | 0.61 |
| 3 [0.67] | 0.73 | 1.00 | 0.59 | 0.59 |
| 4 [0.84] | 0.45 | 0.69 | 0.05 | 0.42 |

Properties of the DAG model's traces, as a share of all test problems. "Accepted" is
the validator's verdict on the finished trace; faithful and sound are defined above.

| inequalities | decoding | accepted | faithful | sound | mean tokens |
|---:|---|---:|---:|---:|---:|
| 2 | free | 1.00 | 0.31 | 0.61 | 534 |
| 2 | masked | 1.00 | 0.31 | 0.61 | 534 |
| 3 | free | 1.00 | 0.19 | 0.59 | 816 |
| 3 | masked | 1.00 | 0.19 | 0.59 | 816 |
| 4 | free | 0.09 | 0.05 | 0.08 | 966 |
| 4 | masked | 1.00 | 0.09 | 0.53 | 875 |

The linear chain solves the in-distribution task completely. The DAG model does not.
It learned the format: every free-running trace for two and three inequalities is
accepted by the validator, and teacher-forced it predicts 99.2% of the record syntax
correctly. What it did not learn is the judgement inside the format. `scripts/diagnose.py`
feeds gold traces to the saved DAG model and scores the argmax at each position:

| position type | share of tokens | teacher-forced accuracy |
|---|---:|---:|
| record syntax | 0.977 | 0.992 |
| numbers inside `@prop` | 0.013 | 0.860 |
| critic mark | 0.004 | 0.713 |
| critic one-word comment | 0.004 | 0.948 |
| final answer | 0.001 | 1.000 |

The critic mark is the token that carries the comparison, and at 0.713 it is barely
above the 0.629 you get by always writing the majority mark. Given gold marks, the
model writes the right answer every time, so the shortfall is in the critic, not in
the summary. The numbers inside `@prop` include the random thresholds of off-topic
proposals, which no model could predict, so 0.860 understates how well it copies.
The direct model is close to the majority-class rate at two and three inequalities
(0.64 against 0.55, 0.73 against 0.67), so it also learned little.

Masking does what it is for and not more. At four inequalities, a length never seen
in training, free decoding produces traces the validator accepts only 9% of the time;
with the mask every trace is accepted. But the answers it leads to are no better than
the direct baseline (0.42 against 0.45), both well under the 0.84 majority rate, and
faithfulness stays at 0.09. In this implementation the validator certifies the
typed structure and nothing more. It does not check that a critic's mark is correct:
`@prop` is checked only for well-typedness, and the agreement between a mark and the
solver's verdict is measured after the fact by the scorer. Making the mask consult
the solver would force correct marks, but then the critic would no longer be doing
anything, so I left it out.

## Limitations

This does not show that structured traces are better or worse than chains of thought,
and the comparison here is confounded in ways that matter. DAG traces are more than ten
times longer than chain traces, the budget was fixed in training steps and not in
tokens or in tuning effort, there was one seed, and there are 64 test problems per
cell. My guess, which I have not tested, is that the critic's decision is a very
small fraction of the loss (0.4% of tokens) and is drowned out by hundreds of
deterministic record tokens; weighting decision tokens, longer training, or a
shorter record syntax are the obvious things to try. The task is synthetic and
shallow, the critic always follows its own proposal immediately, the proposition
solver covers a tiny fragment, and nothing here exercises the paper's claims about
slice topoi or limits.

## Running it

```
pip install -e ".[dev]"
pytest
python scripts/run_experiment.py --steps 2500 --batch 8 --eval-n 64
python scripts/diagnose.py
```

Training the DAG model is the slow part because its traces are long. `--reuse` skips
training and evaluates saved weights. Weights are written to `artifacts/`, which is
not tracked. The outputs behind the tables are kept in `results/`: the evaluation
JSON, the training log and the teacher-forced breakdown.

## Layout

```
dot_lm/
  tokens.py     character vocabulary, role tokens
  solver.py     decision procedure for @prop blocks
  validator.py  online validator, extraction map, decoder mask
  tasks.py      problem generator, gold traces, task-level scorer
  model.py      small transformer with KV cache
  decode.py     greedy decoding with optional validator masking
scripts/        run_experiment.py, diagnose.py
tests/          50 tests
results/        outputs of the run reported above
```

## Reference

*On the Diagram of Thought*, arXiv:2409.10038.
