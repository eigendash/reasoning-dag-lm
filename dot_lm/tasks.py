"""A toy reasoning task with programmatic gold traces.

Problem: is the point (x, y) inside a region given by two to four inequalities?

    (04,06) in x>02, y>05, x<10?

Three trace formats are generated from the same problem so the models can be
compared on equal footing:

  dot     the typed reasoning DAG (propose -> critique -> repair -> summarize)
  chain   a linear list of checks followed by the answer
  direct  the answer alone

In the DAG, each inequality is first *claimed* as written, a critic judges the claim
against the solver, and a false claim is invalidated and then repaired with its
negation. Occasionally a proposer wanders off to a condition that is not in the
problem; the critic invalidates that as off-topic. The summarizer draws `use` edges
from the validated propositions only.
"""

from dataclasses import dataclass

from . import solver
from .tokens import ROLE_TOKEN, EOS

NEGATE = {">": "<=", "<": ">="}
GT_LT = {">": "gt", "<": "lt"}
NEG_NAME = {">": "le", "<": "ge"}
CANDIDATE_CONDITIONS = [("x", ">"), ("x", "<"), ("y", ">"), ("y", "<")]


@dataclass(frozen=True)
class Cond:
    var: str
    op: str
    t: int

    def text(self):
        return f"{self.var}{self.op}{self.t:02d}"


@dataclass(frozen=True)
class Problem:
    x: int
    y: int
    conds: tuple

    def value(self, var):
        return self.x if var == "x" else self.y

    def holds(self, cond):
        v = self.value(cond.var)
        return v > cond.t if cond.op == ">" else v < cond.t

    def answer(self):
        return all(self.holds(c) for c in self.conds)

    def text(self):
        return f"({self.x:02d},{self.y:02d}) in " + ", ".join(c.text() for c in self.conds) + "?"


def sample_problem(rng, n_conds):
    x, y = int(rng.integers(0, 20)), int(rng.integers(0, 20))
    picks = rng.permutation(len(CANDIDATE_CONDITIONS))[:n_conds]
    conds = []
    for i in sorted(picks):
        var, op = CANDIDATE_CONDITIONS[i]
        v = x if var == "x" else y
        want_true = rng.random() < 0.75
        gap = int(rng.integers(1, 7))
        t = v - gap if (op == ">") == want_true else v + gap
        conds.append(Cond(var, op, int(min(19, max(0, t)))))
    return Problem(x, y, tuple(conds))


def claim(problem, cond):
    """The proposition a proposer states for a condition, exactly as written."""
    return {GT_LT[cond.op]: [problem.value(cond.var), cond.t]}


def repair(problem, cond):
    return {NEG_NAME[cond.op]: [problem.value(cond.var), cond.t]}


def relevant(problem, prop):
    """Does the proposition restate a condition of the problem, or its negation?"""
    (name, (lhs, rhs)), = prop.items()
    for c in problem.conds:
        v = problem.value(c.var)
        if (lhs, rhs) == (v, c.t) and name in (GT_LT[c.op], NEG_NAME[c.op]):
            return True
    return False


def judge(problem, prop):
    """The mark a correct critic gives: true to the solver and on topic."""
    return "validated" if solver.evaluate(prop) and relevant(problem, prop) else "invalidated"


def _json(prop):
    import json

    return json.dumps(prop, separators=(",", ":"))


class _Builder:
    def __init__(self, problem):
        self.problem, self.n, self.parts, self.validated = problem, 0, [], []
        self.parts.append(f"{ROLE_TOKEN['problem']}@node id=1 role=problem\n{problem.text()}\n")
        self.n = 1

    def _text(self, var, op, v, t):
        return f"{var}{op}{t:02d}: {v:02d}{op}{t:02d}"

    def proposer(self, prop, text, edges):
        self.n += 1
        lines = [f"{ROLE_TOKEN['proposer']}@node id={self.n} role=proposer"]
        lines += [f"@edge src={s} dst={self.n} kind={k}" for s, k in edges]
        lines += [f"@prop id={self.n} {_json(prop)}", text]
        self.parts.append("\n".join(lines) + "\n")
        return self.n

    def critic(self, target, mark, word):
        self.n += 1
        self.parts.append(
            f"{ROLE_TOKEN['critic']}@node id={self.n} role=critic\n@edge src={target} dst={self.n} kind=critique\n"
            f"@status target={target} mark={mark}\n{word}\n"
        )
        return self.n


def dot_text(problem, rng, distractor_prob=0.3):
    b = _Builder(problem)
    for cond in problem.conds:
        if rng.random() < distractor_prob:
            _distractor(b, problem, cond, rng)
        v = problem.value(cond.var)
        pid = b.proposer(claim(problem, cond), b._text(cond.var, cond.op, v, cond.t), [(1, "use")])
        if problem.holds(cond):
            b.critic(pid, "validated", "ok")
            b.validated.append(pid)
        else:
            cid = b.critic(pid, "invalidated", "wrong")
            neg = NEGATE[cond.op]
            rid = b.proposer(repair(problem, cond), b._text(cond.var, neg, v, cond.t), [(1, "use"), (cid, "refine")])
            b.critic(rid, "validated", "ok")
            b.validated.append(rid)
    b.n += 1
    edges = "".join(f"@edge src={s} dst={b.n} kind=use\n" for s in b.validated)
    answer = "yes" if problem.answer() else "no"
    b.parts.append(f"{ROLE_TOKEN['summarizer']}@node id={b.n} role=summarizer\n{edges}{answer}\n{EOS}")
    return "".join(b.parts)


def _distractor(b, problem, near, rng):
    """A true but off-topic claim about a threshold the problem never mentions."""
    for _ in range(20):
        var, op = CANDIDATE_CONDITIONS[int(rng.integers(0, 4))]
        t = int(rng.integers(0, 20))
        cand = Cond(var, op, t)
        prop = claim(problem, cand)
        if solver.evaluate(prop) and not relevant(problem, prop):
            v = problem.value(var)
            pid = b.proposer(prop, b._text(var, op, v, t), [(1, "use")])
            b.critic(pid, "invalidated", "off-topic")
            return


def chain_text(problem):
    lines = [f"{ROLE_TOKEN['problem']}{problem.text()}\n"]
    for c in problem.conds:
        v = problem.value(c.var)
        lines.append(f"{c.text()}: {v:02d}{c.op}{c.t:02d} {'ok' if problem.holds(c) else 'no'}\n")
    lines.append(f"{'yes' if problem.answer() else 'no'}\n{EOS}")
    return "".join(lines)


def direct_text(problem):
    return f"{ROLE_TOKEN['problem']}{problem.text()}\n{'yes' if problem.answer() else 'no'}\n{EOS}"


def prompt_text(problem, fmt):
    if fmt == "dot":
        return f"{ROLE_TOKEN['problem']}@node id=1 role=problem\n{problem.text()}\n"
    return f"{ROLE_TOKEN['problem']}{problem.text()}\n"


def full_text(problem, fmt, rng, distractor_prob=0.3):
    if fmt == "dot":
        return dot_text(problem, rng, distractor_prob)
    return {"chain": chain_text, "direct": direct_text}[fmt](problem)


def last_line(text):
    """Final non-empty line of a completion, ignoring the end token."""
    lines = [ln for ln in text.replace(EOS, "").split("\n") if ln.strip()]
    return lines[-1].strip() if lines else ""


def score_dot(problem, diagram):
    """Task-level checks on an accepted diagram.

    faithful: every critic mark equals what a correct critic would write
    sound:    every validated proposition is actually true
    correct:  the summarizer's answer matches the ground truth
    """
    faithful = all(judge(problem, diagram.nodes[t].prop) == m for t, m, _ in diagram.statuses)
    sound = all(
        solver.evaluate(n.prop) for n in diagram.nodes.values() if n.role == "proposer" and n.state == "validated"
    )
    expected = "yes" if problem.answer() else "no"
    return {"faithful": faithful, "sound": sound, "correct": diagram.answer() == expected}
