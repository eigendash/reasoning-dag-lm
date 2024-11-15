import numpy as np
import pytest

from dot_lm.tasks import (
    Cond, Problem, chain_text, direct_text, dot_text, judge, last_line, prompt_text, relevant,
    sample_problem, score_dot,
)
from dot_lm.tokens import tokenize
from dot_lm.validator import validate


def problems(n=200, conds=(2, 3, 4), seed=0):
    rng = np.random.default_rng(seed)
    return [sample_problem(rng, conds[i % len(conds)]) for i in range(n)]


def test_problem_semantics():
    p = Problem(4, 6, (Cond("x", ">", 2), Cond("y", ">", 5), Cond("x", "<", 10)))
    assert p.answer() and p.text() == "(04,06) in x>02, y>05, x<10?"
    assert not Problem(4, 6, (Cond("x", ">", 4),)).answer()


def test_sampler_is_balanced_enough_to_learn_from():
    ps = problems(600, conds=(2, 3))
    rate = np.mean([p.answer() for p in ps])
    assert 0.3 < rate < 0.7


def test_every_gold_trace_validates_and_scores_perfectly():
    rng = np.random.default_rng(1)
    for p in problems(300):
        d = validate(tokenize(dot_text(p, rng)))
        assert all(score_dot(p, d).values())


def test_gold_trace_repairs_every_false_claim_and_summarises_only_validated_nodes():
    rng = np.random.default_rng(2)
    seen_repair = seen_distractor = False
    for p in problems(200):
        d = validate(tokenize(dot_text(p, rng, distractor_prob=0.5)))
        invalid = [n for n in d.nodes.values() if n.state == "invalidated"]
        refine = [e for e in d.edges if e[2] == "refine"]
        false_conds = sum(not p.holds(c) for c in p.conds)
        assert len(refine) == false_conds
        seen_repair |= false_conds > 0
        seen_distractor |= len(invalid) > false_conds
        (summary,) = d.summaries()
        assert all(d.nodes[s].state == "validated" for s in d.selected(summary.id))
        assert len(d.selected(summary.id)) == len(p.conds)
    assert seen_repair and seen_distractor


def test_judge_follows_truth_and_relevance():
    p = Problem(4, 6, (Cond("x", ">", 2),))
    assert judge(p, {"gt": [4, 2]}) == "validated"
    assert judge(p, {"gt": [4, 9]}) == "invalidated"  # false
    assert judge(p, {"gt": [6, 0]}) == "invalidated"  # true but off-topic
    assert relevant(p, {"le": [4, 2]}) and not relevant(p, {"lt": [4, 2]})


def test_chain_and_direct_agree_with_ground_truth():
    for p in problems(100):
        for text in (chain_text(p), direct_text(p)):
            assert last_line(text) == ("yes" if p.answer() else "no")
            tokenize(text)  # everything is in the vocabulary


def test_prompt_is_a_prefix_of_the_full_trace():
    rng = np.random.default_rng(3)
    p = problems(1)[0]
    assert dot_text(p, rng).startswith(prompt_text(p, "dot"))
    assert chain_text(p).startswith(prompt_text(p, "chain"))


def test_prompt_length_depends_only_on_the_number_of_conditions():
    for k in (2, 3, 4):
        lengths = {len(tokenize(prompt_text(p, "dot"))) for p in problems(60, conds=(k,))}
        assert len(lengths) == 1


def test_grammar_cannot_see_a_wrongly_validated_claim_but_the_task_scorer_can():
    rng = np.random.default_rng(4)
    checked = 0
    for p in problems(100, conds=(2, 3)):
        if p.answer():
            continue
        text = dot_text(p, rng, distractor_prob=0.0)
        assert "mark=invalidated" in text
        # a critic that waves a false claim through
        flipped = text.replace("mark=invalidated", "mark=validated", 1).replace("wrong", "ok", 1)
        diagram = validate(tokenize(flipped))  # the typed grammar accepts it
        result = score_dot(p, diagram)
        assert not result["faithful"] and not result["sound"]
        checked += 1
    assert checked > 10
