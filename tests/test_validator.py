import numpy as np
import pytest

from dot_lm.tokens import tokenize
from dot_lm.validator import OnlineValidator, Violation, is_valid, validate

PAPER_STYLE = (
    "<problem>@node id=1 role=problem\n(04,06) in x>02, y>05?\n"
    "<proposer>@node id=2 role=proposer\n@edge src=1 dst=2 kind=use\n"
    '@prop id=2 {"gt":[4,2]}\nx>02: 04>02\n'
    "<critic>@node id=3 role=critic\n@edge src=2 dst=3 kind=critique\n"
    "@status target=2 mark=validated\nok\n"
    "<proposer>@node id=4 role=proposer\n@edge src=1 dst=4 kind=use\n"
    '@prop id=4 {"gt":[6,0]}\ny>00: 06>00\n'
    "<critic>@node id=5 role=critic\n@edge src=4 dst=5 kind=critique\n"
    "@status target=4 mark=invalidated\noff-topic\n"
    "<proposer>@node id=6 role=proposer\n@edge src=1 dst=6 kind=use\n@edge src=5 dst=6 kind=refine\n"
    '@prop id=6 {"gt":[6,5]}\ny>05: 06>05\n'
    "<critic>@node id=7 role=critic\n@edge src=6 dst=7 kind=critique\n"
    "@status target=6 mark=validated\nok\n"
    "<summarizer>@node id=8 role=summarizer\n@edge src=2 dst=8 kind=use\n@edge src=6 dst=8 kind=use\nyes\n<eos>"
)


def trace(text=PAPER_STYLE):
    return tokenize(text)


def test_a_well_formed_trace_is_accepted_and_extracted():
    d = validate(trace())
    assert [n.role for n in d.nodes.values()] == [
        "problem", "proposer", "critic", "proposer", "critic", "proposer", "critic", "summarizer"
    ]
    assert d.nodes[2].state == "validated" and d.nodes[4].state == "invalidated"
    assert d.selected(8) == [2, 6]  # the invalidated proposal is not selectable
    assert d.answer() == "yes"
    assert (5, 6, "refine") in d.edges and (2, 3, "critique") in d.edges
    assert "digraph" in d.to_dot()


def test_every_prefix_of_a_good_trace_stays_allowed_token_by_token():
    v = OnlineValidator()
    for token in trace():
        assert token in v.allowed()
        v.advance(token)
    assert v.phase == "end" and v.allowed() == set()


def corrupt(old, new, count=1):
    assert old in PAPER_STYLE
    return trace(PAPER_STYLE.replace(old, new, count))


@pytest.mark.parametrize(
    "bad",
    [
        corrupt("@node id=3 role=critic", "@node id=4 role=critic"),  # ids must be fresh and consecutive
        corrupt("@edge src=2 dst=3 kind=critique", "@edge src=2 dst=9 kind=critique"),  # edge must end at current node
        corrupt("@edge src=2 dst=3 kind=critique", "@edge src=1 dst=3 kind=critique"),  # critique must start at a proposer
        corrupt("@edge src=1 dst=2 kind=use", "@edge src=3 dst=2 kind=use"),  # source not yet emitted
        corrupt("@status target=2 mark=validated", "@status target=4 mark=validated"),  # not the critiqued target
        corrupt("@edge src=6 dst=8 kind=use", "@edge src=4 dst=8 kind=use"),  # summarizer cites an invalidated proposal
        corrupt('{"gt":[4,2]}', '{"gt":[4]}'),  # typed block does not type-check
        corrupt("x>02: 04>02", "@x>02"),  # free text may not start with @
        corrupt("<critic>@node id=3 role=critic", "<critic>@node id=3 role=proposer"),  # token and record disagree
    ],
)
def test_ill_formed_traces_are_rejected(bad):
    assert not is_valid(bad)


def test_unterminated_and_post_eos_traces_are_rejected():
    assert not is_valid(trace()[:-1])
    assert not is_valid(trace() + ["\n"])


def test_summarizer_needs_a_validated_proposal_and_no_open_critique():
    v = OnlineValidator()
    for t in tokenize(PAPER_STYLE.split("<proposer>")[0]):
        v.advance(t)
    assert v.allowed() == {"<proposer>"}  # nothing to critique, nothing to summarise
    for t in tokenize("<proposer>@node id=2 role=proposer\n@edge src=1 dst=2 kind=use\n"):
        v.advance(t)
    for t in tokenize('@prop id=2 {"gt":[4,2]}\nx>02: 04>02\n'):
        v.advance(t)
    assert v.allowed() == {"<proposer>", "<critic>"}  # the open proposal must be critiqued first


def test_status_is_first_writer_wins():
    v = OnlineValidator()
    for t in trace():
        v.advance(t)
    # node 2 is already validated, so a later critic cannot critique it again
    v2 = OnlineValidator()
    prefix = PAPER_STYLE.split("<proposer>@node id=4")[0]
    for t in tokenize(prefix + "<proposer>@node id=4 role=proposer\n@edge src=1 dst=4 kind=use\n"
                      '@prop id=4 {"gt":[6,5]}\nx\n<critic>@node id=5 role=critic\n'):
        v2.advance(t)
    allowed_edges = {c for c in v2._cands}
    assert "@edge src=2 dst=5 kind=critique" not in allowed_edges
    assert "@edge src=4 dst=5 kind=critique" in allowed_edges


PROP = '{"gt":[4,2]}'


def steer_prop(v, options):
    """Random characters never form a valid JSON proposition, so finish it by hand."""
    head = f"@prop id={v.cur} "
    if v.phase == "line" and v.buf.startswith(head):
        body = v.buf[len(head):]
        if body == PROP:
            return ["\n"]
        if PROP.startswith(body):
            return [PROP[len(body)]]
    return options


def test_random_masked_walks_always_produce_accepted_traces():
    rng = np.random.default_rng(0)
    finished = 0
    for _ in range(60):
        v, out = OnlineValidator(), []
        for _step in range(3000):
            options = sorted(v.allowed())
            if not options:
                break
            # steer toward termination: end lines early, and finish once a summary is possible
            for special in ("<eos>", "<summarizer>"):
                if special in options and rng.random() < 0.7:
                    options = [special]
            if "\n" in options and rng.random() < 0.3:
                options = ["\n"]
            options = steer_prop(v, options)
            token = options[int(rng.integers(0, len(options)))]
            out.append(token)
            v.advance(token)
        if v.phase == "end":
            finished += 1
            assert is_valid(out)  # whatever the mask lets through, the checker accepts
    assert finished >= 30  # the walks really do reach the end, so the assertion above ran
