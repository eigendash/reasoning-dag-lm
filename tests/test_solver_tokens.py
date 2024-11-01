import pytest

from dot_lm import solver
from dot_lm.tokens import EOS, STOI, decode, detokenize, encode, tokenize


@pytest.mark.parametrize(
    "text,truth",
    [
        ('{"gt":[4,2]}', True),
        ('{"gt":[2,4]}', False),
        ('{"le":[3,3]}', True),
        ('{"ne":[3,3]}', False),
        ('{"eq":[{"add":[3,4]},7]}', True),
        ('{"lt":[{"mul":[2,{"sub":[9,4]}]},9]}', False),
        ('{"gt":[-1,-2]}', True),
    ],
)
def test_solver_evaluates_comparisons_and_arithmetic(text, truth):
    assert solver.evaluate(solver.parse_prop(text)) is truth


@pytest.mark.parametrize(
    "text",
    ["", "{", '{"gt":[1]}', '{"gt":[1,2,3]}', '{"xx":[1,2]}', '{"gt":[1,2],"lt":[1,2]}',
     '{"gt":[true,2]}', '{"gt":["a",2]}', '{"gt":[{"pow":[2,2]},2]}', "[1,2]", "4"],
)
def test_solver_rejects_malformed_propositions(text):
    with pytest.raises(solver.PropError):
        solver.parse_prop(text)


def test_solver_depth_is_bounded():
    deep = 1
    for _ in range(6):
        deep = {"add": [deep, 1]}
    with pytest.raises(solver.PropError):
        solver.evaluate({"gt": [deep, 0]})


def test_tokenizer_round_trips_and_keeps_role_tokens_whole():
    text = "<critic>@node id=3 role=critic\n@status target=2 mark=validated\nok\n<eos>"
    tokens = tokenize(text)
    assert tokens[0] == "<critic>" and tokens[-1] == EOS
    assert detokenize(tokens) == text
    assert decode(encode(text)) == text
    assert all(t in STOI for t in tokens)


def test_comparison_characters_are_not_mistaken_for_specials():
    assert tokenize("x<10, y>5") == list("x<10, y>5")


def test_unknown_characters_are_rejected():
    with pytest.raises(ValueError):
        tokenize("Q?")
