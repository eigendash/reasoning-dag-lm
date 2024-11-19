import mlx.core as mx
import numpy as np

from dot_lm.decode import generate
from dot_lm.model import CharLM, Config
from dot_lm.tasks import dot_text, prompt_text, sample_problem
from dot_lm.tokens import EOS, STOI, VOCAB, tokenize
from dot_lm.validator import is_valid, validate


def tiny_model():
    mx.random.seed(0)
    m = CharLM(Config(dim=32, heads=2, layers=2, ffn=64))
    mx.eval(m.parameters())
    return m


def test_cached_decoding_matches_a_full_forward_pass():
    model = tiny_model()
    ids = mx.random.randint(0, len(VOCAB), (3, 40))
    full = model(ids)
    caches = model.new_caches(3, 40)
    head = model(ids[:, :25], caches)
    steps = [model(ids[:, i : i + 1], caches) for i in range(25, 40)]
    stitched = mx.concatenate([head] + steps, axis=1)
    np.testing.assert_allclose(np.array(stitched), np.array(full), atol=1e-4)


class ScriptedModel:
    """Stands in for a language model that writes a fixed trace but is tempted by one bad token."""

    def __init__(self, script, tempt_at=None, tempt=None):
        self.script, self.tempt_at, self.tempt, self.pos = [STOI[t] for t in script], tempt_at, tempt, 0

    def new_caches(self, batch, length):
        self.pos = 0
        return [None]

    def __call__(self, ids, caches):
        self.pos += ids.shape[1]
        logits = np.zeros((ids.shape[0], 1, len(VOCAB)), dtype=np.float32)
        if self.pos < len(self.script):
            logits[:, 0, self.script[self.pos]] = 10.0
            if self.pos == self.tempt_at:
                logits[:, 0, STOI[self.tempt]] = 20.0
        else:
            logits[:, 0, STOI[EOS]] = 10.0
        return mx.array(logits)


def scripted_case():
    rng = np.random.default_rng(0)
    p = sample_problem(rng, 3)
    text = dot_text(p, rng, distractor_prob=0.0)
    tokens = tokenize(text)
    prompt = tokenize(prompt_text(p, "dot"))
    first_critic = tokens.index("<critic>")
    return tokens, prompt, first_critic


def test_unconstrained_decoding_follows_the_temptation_and_breaks_the_trace():
    tokens, prompt, at = scripted_case()
    model = ScriptedModel(tokens, tempt_at=at, tempt="<summarizer>")
    (out,) = generate(model, [prompt], max_new=len(tokens), constrained=False)
    assert out[at - len(prompt)] == "<summarizer>"
    assert not is_valid(prompt + out)


def test_constrained_decoding_refuses_the_temptation_and_returns_a_valid_trace():
    tokens, prompt, at = scripted_case()
    model = ScriptedModel(tokens, tempt_at=at, tempt="<summarizer>")
    (out,) = generate(model, [prompt], max_new=len(tokens) + 5, constrained=True)
    assert out[at - len(prompt)] == "<critic>"
    assert out[-1] == EOS and prompt + out == tokens
    assert validate(prompt + out).answer() in ("yes", "no")


def test_constrained_decoding_with_an_untrained_model_never_leaves_the_grammar():
    from dot_lm.validator import OnlineValidator, feed

    model = tiny_model()
    p = sample_problem(np.random.default_rng(1), 2)
    prompt = tokenize(prompt_text(p, "dot"))
    outs = generate(model, [prompt] * 4, max_new=300, constrained=True)
    for out in outs:
        v = feed(prompt, OnlineValidator())
        for tok in out:
            if tok == EOS:
                break
            assert tok in v.allowed()  # replaying the output never trips a rule
            v.advance(tok)
