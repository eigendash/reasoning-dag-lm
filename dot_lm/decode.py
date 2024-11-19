"""Greedy batched decoding, optionally masked by one online validator per sequence."""

import mlx.core as mx
import numpy as np

from .tokens import EOS, ITOS, STOI, VOCAB, tokenize
from .validator import OnlineValidator, feed


def generate(model, prompts, max_new, constrained=False):
    """Decode completions for prompts of equal token length.

    With constrained=True each step masks the logits to the tokens the validator
    would accept, so every finished output is a well-formed trace by construction.
    Returns a list of completions as token-string lists (ending in <eos> if finished).
    """
    batch, length = len(prompts), len(prompts[0])
    assert all(len(p) == length for p in prompts), "prompts in a batch must share a length"
    validators = None
    if constrained:
        validators = [feed(p, OnlineValidator()) for p in prompts]

    caches = model.new_caches(batch, length + max_new)
    ids = mx.array([[STOI[t] for t in p] for p in prompts])
    last = model(ids, caches)[:, -1]
    out = [[] for _ in range(batch)]
    done = [False] * batch

    for _ in range(max_new):
        if constrained:
            mask = np.zeros((batch, len(VOCAB)), dtype=bool)
            for i, v in enumerate(validators):
                allowed = v.allowed() if not done[i] else {EOS}
                for tok in allowed:
                    mask[i, STOI[tok]] = True
                if not allowed:
                    mask[i, STOI[EOS]] = True
            last = mx.where(mx.array(mask), last, -mx.inf)
        nxt = mx.argmax(last, axis=-1)
        picked = np.array(nxt)
        for i in range(batch):
            if done[i]:
                continue
            tok = ITOS[int(picked[i])]
            out[i].append(tok)
            if constrained and tok != EOS:
                validators[i].advance(tok)
            if tok == EOS:
                done[i] = True
        if all(done):
            break
        last = model(nxt[:, None], caches)[:, -1]
    return out


def prompt_tokens(text):
    return tokenize(text)
