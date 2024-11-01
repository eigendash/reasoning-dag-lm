"""Character-level vocabulary with four role tokens and an end token."""

ROLES = ("problem", "proposer", "critic", "summarizer")
ROLE_TOKEN = {role: f"<{role}>" for role in ROLES}
ROLE_OF = {tok: role for role, tok in ROLE_TOKEN.items()}
EOS = "<eos>"
PAD = "<pad>"

SPECIALS = [PAD, EOS, *ROLE_TOKEN.values()]
CHARS = list("\n 0123456789abcdefghijklmnopqrstuvwxyz()[]{},.:?<>=@\"-+*_/")
VOCAB = SPECIALS + CHARS
STOI = {tok: i for i, tok in enumerate(VOCAB)}
ITOS = dict(enumerate(VOCAB))

_MULTI = sorted([EOS, *ROLE_TOKEN.values()], key=len, reverse=True)


def tokenize(text):
    """Split text into single characters and the multi-character special tokens."""
    out, i = [], 0
    while i < len(text):
        for special in _MULTI:
            if text.startswith(special, i):
                out.append(special)
                i += len(special)
                break
        else:
            if text[i] not in STOI:
                raise ValueError(f"character {text[i]!r} is not in the vocabulary")
            out.append(text[i])
            i += 1
    return out


def detokenize(tokens):
    return "".join(t for t in tokens if t != PAD)


def encode(text):
    return [STOI[t] for t in tokenize(text)]


def decode(ids):
    return detokenize([ITOS[int(i)] for i in ids])
