"""A small decoder-only transformer over the character vocabulary."""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .tokens import VOCAB


@dataclass
class Config:
    vocab: int = len(VOCAB)
    dim: int = 128
    heads: int = 4
    layers: int = 4
    ffn: int = 384


class KVCache:
    """Preallocated key/value buffers, filled in place as decoding proceeds."""

    def __init__(self, batch, heads, length, head_dim):
        self.k = mx.zeros((batch, heads, length, head_dim))
        self.v = mx.zeros((batch, heads, length, head_dim))
        self.offset = 0

    def update(self, k, v):
        end = self.offset + k.shape[2]
        self.k[:, :, self.offset:end, :] = k
        self.v[:, :, self.offset:end, :] = v
        self.offset = end
        return self.k[:, :, :end, :], self.v[:, :, :end, :]


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.rope = nn.RoPE(dim // heads)

    def __call__(self, x, cache=None):
        B, T, D = x.shape
        q, k, v = (t.reshape(B, T, self.heads, -1).transpose(0, 2, 1, 3) for t in mx.split(self.qkv(x), 3, axis=-1))
        offset = cache.offset if cache is not None else 0
        q, k = self.rope(q, offset=offset), self.rope(k, offset=offset)
        if cache is not None:
            k, v = cache.update(k, v)
        mask = "causal" if T > 1 else None
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=q.shape[-1] ** -0.5, mask=mask)
        return self.out(o.transpose(0, 2, 1, 3).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.dim)
        self.attn = Attention(cfg.dim, cfg.heads)
        self.norm2 = nn.RMSNorm(cfg.dim)
        self.gate = nn.Linear(cfg.dim, cfg.ffn, bias=False)
        self.up = nn.Linear(cfg.dim, cfg.ffn, bias=False)
        self.down = nn.Linear(cfg.ffn, cfg.dim, bias=False)

    def __call__(self, x, cache=None):
        x = x + self.attn(self.norm1(x), cache)
        h = self.norm2(x)
        return x + self.down(nn.silu(self.gate(h)) * self.up(h))


class CharLM(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or Config()
        self.embed = nn.Embedding(self.cfg.vocab, self.cfg.dim)
        self.blocks = [Block(self.cfg) for _ in range(self.cfg.layers)]
        self.norm = nn.RMSNorm(self.cfg.dim)
        self.head = nn.Linear(self.cfg.dim, self.cfg.vocab, bias=False)

    def __call__(self, tokens, caches=None):
        x = self.embed(tokens)
        for i, block in enumerate(self.blocks):
            x = block(x, caches[i] if caches else None)
        return self.head(self.norm(x))

    def new_caches(self, batch, length):
        c = self.cfg
        return [KVCache(batch, c.heads, length, c.dim // c.heads) for _ in range(c.layers)]
