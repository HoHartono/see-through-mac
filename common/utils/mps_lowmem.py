"""Run See-through within a 24 GB Mac's unified memory without swapping.

All changes keep the maths identical, they only work in smaller pieces:
- attention: MPS builds the whole attention matrix in fp32 (21 GB for one layer at 1280 px); split it
  over batch and heads (bit-identical), or over queries for one huge head (softmax is per query row)
- GroupNorm: MPS runs it in fp32; split over batch, or over channels at group boundaries
- feed-forward: GEGLU widens every token 8x; tokens are independent, so run them in slices
- MemPhase: prints the lowest free RAM and the swap written during a phase, to see what needs memory

Set ST_LOWMEM=0 to turn the chunking off (for comparing outputs), ST_CHUNK_MB to change the piece size.
"""
import os, subprocess, threading

import psutil
import torch
import torch.nn.functional as F

CHUNK_MB = int(os.environ.get("ST_CHUNK_MB", 512))

_sdpa = F.scaled_dot_product_attention
_group_norm = torch.nn.GroupNorm.forward
_feed_forward = None
_installed = False


def _attn_mb(query, key):
    return query.numel() // query.shape[-1] * key.shape[-2] * 4 / 1e6


def chunked_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kw):
    if _attn_mb(query, key) <= CHUNK_MB or is_causal or attn_mask is not None or query.ndim != 4:
        return _sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kw)
    out = torch.empty(*query.shape[:-1], value.shape[-1], dtype=query.dtype, device=query.device)
    _split_attn(query, key, value, out, 0, dropout_p, kw)
    return out


def _split_attn(query, key, value, out, dim, dropout_p, kw):
    n = query.shape[dim]
    per = _attn_mb(query, key) / n
    if per > CHUNK_MB and dim < 2:
        for s in range(n):   # one slice is still too big: split the next dim inside it
            _split_attn(query.narrow(dim, s, 1), key.narrow(dim, s, 1), value.narrow(dim, s, 1),
                        out.narrow(dim, s, 1), dim + 1, dropout_p, kw)
        return
    step = max(1, int(CHUNK_MB / per))
    for s in range(0, n, step):
        q = query.narrow(dim, s, min(step, n - s))
        k = key if dim == 2 else key.narrow(dim, s, q.shape[dim])
        v = value if dim == 2 else value.narrow(dim, s, q.shape[dim])
        out.narrow(dim, s, q.shape[dim]).copy_(_sdpa(q, k, v, dropout_p=dropout_p, **kw))


def chunked_group_norm(self, x):
    fp32_mb = x.numel() * 4 / 1e6
    if fp32_mb <= CHUNK_MB or x.ndim < 3:
        return _group_norm(self, x)
    if x.shape[0] > 1:
        step = max(1, int(x.shape[0] * CHUNK_MB / fp32_mb))
        return torch.cat([chunked_group_norm(self, c) for c in x.split(step, 0)], 0)
    g, per = self.num_groups, x.shape[1] // self.num_groups
    step = max(1, int(g * CHUNK_MB / fp32_mb))
    out = torch.empty_like(x)
    for s in range(0, g, step):
        n = min(step, g - s)
        c0, c1 = s * per, (s + n) * per
        w = self.weight[c0:c1] if self.affine else None
        b = self.bias[c0:c1] if self.affine else None
        out[:, c0:c1] = F.group_norm(x[:, c0:c1], n, w, b, self.eps)
    return out


def chunked_feed_forward(self, hidden_states, *args, **kwargs):
    wide_mb = hidden_states.numel() * 8 * hidden_states.element_size() / 1e6
    if wide_mb <= CHUNK_MB or hidden_states.ndim < 2:
        return _feed_forward(self, hidden_states, *args, **kwargs)
    step = max(1, int(hidden_states.shape[-2] * CHUNK_MB / wide_mb))
    return torch.cat([_feed_forward(self, c, *args, **kwargs) for c in hidden_states.split(step, -2)], -2)


def install():
    global _feed_forward, _installed
    if _installed or os.environ.get("ST_LOWMEM", "1") == "0":
        return
    from diffusers.models.attention import FeedForward
    _feed_forward = FeedForward.forward
    F.scaled_dot_product_attention = chunked_sdpa
    torch.nn.GroupNorm.forward = chunked_group_norm
    FeedForward.forward = chunked_feed_forward
    _installed = True


def swapouts_mb():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith("Swapouts"):
            return int(line.split(":")[1].strip().rstrip(".")) * 16384 / 1e6
    return 0.0


class MemPhase:
    """with MemPhase('load unet'): ...  -> [mem] load unet: lowest free RAM 9.1 GB, swap written 0 MB"""

    def __init__(self, label, interval=0.5):
        self.label, self.interval = label, interval
        self._stop = threading.Event()

    def _run(self):
        while not self._stop.wait(self.interval):
            self.low = min(self.low, psutil.virtual_memory().available)

    def __enter__(self):
        self.swap0, self.low = swapouts_mb(), psutil.virtual_memory().available
        threading.Thread(target=self._run, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self.low = min(self.low, psutil.virtual_memory().available)
        print(f"[mem] {self.label}: lowest free RAM {self.low / 1e9:.1f} GB, "
              f"swap written {swapouts_mb() - self.swap0:.0f} MB", flush=True)
