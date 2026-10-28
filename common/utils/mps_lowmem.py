"""Run See-through within a 24 GB Mac's unified memory without swapping.

All changes keep the maths identical, they only work in smaller pieces:
- attention: MPS builds the whole attention matrix in fp32 (21 GB for one layer at 1280 px); split it
  over batch and heads (bit-identical), or over queries for one huge head (softmax is per query row)
- GroupNorm: MPS runs it in fp32; split over batch, or over channels at group boundaries
- feed-forward: GEGLU widens every token 8x; tokens are independent, so run them in slices
- loading: from_pretrained reads all weights into RAM, then .to('mps') copies them; on unified memory both
  copies share the same RAM (~15 GB peak for the 10 GB LayerDiff). load_to_device reads one tensor at a time
  with F_NOCACHE, so neither a RAM copy nor macOS's file cache of the 8 GB file builds up
- free memory: psutil's "available" is free + inactive, and inactive includes other apps' idle memory that
  macOS can only reclaim by swapping it out. free_gb() counts free pages plus file cache macOS can drop
- wait_for_free_ram: wait before loading until the weights fit in free memory, and say what is short
- make_room: turn droppable file cache into free pages slowly, so the fast load doesn't make macOS swap
- start_swap_guard: stops the run if swap grows past a limit, so a run can't keep writing to the SSD
- MemPhase: prints the lowest free memory and how much swap grew during a phase, to see what needs memory

Set ST_LOWMEM=0 to turn the chunking off (for comparing outputs), ST_CHUNK_MB to change the piece size.
"""
import fcntl, json, os, re, struct, subprocess, threading, time

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


F_NOCACHE = 48   # macOS fcntl: reads bypass the file cache
_ST_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


def read_safetensors_nocache(path):
    """yield (name, cpu tensor) one at a time, reading with F_NOCACHE; each tensor is a fresh small buffer"""
    fd = os.open(path, os.O_RDONLY)
    try:
        fcntl.fcntl(fd, F_NOCACHE, 1)
        (n,) = struct.unpack("<Q", os.pread(fd, 8, 0))
        header = json.loads(os.pread(fd, n, 8))
        header.pop("__metadata__", None)
        for name, info in sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0]):
            start, end = info["data_offsets"]
            buf = bytearray(end - start)
            view, got = memoryview(buf), 0
            while got < len(buf):
                got += os.preadv(fd, [view[got:]], 8 + n + start + got)
            dtype = _ST_DTYPES[info["dtype"]]
            yield name, (torch.frombuffer(buf, dtype=dtype) if buf else torch.empty(0, dtype=dtype)).view(info["shape"])
    finally:
        os.close(fd)


def load_to_device(cls, pretrained, subfolder=None, device="mps", dtype=torch.bfloat16):
    """cls.from_pretrained(...).to(device, dtype), but never holding a second copy of the weights"""
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device

    folder = os.path.join(pretrained, subfolder) if subfolder else pretrained
    with init_empty_weights(include_buffers=False):
        model = cls.from_config(cls.load_config(folder))
    expected, seen = set(model.state_dict().keys()), set()
    for name, tensor in read_safetensors_nocache(os.path.join(folder, "diffusion_pytorch_model.safetensors")):
        if name not in expected:
            raise RuntimeError(f"{cls.__name__}: unexpected weight {name}")
        # dtype must be passed: without it accelerate casts to the empty model's float32 (double memory)
        set_module_tensor_to_device(model, name, device, value=tensor.to(device, dtype), dtype=dtype)
        seen.add(name)
    if seen != expected:
        raise RuntimeError(f"{cls.__name__}: missing weights {sorted(expected - seen)[:5]}")
    return model.to(device).eval()   # moves the few buffers created outside the weights file


# file cache macOS keeps for files in use (app code etc.); measured: it dropped the cache down to ~5.2 GB
# without swapping, then started compressing app memory
CACHE_FLOOR_GB = 5.5


def _vm_gb():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+)", out).group(1))
    return {m.group(1): int(m.group(2)) * page / 1e9 for m in re.finditer(r'^"?([^":]+)"?:\s+(\d+)\.', out, re.M)}


def pages_free_gb():
    """free right now, with nothing to drop first"""
    v = _vm_gb()
    return v["Pages free"] + v["Pages speculative"] + v["Pages purgeable"]


def free_gb():
    """memory macOS hands out without swapping: free + speculative + purgeable pages, plus the file cache
    above CACHE_FLOOR_GB (clean copies of files; dropping them writes nothing). psutil's "available" also
    counts other apps' inactive memory, which can only be freed by swapping it out."""
    v = _vm_gb()
    free = v["Pages free"] + v["Pages speculative"] + v["Pages purgeable"]
    return free + max(0.0, v["File-backed pages"] - CACHE_FLOOR_GB)


def make_room(need_gb, step_gb=0.5, pause_s=0.15, max_swap_mb=50):
    """Turn droppable file cache into free pages before a fast load.
    Loading reads at several GB/s; when that has to come out of the file cache, macOS can't drop the cache
    fast enough and swaps a little instead (251 MB in a 640 px test). Taking the memory in 512 MB steps
    first gave 12.5 GB with zero swap, and releasing it left it all free for the load."""
    if pages_free_gb() >= need_gb:
        return
    t0, swap0, held = time.time(), swap_used_mb(), []
    try:
        while len(held) * step_gb + pages_free_gb() < need_gb:
            if swap_used_mb() - swap0 > max_swap_mb:
                print("[mem] make room: macOS started swapping, stopping early", flush=True)
                break
            held.append(torch.ones(int(step_gb * 2**30) // 2, dtype=torch.bfloat16, device="mps"))
            torch.mps.synchronize()
            time.sleep(pause_s)
    finally:
        del held
        torch.mps.empty_cache()
    print(f"[mem] make room: {pages_free_gb():.1f} GB free for loading ({time.time() - t0:.0f}s)", flush=True)


def wait_for_free_ram(need_gb, what, timeout_s=300):
    """wait until need_gb is free; after timeout_s carry on anyway (the SSD guard still watches)"""
    t0, said = time.time(), 0
    while (free := free_gb()) < need_gb:
        if time.time() - t0 > timeout_s:
            print(f"[mem] still only {free:.1f} GB free after {timeout_s}s; loading {what} anyway", flush=True)
            return
        if time.time() - said > 30:
            print(f"[mem] {what} needs {need_gb:.1f} GB free memory, {free:.1f} GB is free: "
                  f"close apps (browser, Godot, ComfyUI) and it continues by itself", flush=True)
            said = time.time()
        time.sleep(2)


def swap_used_mb():
    """MB in the swap files now. vm_stat's Swapouts counter overstates what reaches the SSD."""
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    return float(re.search(r"used = ([\d.]+)M", out).group(1))


def start_swap_guard(limit_mb=1500, interval_s=2):
    """stop the process when the swap files grow by more than limit_mb (SSD writes); 0 turns it off"""
    if not limit_mb:
        return
    start = swap_used_mb()

    def watch():
        while True:
            time.sleep(interval_s)
            grew = swap_used_mb() - start
            if grew > limit_mb:
                print(f"[swap-guard] STOPPED: swap grew {grew:.0f} MB. Close apps, or lower --resolution.", flush=True)
                os._exit(3)
    threading.Thread(target=watch, daemon=True).start()


class MemPhase:
    """with MemPhase('load unet'): ...  -> [mem] load unet: lowest free 4.2 GB, swap grew 0 MB, disk writes 12 MB"""

    def __init__(self, label, interval=0.5):
        self.label, self.interval = label, interval
        self._stop = threading.Event()

    def _run(self):
        while not self._stop.wait(self.interval):
            self.low = min(self.low, free_gb())
            self.swap_peak = max(self.swap_peak, swap_used_mb())

    def __enter__(self):
        self.swap0 = self.swap_peak = swap_used_mb()
        self.low, self.w0 = free_gb(), psutil.disk_io_counters().write_bytes
        threading.Thread(target=self._run, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self.low = min(self.low, free_gb())
        self.swap_peak = max(self.swap_peak, swap_used_mb())
        print(f"[mem] {self.label}: lowest free {self.low:.1f} GB, "
              f"swap grew {max(0, self.swap_peak - self.swap0):.0f} MB, "
              f"disk writes {(psutil.disk_io_counters().write_bytes - self.w0) / 1e6:.0f} MB", flush=True)
