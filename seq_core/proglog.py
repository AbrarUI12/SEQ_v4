"""Step-level progress logging for the long GPU jobs.

These runs take hours and previously emitted almost nothing between coarse milestones, so a
stall was indistinguishable from healthy work. Every helper here prints the *elapsed time since
process start* plus current/peak GPU memory and process RSS, so the last line of a log is enough
to tell whether a job is progressing and what it is doing.

    from seq_core.proglog import Stage, banner, heartbeat, mem_str

    banner(LOGGER, 2, 5, "sequential GPTQ pass")
    with Stage(LOGGER, "block 3/28 hessians", every=1):
        ...
    heartbeat(LOGGER, "quantize down_proj", i, total, t0)

Design notes: no background threads (they interleave badly with tqdm and CUDA), no dependency
beyond torch/psutil-optional, and every call is cheap enough to sit inside a per-layer loop.
"""
from __future__ import annotations

import os
import time
from typing import Optional

_T0 = time.monotonic()

# HTTP chatter from huggingface_hub/httpx drowns out progress in multi-hour logs, and none of
# it indicates health. Callers that want it can re-enable these loggers.
def quiet_http_logs(level: int = 30) -> None:
    """Raise the level of noisy third-party loggers (default: WARNING)."""
    import logging as _l

    for name in ("httpx", "httpcore", "urllib3", "huggingface_hub",
                 "filelock", "fsspec", "datasets"):
        _l.getLogger(name).setLevel(level)


def elapsed() -> float:
    """Seconds since this module was first imported (≈ process start)."""
    return time.monotonic() - _T0


def _fmt_secs(s: float) -> str:
    s = float(s)
    if s < 60:
        return f"{s:5.1f}s"
    if s < 3600:
        return f"{int(s // 60):d}m{int(s % 60):02d}s"
    return f"{int(s // 3600):d}h{int((s % 3600) // 60):02d}m"


def mem_str() -> str:
    """Compact GPU (current/peak allocated, reserved) and process-RSS summary."""
    parts = []
    try:
        import torch

        if torch.cuda.is_available():
            gb = 1024 ** 3
            parts.append(
                "gpu %.2f/%.2fGB alloc, %.2fGB reserved"
                % (torch.cuda.memory_allocated() / gb,
                   torch.cuda.max_memory_allocated() / gb,
                   torch.cuda.memory_reserved() / gb)
            )
    except Exception:  # noqa: BLE001
        pass
    rss = _rss_gb()
    if rss is not None:
        parts.append("rss %.2fGB" % rss)
    return " | ".join(parts) if parts else "mem n/a"


def _rss_gb() -> Optional[float]:
    try:  # Linux/WSL: cheap and dependency-free
        with open(f"/proc/{os.getpid()}/statm", "r", encoding="ascii") as fh:
            pages = int(fh.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / (1024 ** 3)
    except Exception:  # noqa: BLE001
        try:
            import psutil  # optional

            return psutil.Process().memory_info().rss / (1024 ** 3)
        except Exception:  # noqa: BLE001
            return None


def banner(logger, step: int, total: int, what: str) -> None:
    """Top-level phase marker, so a log reads as a checklist."""
    logger.info("=" * 12 + " PHASE %d/%d: %s [t+%s] " + "=" * 12,
                step, total, what, _fmt_secs(elapsed()))


def heartbeat(logger, what: str, done: int, total: int, t_start: float,
              every: int = 1, extra: str = "") -> None:
    """Progress inside a long inner loop; emits only every `every` items.

    Prints completion fraction, rate and a projected remaining time so a slow-but-moving loop is
    obviously distinguishable from a hung one.
    """
    if every > 1 and done % every != 0 and done != total:
        return
    dt = max(1e-9, time.monotonic() - t_start)
    rate = done / dt
    eta = (total - done) / rate if rate > 0 else float("nan")
    logger.info("    %s: %d/%d (%.1f%%) %s elapsed, %.1f/s, eta %s%s | %s",
                what, done, total, 100.0 * done / max(1, total), _fmt_secs(dt),
                rate, _fmt_secs(eta), (" | " + extra) if extra else "", mem_str())


class Stage:
    """Context manager logging START/END of a named step with timing and memory.

    `total`/`index` populate a running ETA across repeated stages (e.g. decoder blocks).
    """

    def __init__(self, logger, name: str, *, index: Optional[int] = None,
                 total: Optional[int] = None, quiet_start: bool = False):
        self.logger = logger
        self.name = name
        self.index = index
        self.total = total
        self.quiet_start = quiet_start
        self.t0 = 0.0

    def __enter__(self) -> "Stage":
        self.t0 = time.monotonic()
        if not self.quiet_start:
            self.logger.info("  [t+%s] START %s | %s",
                             _fmt_secs(elapsed()), self.name, mem_str())
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        dt = time.monotonic() - self.t0
        if exc_type is not None:
            self.logger.error("  [t+%s] FAILED %s after %s: %s: %s",
                              _fmt_secs(elapsed()), self.name, _fmt_secs(dt),
                              exc_type.__name__, exc)
            return False
        eta = ""
        if self.index is not None and self.total:
            remaining = self.total - self.index
            if remaining > 0:
                # mean-rate ETA over completed work, not just this stage
                mean = elapsed() / max(1, self.index)
                eta = f" | eta {_fmt_secs(mean * remaining)} for {remaining} more"
        self.logger.info("  [t+%s] END   %s in %s%s | %s",
                         _fmt_secs(elapsed()), self.name, _fmt_secs(dt), eta, mem_str())
        return False
