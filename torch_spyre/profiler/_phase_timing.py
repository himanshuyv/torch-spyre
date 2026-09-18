# Copyright 2025-2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Lightweight phase timing for the Spyre compile and launch pipeline.

Answers "where did the time go?" across the three cost centers a Spyre workload
pays for:

* ``frontend.*`` -- torch-spyre's Inductor frontend (Python, in-process).
* ``backend.*``  -- the DeepTools backend compiler (``dxp_standalone`` /
  ``dbo-opt``), which runs as a child process and may run in parallel.
* ``runtime.*``  -- host-side launch into the flex runtime.

Design mirrors flex's ``GlobalTimingProfile`` (see flex
``src/telemetry/README.md``): a single env var arms it, the disabled path is a
module-level boolean test, and the output is an aggregated console table rather
than a per-event trace. It deliberately does NOT depend on ``torch.profiler``:
Kineto/AIUPTI collection is the overhead this is meant to avoid, and
``record_function`` markers only materialize while a profiler session is live.

Enable with::

    TORCH_SPYRE_PHASE_TIMING=1

Optional::

    TORCH_SPYRE_PHASE_TIMING_JSON=<path>   also write the report as JSON

Timings are wall-clock (``time.perf_counter_ns``). Backend-compiler phases also
record child-process CPU time via ``getrusage(RUSAGE_CHILDREN)``, which
separates "the backend compiler is computing" from "we are waiting on I/O".

Nesting is tracked per thread so a parent phase can report *self* time
(exclusive of instrumented children) alongside its total. Phase records
themselves are guarded by a lock, so concurrent compiles aggregate correctly.

Concurrency caveat: DXP compilation is submitted to Inductor's process pool
when ``async_dxp_compile`` is on and compile threads > 1. Work in those worker
processes cannot be seen from here, so backend time is measured at the two
parent-side boundaries that bound it -- submission and future resolution --
rather than inside the worker. Summed backend wall time can then exceed real
elapsed time when several kernels compile in parallel; ``backend.dxp_wait`` is
the honest end-to-end figure. See ``docs/source/user_guide/profiling``.
"""

import atexit
import json
import os
import resource
import sys
import threading
import time
from contextlib import contextmanager
from typing import Any, Optional


def _env_flag(name: str, default: str = "0") -> bool:
    """Parse a boolean env var accepting 1/0, true/false, yes/no, on/off."""
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


# Read once at import. Static for the process lifetime, so the hot path is a
# single global load -- matching the flex TIMING backend's cached-state design.
ENABLED: bool = _env_flag("TORCH_SPYRE_PHASE_TIMING")

_JSON_PATH: Optional[str] = os.getenv("TORCH_SPYRE_PHASE_TIMING_JSON") or None

# Phase name -> accumulated stats. Guarded by _lock.
#   count      number of completed intervals
#   total_ns   inclusive wall time
#   self_ns    exclusive wall time (total minus instrumented children)
#   root_ns    portion measured with no instrumented parent (for group totals)
#   extra      free-form per-phase counters (e.g. child CPU time)
_records: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()

# Per-thread stack of [phase_name, child_ns] frames, for self-time accounting.
_stack = threading.local()

_t_origin_ns = time.perf_counter_ns()
_dumped = False


def _frames() -> list:
    frames = getattr(_stack, "frames", None)
    if frames is None:
        frames = []
        _stack.frames = frames
    return frames


def _record(
    name: str,
    total_ns: int,
    self_ns: int,
    extra: Optional[dict] = None,
    root_ns: int = 0,
    bump_count: bool = True,
):
    """Accumulate one sample.

    ``root_ns`` is the portion of ``total_ns`` measured with no instrumented
    parent on the stack. Group totals sum ``root_ns`` so nesting can never
    double-count, regardless of how deep a phase name looks.
    """
    with _lock:
        rec = _records.get(name)
        if rec is None:
            rec = {
                "count": 0,
                "total_ns": 0,
                "self_ns": 0,
                "root_ns": 0,
                "extra": {},
            }
            _records[name] = rec
        if bump_count:
            rec["count"] += 1
        rec["total_ns"] += total_ns
        rec["self_ns"] += self_ns
        rec["root_ns"] += root_ns
        if extra:
            for k, v in extra.items():
                rec["extra"][k] = rec["extra"].get(k, 0) + v


@contextmanager
def phase(name: str, **extra_counters):
    """Time a named phase. No-op (near-zero cost) when disabled.

    ``extra_counters`` are summed into the phase's ``extra`` dict, for
    non-time counters such as kernel counts or cache hits.
    """
    if not ENABLED:
        yield
        return

    frames = _frames()
    frames.append([name, 0])
    t0 = time.perf_counter_ns()
    try:
        yield
    finally:
        total_ns = time.perf_counter_ns() - t0
        frame = frames.pop()
        child_ns = frame[1]
        # Attribute our full duration to the parent's child total, so the
        # parent's self time excludes us. With no parent we are a root sample,
        # and our full duration counts toward the group total.
        if frames:
            frames[-1][1] += total_ns
            root_ns = 0
        else:
            root_ns = total_ns
        _record(
            name,
            total_ns,
            total_ns - child_ns,
            extra_counters or None,
            root_ns=root_ns,
        )


@contextmanager
def phase_with_child_cpu(name: str, **extra_counters):
    """Like :func:`phase`, but also accumulates child-process CPU time.

    Use around a ``subprocess`` call to the backend compiler: wall time alone
    cannot distinguish a compute-bound backend from one blocked on I/O.
    """
    if not ENABLED:
        yield
        return

    ru0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu0 = ru0.ru_utime + ru0.ru_stime
    with phase(name, **extra_counters):
        yield
    ru1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu1 = ru1.ru_utime + ru1.ru_stime
    # RUSAGE_CHILDREN is process-wide and only counts *reaped* children, so
    # this is a lower bound under concurrency. Recorded as a diagnostic, not a
    # billed sample -- hence bump_count=False.
    _record(
        name,
        0,
        0,
        {"child_cpu_ns": int((cpu1 - cpu0) * 1e9)},
        bump_count=False,
    )


def count(name: str, **counters):
    """Record counters against ``name`` without timing anything."""
    if not ENABLED:
        return
    _record(name, 0, 0, counters, bump_count=False)


def reset():
    """Drop all accumulated phases (useful between measured regions)."""
    global _t_origin_ns
    with _lock:
        _records.clear()
    _t_origin_ns = time.perf_counter_ns()


def _cache_state() -> str:
    """Describe cache settings, since a cache hit makes compile time vanish."""
    try:
        import torch

        inductor_off = bool(torch._inductor.config.force_disable_caches)
    except Exception:
        inductor_off = False
    try:
        from torch_spyre._inductor import config as _spyre_config

        kernel_on = bool(_spyre_config.spyre_kernel_cache)
    except Exception:
        kernel_on = True
    return (
        f"inductor={'off' if inductor_off else 'ON'} "
        f"kernel={'ON' if kernel_on and not inductor_off else 'off'}"
    )


def _dxp_mode() -> str:
    """Describe whether DXP compiles in-process or in the pool."""
    try:
        from torch._inductor.async_compile import get_compile_threads
        from torch_spyre._inductor import config as _spyre_config

        threads = get_compile_threads()
        if _spyre_config.async_dxp_compile and threads > 1:
            return f"pool({threads} threads)"
        return "inline"
    except Exception:
        return "unknown"


def report() -> str:
    """Render the aggregated phase table."""
    with _lock:
        snapshot = {k: dict(v, extra=dict(v["extra"])) for k, v in _records.items()}

    if not snapshot:
        return "=== torch-spyre phase timing: no phases recorded ==="

    elapsed_ms = (time.perf_counter_ns() - _t_origin_ns) / 1e6

    # Group by top-level prefix (frontend / backend / runtime), then sort each
    # group by total time so the expensive phase is the one you read first.
    def sort_key(item):
        name, rec = item
        return (name.split(".")[0], -rec["total_ns"], name)

    rows = sorted(snapshot.items(), key=sort_key)

    name_w = max(52, min(72, max(len(n) for n in snapshot) + 4))
    width = name_w + 32

    # Sum only root samples (no instrumented parent), so a group total is real
    # wall time rather than an accidental sum over nested phases.
    total_by_group: dict[str, int] = {}
    for name, rec in rows:
        group = name.split(".")[0]
        if rec["root_ns"]:
            total_by_group[group] = total_by_group.get(group, 0) + rec["root_ns"]

    lines = []
    lines.append(f"=== torch-spyre phase timing (caches: {_cache_state()}) ===")
    lines.append(f"process elapsed: {elapsed_ms:.1f} ms   dxp: {_dxp_mode()}")
    lines.append("")
    lines.append(
        f"{'phase':<{name_w}}{'count':>6}{'total_ms':>11}{'self_ms':>10}  notes"
    )
    lines.append("-" * width)

    current_group = None
    for name, rec in rows:
        group = name.split(".")[0]
        if current_group is not None and group != current_group:
            lines.append("")
        current_group = group

        notes = []
        extra = rec["extra"]
        if "child_cpu_ns" in extra:
            notes.append(f"child_cpu={extra['child_cpu_ns'] / 1e6:.1f}ms")
        for k, v in sorted(extra.items()):
            if k != "child_cpu_ns":
                notes.append(f"{k}={v}")

        # Indent phases that only ever ran nested, so the table reads as a
        # tree instead of a flat list of similar-looking names.
        label = name if rec["root_ns"] else "  " + name
        lines.append(
            f"{label:<{name_w}}{rec['count']:>6}"
            f"{rec['total_ns'] / 1e6:>11.1f}"
            f"{rec['self_ns'] / 1e6:>10.1f}"
            f"  {' '.join(notes)}"
        )

    lines.append("-" * width)

    # Per-group totals. `wall` is real elapsed time for groups entered at top
    # level; `nested` groups (e.g. backend inside a compile) report the sum of
    # their phases, which is time already inside another group's wall figure.
    nested_by_group: dict[str, int] = {}
    for name, rec in rows:
        group = name.split(".")[0]
        if not rec["root_ns"] and rec["total_ns"]:
            nested_by_group[group] = nested_by_group.get(group, 0) + rec["total_ns"]

    for group in sorted(set(total_by_group) | set(nested_by_group)):
        if group in total_by_group:
            lines.append(
                f"{group + ' (wall, top-level)':<{name_w}}{'':>6}"
                f"{total_by_group[group] / 1e6:>11.1f}"
            )
        else:
            lines.append(
                f"{group + ' (nested, incl. above)':<{name_w}}{'':>6}"
                f"{nested_by_group[group] / 1e6:>11.1f}"
            )

    lines.append("")
    lines.append(
        "note: frontend/backend are one-time compile costs; runtime.* recurs "
        "per iteration."
    )
    if "pool" in _dxp_mode():
        lines.append(
            "note: DXP ran in the process pool -- backend.dxp_submit excludes "
            "worker time; backend.dxp_wait is the end-to-end figure."
        )
    return "\n".join(lines)


def as_dict() -> dict:
    """The report as a plain dict, for JSON export or programmatic checks."""
    with _lock:
        snapshot = {k: dict(v, extra=dict(v["extra"])) for k, v in _records.items()}
    return {
        "elapsed_ms": (time.perf_counter_ns() - _t_origin_ns) / 1e6,
        "caches": _cache_state(),
        "dxp_mode": _dxp_mode(),
        "phases": {
            name: {
                "count": rec["count"],
                "total_ms": rec["total_ns"] / 1e6,
                "self_ms": rec["self_ns"] / 1e6,
                "root_ms": rec["root_ns"] / 1e6,
                **{
                    k: (v / 1e6 if k.endswith("_ns") else v)
                    for k, v in rec["extra"].items()
                },
            }
            for name, rec in snapshot.items()
        },
    }


def dump():
    """Print the report (and write JSON if requested). Idempotent."""
    global _dumped
    if not ENABLED or _dumped:
        return
    _dumped = True
    if not _records:
        return
    print(report(), file=sys.stderr)
    if _JSON_PATH:
        try:
            with open(_JSON_PATH, "w") as fh:
                json.dump(as_dict(), fh, indent=2, sort_keys=True)
            print(f"[phase timing] wrote {_JSON_PATH}", file=sys.stderr)
        except OSError as exc:
            print(f"[phase timing] could not write JSON: {exc}", file=sys.stderr)


def get_phase_timing_report(as_json: bool = False):
    """Return the phase-timing report.

    Args:
        as_json: when True return a dict instead of the rendered table.

    Returns None when phase timing is disabled
    (``TORCH_SPYRE_PHASE_TIMING`` unset).
    """
    if not ENABLED:
        return None
    return as_dict() if as_json else report()


def phase_timing_enabled() -> bool:
    """True when ``TORCH_SPYRE_PHASE_TIMING`` armed phase timing."""
    return ENABLED


def reset_phase_timing() -> None:
    """Clear accumulated phases, e.g. to exclude warm-up from a measurement."""
    reset()


if ENABLED:
    atexit.register(dump)
