# Phase Timing

**Stack:** torch-spyre (new, Inductor-based).

Phase timing answers *"how much time went to the torch-spyre frontend, to the
DeepTools backend compiler, and to launching on the flex runtime?"* — a
compile-time and launch-time cost breakdown, printed as an aggregated table.

It is deliberately **not** a trace. Kineto/AIUPTI activity collection carries
real overhead and requires an active `torch.profiler` session; phase timing is a
handful of `perf_counter_ns` calls behind one environment variable, so it can be
left in place on runs where a trace would distort the thing being measured.

The design mirrors the flex runtime's `GlobalTimingProfile` backend
(`FLEX_TIMING_PROFILE`): one env var arms it, the disabled path is a
module-level boolean test, and the output is an aggregated console table rather
than a per-event record stream.

## Enabling

```bash
TORCH_SPYRE_PHASE_TIMING=1 \
  TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
  SPYRE_KERNEL_CACHE=0 \
  python3 my_script.py
```

The report is written to **stderr** when the process exits.

| Variable | Purpose |
|---|---|
| `TORCH_SPYRE_PHASE_TIMING=1` | Arm phase timing (accepts `1/0`, `true/false`, `yes/no`, `on/off`) |
| `TORCH_SPYRE_PHASE_TIMING_JSON=<path>` | Also write the report as JSON |

:::{important}
Both cache variables matter. On a kernel-cache hit neither `generate_bundle`
nor `dxp_standalone` runs at all, so backend compile time silently reads as
zero. `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` and `SPYRE_KERNEL_CACHE=0` are two
*independent* caches and both must be off for a true cold-compile measurement.
The report header prints the observed cache state so a saved report stays
interpretable later.
:::

## Example output

```text
=== torch-spyre phase timing (caches: inductor=off kernel=off) ===
process elapsed: 8241.3 ms   dxp: pool(32 threads)

phase                                                    count   total_ms   self_ms  notes
----------------------------------------------------------------------------------------
  backend.dxp_wait                                           3     3104.2    3104.2
  backend.dxp_submit                                         3        8.1       8.1
  backend.kernel_cache                                       0        0.0       0.0  hits=0 misses=3

frontend.compile_fx                                          1     6012.7     418.3
  frontend.CustomPreSchedulingPasses.propagate_layouts       6     1932.1    1932.1
  frontend.CustomPreSchedulingPasses.work_division           6      884.5     884.5
  frontend.codegen_node                                      3      412.9     412.9
  frontend.generate_bundle                                   3      221.8     221.8

runtime.launch_jobplan                                      10       42.1      42.1
runtime.prepare_kernel                                       3       88.4      88.4
----------------------------------------------------------------------------------------
backend (nested, incl. above)                                       3112.3
frontend (wall, top-level)                                          6012.7
runtime (wall, top-level)                                            130.5
```

## Reading the table

- **`total_ms` is inclusive; `self_ms` excludes instrumented children.** A
  parent phase's `self_ms` is the work it did that no finer phase accounts for
  — so `frontend.compile_fx` self time is lowering, scheduling and glue not
  attributed to a named pass.
- **Indented rows ran nested.** The group footer distinguishes `wall` (group
  entered at top level, so the figure is real elapsed time) from
  `nested, incl. above` (the group ran inside another group and its time is
  already counted in that group's wall figure). Backend compilation happens
  inside `compile_fx`, so `backend` is normally nested.
- **`runtime.launch_jobplan` is host-side enqueue cost, not device time.**
  Launches into flex are asynchronous (`SpyreStream::launchCompute` →
  `flex::RuntimeStream::launchOperationCompute`), so this phase measures the
  cost of *submitting* work. Device execution time needs
  `ProfilerActivity.PrivateUse1` (see [PyTorch Profiler](pytorch_profiler.md))
  or the flex runtime's own `FLEX_TIMING_PROFILE`.
- **The groups are different kinds of cost.** `frontend` and `backend` are
  one-time compile costs; `runtime.*` recurs on every iteration. Do not add
  them into a single number.

### DXP runs in a process pool

The header reports `dxp: pool(N threads)` or `dxp: inline`. When
`async_dxp_compile` is on and Inductor has more than one compile thread, DXP
compilation is submitted to Inductor's **subprocess** pool, so the work happens
in a different process from the accumulator. Backend time is therefore measured
at the two parent-side boundaries that bound it:

| Phase | Meaning |
|---|---|
| `backend.dxp_submit` | Handing the bundle to the pool. Small on the pooled path; on the inline path it contains the entire DXP run. |
| `backend.dxp_wait` | Blocking on the compile future, resolved during `async_compile.wait()`. **This is the end-to-end backend figure.** |
| `backend.dxp_standalone` | The `subprocess.run` itself, with `child_cpu`. Only recorded when DXP runs inline — in a pool worker the record dies with the worker process. |

With several kernels compiling in parallel, summed `dxp_wait` can exceed real
elapsed time, because the waits overlap. For a clean single-threaded
attribution, force inline compilation:

```bash
TORCHINDUCTOR_COMPILE_THREADS=1 TORCH_SPYRE_PHASE_TIMING=1 python3 my_script.py
```

That also makes `backend.dxp_standalone` and its `child_cpu` figure appear,
which is what distinguishes a compute-bound backend compiler from one blocked
on I/O. (`child_cpu` comes from `getrusage(RUSAGE_CHILDREN)`, which counts only
reaped children process-wide, so treat it as a lower bound.)

## Instrumented phases

| Phase | Where |
|---|---|
| `frontend.compile_fx` | `torch_spyre/_inductor/__init__.py` — the Spyre `compile_fx` wrapper; outermost frontend boundary |
| `frontend.<Pipeline>` | `torch_spyre/_inductor/passes.py` — whole FX-graph pass pipeline |
| `frontend.<Pipeline>.<pass>` | `torch_spyre/_inductor/passes.py` — every pass in all six pass pipelines |
| `frontend.codegen_node` | `torch_spyre/_inductor/scheduler.py` — LoopLevelIR → `OpSpec` → SDSC |
| `frontend.generate_bundle` | `torch_spyre/execution/async_compile.py` — in-process bundle emit |
| `backend.dxp_submit` | `torch_spyre/execution/async_compile.py` — submit DXP (inline or pooled) |
| `backend.dxp_wait` | `torch_spyre/execution/async_compile.py` — resolve the compile future |
| `backend.dxp_standalone` | `torch_spyre/execution/async_compile.py` — the `dxp_standalone` subprocess (inline path) |
| `backend.dbo_opt` | `torch_spyre/execution/async_compile.py` — `dbo-opt` on the KTIR path |
| `backend.kernel_cache` | hit/miss counters, so a suspiciously cheap backend is explained |
| `runtime.prepare_kernel` | `torch_spyre/execution/kernel_runner.py` — one-time bundle load into the runtime |
| `runtime.launch_jobplan` | `torch_spyre/execution/kernel_runner.py` — host-side enqueue into flex |

## Python API

```python
import torch
import torch_spyre

if torch_spyre.profiler.phase_timing_enabled():
    # Drop warm-up/compile phases, then measure only steady-state iterations.
    torch_spyre.profiler.reset_phase_timing()
    for _ in range(10):
        model(inputs)
    print(torch_spyre.profiler.get_phase_timing_report())

    # Or as structured data.
    data = torch_spyre.profiler.get_phase_timing_report(as_json=True)
    print(data["phases"]["runtime.launch_jobplan"]["total_ms"])
```

`get_phase_timing_report()` returns `None` when phase timing is disabled.

## Relationship to the other tools

| Question | Tool |
|---|---|
| Where did compile time go (frontend vs. DeepTools)? | **Phase timing** |
| How long did each device kernel take? | `torch.profiler` with `ProfilerActivity.PrivateUse1` — see [PyTorch Profiler](pytorch_profiler.md) |
| Where did time go *inside* the flex runtime? | `FLEX_TIMING_PROFILE` (flex's own timing backend) |
| How close is a kernel to the hardware bound? | PT-active utilization — see [Performance Analysis Methodology](performance_analysis_methodology.md) |

## See also

- [Performance Analysis Methodology](performance_analysis_methodology.md) — runtime breakdown from a trace
- [Environment variables](environment_variables.md) — compiler pipeline logging
- [PyTorch Profiler](pytorch_profiler.md) — device-side kernel timing
