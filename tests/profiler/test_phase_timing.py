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

"""Unit tests for the phase-timing accumulator.

These exercise the accounting logic directly rather than through a compile, so
they need no device: the point is that ``self_ms`` excludes nested children,
that group totals never double-count, and that the disabled path stays inert.
"""

import os
import subprocess
import sys
import threading
import time
import unittest
from unittest.mock import patch

from torch_spyre.profiler import _phase_timing


class PhaseTimingTest(unittest.TestCase):
    def setUp(self):
        # The module reads its env var once at import; force it on for the
        # duration of each test and start from a clean slate.
        self._prev_enabled = _phase_timing.ENABLED
        _phase_timing.ENABLED = True
        _phase_timing.reset()

    def tearDown(self):
        _phase_timing.ENABLED = self._prev_enabled
        _phase_timing.reset()

    def _phases(self):
        return _phase_timing.as_dict()["phases"]

    def test_records_a_single_phase(self):
        with _phase_timing.phase("frontend.compile_fx"):
            time.sleep(0.02)

        rec = self._phases()["frontend.compile_fx"]
        self.assertEqual(rec["count"], 1)
        self.assertGreaterEqual(rec["total_ms"], 15.0)
        # A phase with no children spends all of its time in itself.
        self.assertAlmostEqual(rec["total_ms"], rec["self_ms"], places=3)

    def test_self_time_excludes_nested_children(self):
        with _phase_timing.phase("frontend.compile_fx"):
            time.sleep(0.02)
            with _phase_timing.phase("frontend.codegen_node"):
                time.sleep(0.04)

        phases = self._phases()
        parent = phases["frontend.compile_fx"]
        child = phases["frontend.codegen_node"]

        self.assertGreaterEqual(parent["total_ms"], child["total_ms"])
        # Parent self time is its own work only, so the child's duration must
        # not appear in it.
        self.assertLess(parent["self_ms"], child["total_ms"])
        self.assertAlmostEqual(
            parent["total_ms"] - child["total_ms"],
            parent["self_ms"],
            delta=5.0,
        )

    def test_repeated_phase_accumulates(self):
        for _ in range(4):
            with _phase_timing.phase("runtime.launch_jobplan"):
                time.sleep(0.005)

        rec = self._phases()["runtime.launch_jobplan"]
        self.assertEqual(rec["count"], 4)
        self.assertGreaterEqual(rec["total_ms"], 15.0)

    def test_only_root_samples_count_toward_group_total(self):
        """A nested phase must not inflate its group's wall time."""
        with _phase_timing.phase("frontend.compile_fx"):
            with _phase_timing.phase("backend.dxp_standalone"):
                time.sleep(0.01)

        phases = self._phases()
        # compile_fx ran at top level; dxp ran inside it.
        self.assertGreater(phases["frontend.compile_fx"]["root_ms"], 0.0)
        self.assertEqual(phases["backend.dxp_standalone"]["root_ms"], 0.0)

    def test_counters_do_not_create_timing_samples(self):
        _phase_timing.count("backend.kernel_cache", hits=2, misses=1)
        rec = self._phases()["backend.kernel_cache"]
        self.assertEqual(rec["hits"], 2)
        self.assertEqual(rec["misses"], 1)
        # A pure counter is not a timed interval.
        self.assertEqual(rec["count"], 0)
        self.assertEqual(rec["total_ms"], 0.0)

    def test_child_cpu_is_captured_without_inflating_count(self):
        with _phase_timing.phase_with_child_cpu("backend.dxp_standalone"):
            subprocess.run(
                [sys.executable, "-c", "sum(i * i for i in range(500000))"],
                check=True,
            )

        rec = self._phases()["backend.dxp_standalone"]
        # One invocation, even though child CPU is recorded separately.
        self.assertEqual(rec["count"], 1)
        self.assertGreater(rec["child_cpu_ns"], 0.0)

    def test_exception_still_records_the_phase(self):
        with self.assertRaises(ValueError):
            with _phase_timing.phase("frontend.compile_fx"):
                raise ValueError("boom")

        # A failed compile is exactly when the timing matters, so the phase
        # must survive the exception.
        self.assertEqual(self._phases()["frontend.compile_fx"]["count"], 1)

    def test_nested_exception_does_not_corrupt_the_stack(self):
        """An inner failure must still pop its frame, or later phases skew."""
        with _phase_timing.phase("frontend.compile_fx"):
            with self.assertRaises(RuntimeError):
                with _phase_timing.phase("backend.dxp_standalone"):
                    raise RuntimeError("backend failed")

        # Stack unwound cleanly, so a subsequent top-level phase is still root.
        with _phase_timing.phase("runtime.launch_jobplan"):
            pass
        self.assertGreaterEqual(
            self._phases()["runtime.launch_jobplan"]["root_ms"], 0.0
        )
        self.assertEqual(len(_phase_timing._frames()), 0)

    def test_threads_accumulate_into_shared_records(self):
        def work():
            with _phase_timing.phase("frontend.compile_fx"):
                time.sleep(0.01)

        threads = [threading.Thread(target=work) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        rec = self._phases()["frontend.compile_fx"]
        self.assertEqual(rec["count"], 4)
        # Each thread has its own nesting stack, so all four are root samples.
        self.assertGreater(rec["root_ms"], 0.0)

    def test_nesting_is_per_thread(self):
        """A phase on one thread must not be treated as another's parent."""
        started = threading.Event()
        release = threading.Event()

        def outer():
            with _phase_timing.phase("frontend.compile_fx"):
                started.set()
                release.wait(timeout=5)

        t = threading.Thread(target=outer)
        t.start()
        self.assertTrue(started.wait(timeout=5))
        # While the other thread sits inside compile_fx, a phase here is still
        # top-level.
        with _phase_timing.phase("runtime.launch_jobplan"):
            time.sleep(0.005)
        release.set()
        t.join()

        self.assertGreater(self._phases()["runtime.launch_jobplan"]["root_ms"], 0.0)

    def test_report_renders(self):
        with _phase_timing.phase("frontend.compile_fx"):
            with _phase_timing.phase("backend.dxp_standalone"):
                time.sleep(0.005)
        _phase_timing.count("backend.kernel_cache", misses=1)

        text = _phase_timing.report()
        self.assertIn("torch-spyre phase timing", text)
        self.assertIn("frontend.compile_fx", text)
        self.assertIn("backend.dxp_standalone", text)
        # Group footers distinguish real wall time from nested time.
        self.assertIn("wall, top-level", text)
        self.assertIn("nested, incl. above", text)

    def test_report_without_records(self):
        self.assertIn("no phases recorded", _phase_timing.report())

    def test_reset_clears_records(self):
        with _phase_timing.phase("frontend.compile_fx"):
            pass
        self.assertIn("frontend.compile_fx", self._phases())
        _phase_timing.reset()
        self.assertEqual(self._phases(), {})

    def test_disabled_records_nothing(self):
        _phase_timing.ENABLED = False
        with _phase_timing.phase("frontend.compile_fx"):
            time.sleep(0.005)
        with _phase_timing.phase_with_child_cpu("backend.dxp_standalone"):
            pass
        _phase_timing.count("backend.kernel_cache", hits=1)
        self.assertEqual(self._phases(), {})

    def test_public_api_reflects_enabled_state(self):
        from torch_spyre import profiler

        self.assertTrue(profiler.phase_timing_enabled())
        with _phase_timing.phase("frontend.compile_fx"):
            pass
        self.assertIn("frontend.compile_fx", profiler.get_phase_timing_report())
        self.assertIn(
            "frontend.compile_fx",
            profiler.get_phase_timing_report(as_json=True)["phases"],
        )

        _phase_timing.ENABLED = False
        self.assertFalse(profiler.phase_timing_enabled())
        self.assertIsNone(profiler.get_phase_timing_report())

    def test_env_flag_parsing(self):
        for value in ("1", "true", "TRUE", "yes", "on", " on "):
            with patch.dict(os.environ, {"X_FLAG": value}):
                self.assertTrue(_phase_timing._env_flag("X_FLAG"), value)
        for value in ("0", "false", "no", "off", ""):
            with patch.dict(os.environ, {"X_FLAG": value}):
                self.assertFalse(_phase_timing._env_flag("X_FLAG"), value)


if __name__ == "__main__":
    unittest.main()
