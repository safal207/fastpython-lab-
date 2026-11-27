"""Hardware evidence boundaries and native probe correctness gates."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from alexpy import memory_lab as mem


class MemoryLabTests(unittest.TestCase):
    def test_native_ring_is_a_cycle_and_scan_has_exact_checksum(self):
        if mem._memprobe is None:
            self.skipTest("native extension not built")
        for nodes in (2, 3, 8, 127):
            ring = mem._memprobe.make_ring(nodes, 2026)
            self.assertEqual(mem._memprobe.nodes(ring), nodes)
            self.assertEqual(mem._memprobe.walk(ring, nodes), 0)
            self.assertEqual(mem._memprobe.scan(ring, 2), nodes * (nodes - 1))
            self.assertEqual(mem._memprobe.walk(ring, nodes + 1),
                             mem._memprobe.walk(ring, 1))

    def test_native_rejects_invalid_sizes_and_capsules(self):
        if mem._memprobe is None:
            self.skipTest("native extension not built")
        with self.assertRaises(ValueError):
            mem._memprobe.make_ring(1)
        with self.assertRaises(ValueError):
            mem._memprobe.make_ring(16_777_217)
        ring = mem._memprobe.make_ring(8)
        with self.assertRaises(ValueError):
            mem._memprobe.walk(ring, -1)
        with self.assertRaises(ValueError):
            mem._memprobe.scan(ring, 0)
        with self.assertRaises(ValueError):
            mem._memprobe.walk(ring, 200_000_001)
        with self.assertRaises(ValueError):
            mem._memprobe.scan(ring, 200_000_001)
        with self.assertRaises(ValueError):
            mem._memprobe.walk("not a capsule", 8)

    def test_topology_size_parser(self):
        self.assertEqual(mem.parse_cache_size("32K\n"), 32 * 1024)
        self.assertEqual(mem.parse_cache_size("1M"), 1024 * 1024)
        self.assertEqual(mem.parse_cache_size("4G"), 4 * 1024**3)
        with self.assertRaises(ValueError):
            mem.parse_cache_size("unknown")

    def test_fake_sysfs_is_only_reported_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for idx, (level, kind, size) in enumerate([(1, "Data", "32K"), (2, "Unified", "1M")]):
                d = root / f"index{idx}"; d.mkdir()
                (d / "level").write_text(str(level))
                (d / "type").write_text(kind)
                (d / "size").write_text(size)
                (d / "shared_cpu_list").write_text("0-1")
            rows = mem.linux_cache_topology(root)
            self.assertEqual([(r["level"], r["bytes"]) for r in rows],
                             [(1, 32768), (2, 1048576)])
            self.assertTrue(all(r["source"] == "linux_sysfs_reported" for r in rows))

    @staticmethod
    def example_cell(epoch, size, latency, time_ns):
        return mem.MemoryCell(epoch, size, time_ns, latency, 1e9, 100,
                              100, (100,), (100,))

    def test_space_and_time_edges_and_cache_crossing(self):
        rows = [self.example_cell(0, 4096, 2., 100),
                self.example_cell(0, 65536, 4., 200),
                self.example_cell(1, 4096, 3., 300),
                self.example_cell(1, 65536, 6., 400)]
        edges = mem.analyze_transitions(rows, [{"level": 1, "type": "Data", "bytes": 32768}])
        self.assertEqual(len(edges["spatial_edges"]), 2)
        self.assertEqual(len(edges["temporal_edges"]), 2)
        self.assertEqual(edges["spatial_edges"][0]["reported_cache_boundaries_crossed"], ["L1:Data"])
        self.assertAlmostEqual(edges["spatial_edges"][0]["chase_latency_ratio"], 2.)
        self.assertGreater(edges["temporal_edges"][0]["d_log_latency_per_elapsed_second"], 0)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            mem.analyze_transitions(rows + [rows[0]], [])
        with self.assertRaisesRegex(ValueError, "temporal"):
            mem.analyze_transitions([rows[0], self.example_cell(1, 4096, 2, 90)], [])

    def test_validate_raw_samples_and_evidence(self):
        with self.assertRaises(ValueError):
            mem.MemoryCell(0, 4096, 0, float('nan'), 1e9, 10, 10, (3,), (4,))
        with self.assertRaises(ValueError):
            mem.MemoryCell(0, 4096, 0, 2., 1e9, 10, 10, tuple(), (4,))
        with self.assertRaises(ValueError):
            mem.make_report([], [])
        report = mem.make_report([self.example_cell(0, 4096, 2., 100)], [])
        self.assertEqual(report["hardware_events"]["status"], "NOT_REQUESTED")
        self.assertIn("NO_AUTOPROMOTION", report["evidence"])
        self.assertNotIn("cache_miss_count", report["cells"][0])
        json.dumps(report)

    def test_invalid_measurement_configuration_rejected(self):
        with self.assertRaises(ValueError):
            mem.collect_memory_lattice(sizes_kib=(4,4))
        with self.assertRaises(ValueError):
            mem.collect_memory_lattice(sizes_kib=(0,))
        with self.assertRaises(ValueError):
            mem.collect_memory_lattice(epochs=0)
        with self.assertRaises(ValueError):
            mem.collect_memory_lattice(target_accesses=30)

    def test_native_lattice_smoke_and_time_separation(self):
        if mem._memprobe is None:
            self.skipTest("native extension not built")
        cells = mem.collect_memory_lattice(sizes_kib=(1,4,32), epochs=2,
                                           repeats=2, target_accesses=1000)
        self.assertEqual(len(cells), 6)
        self.assertTrue(all(c.chase_ns_per_access > 0 for c in cells))
        report = mem.make_report(cells, [])
        self.assertEqual(len(report["transitions"]["spatial_edges"]), 4)
        self.assertEqual(len(report["transitions"]["temporal_edges"]), 3)
        self.assertTrue(all(len(c.chase_samples_ns) == 2 for c in cells))

    def test_perf_unavailable_is_not_fabricated_as_zero(self):
        if mem._memprobe is None:
            self.skipTest("native extension not built")
        ring = mem._memprobe.make_ring(32)
        with patch.object(mem, "_open_perf_event", side_effect=PermissionError(13, "blocked")):
            counters = mem.try_hardware_counters(ring, steps=128)
        if sys.platform.startswith("linux"):
            self.assertEqual(counters["status"], "UNAVAILABLE_NO_HARDWARE_COUNTERS")
            self.assertIsNone(counters["l1d_read_misses"]["count"])
            self.assertIsNone(counters["llc_read_misses"]["count"])
            self.assertEqual(counters["l2_read_misses"]["status"],
                             "UNAVAILABLE_MODEL_SPECIFIC_EVENT_REQUIRED")
        else:
            self.assertEqual(counters["status"], "UNAVAILABLE_NOT_LINUX")


if __name__ == "__main__":
    unittest.main()
