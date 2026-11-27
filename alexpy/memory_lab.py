"""Hardware-aware memory-access experiment with a fail-closed evidence boundary.

Two native microbenchmarks keep Python out of the hot loop. Sysfs reports
*advertised* cache topology. Optional Linux perf_event_open supplies real
counter readings only when available; never infer cache misses from timing.
The experiment does NOT measure whole-Python speedup or isolate cache causality.
"""
from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import errno
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import sys
import time

try:
    from . import _memprobe
except ImportError:
    _memprobe = None


def parse_cache_size(value: str) -> int:
    match = re.fullmatch(r"\s*(\d+)\s*([KMG]?)B?\s*", value.upper())
    if not match:
        raise ValueError("unsupported cache-size string")
    return int(match.group(1)) * 1024 ** {"": 0, "K": 1, "M": 2, "G": 3}[match.group(2)]


def linux_cache_topology(root: Path = Path("/sys/devices/system/cpu/cpu0/cache")) -> list[dict]:
    """Reported metadata, not measured misses or guaranteed usable capacity."""
    if not root.is_dir():
        return []
    caches = []
    for directory in sorted(root.glob("index*")):
        try:
            level = int((directory / "level").read_text().strip())
            kind = (directory / "type").read_text().strip()
            size = parse_cache_size((directory / "size").read_text())
            shared = (directory / "shared_cpu_list").read_text().strip()
        except (OSError, ValueError):
            continue
        if kind not in ("Data", "Unified", "Instruction") or level < 1:
            continue
        caches.append({"level": level, "type": kind, "bytes": size,
                       "shared_cpu_list": shared, "source": "linux_sysfs_reported"})
    return sorted(caches, key=lambda x: (x["level"], x["type"]))


def cpu_description() -> str | None:
    if sys.platform != "linux":
        return None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name") or line.startswith("Hardware"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


@dataclass(frozen=True)
class MemoryCell:
    epoch: int
    footprint_bytes: int
    timestamp_ns: int
    chase_ns_per_access: float
    sequential_effective_bytes_per_sec: float
    walk_steps: int
    scan_reads: int
    chase_samples_ns: tuple[int, ...]
    scan_samples_ns: tuple[int, ...]

    def __post_init__(self):
        if self.epoch < 0 or self.footprint_bytes < 8 or self.timestamp_ns < 0:
            raise ValueError("invalid memory cell coordinates")
        if self.walk_steps < 1 or self.scan_reads < 1:
            raise ValueError("invalid counts")
        if not math.isfinite(self.chase_ns_per_access) or self.chase_ns_per_access <= 0:
            raise ValueError("invalid chase latency")
        if not math.isfinite(self.sequential_effective_bytes_per_sec) or self.sequential_effective_bytes_per_sec <= 0:
            raise ValueError("invalid scan throughput")
        if not self.chase_samples_ns or not self.scan_samples_ns or any(
            type(x) is not int or x <= 0 for x in self.chase_samples_ns + self.scan_samples_ns
        ):
            raise ValueError("missing or invalid raw timings")


def analyze_transitions(cells: list[MemoryCell], caches: list[dict]) -> dict:
    """Space/elapsed-time lattice edges. Cache crossings are hypotheses only."""
    seen = set()
    for cell in cells:
        key = (cell.epoch, cell.footprint_bytes)
        if key in seen:
            raise ValueError("duplicate memory lattice cell")
        seen.add(key)
    space = []
    for epoch in sorted({x.epoch for x in cells}):
        rows = sorted((c for c in cells if c.epoch == epoch), key=lambda c: c.footprint_bytes)
        for left, right in zip(rows, rows[1:]):
            d = math.log(right.footprint_bytes / left.footprint_bytes)
            slope = math.log(right.chase_ns_per_access / left.chase_ns_per_access) / d
            boundaries = [f"L{c['level']}:{c['type']}" for c in caches
                          if left.footprint_bytes < c["bytes"] <= right.footprint_bytes
                          and c["type"] != "Instruction"]
            space.append({"epoch": epoch, "from_bytes": left.footprint_bytes,
                          "to_bytes": right.footprint_bytes,
                          "d_log_chase_latency_d_log_working_set": slope,
                          "chase_latency_ratio": right.chase_ns_per_access / left.chase_ns_per_access,
                          "reported_cache_boundaries_crossed": boundaries,
                          "interpretation": "OBSERVATIONAL_NOT_CACHE_CAUSALITY"})
    temporal = []
    for size in sorted({x.footprint_bytes for x in cells}):
        rows = sorted((c for c in cells if c.footprint_bytes == size), key=lambda c: c.epoch)
        for left, right in zip(rows, rows[1:]):
            if right.epoch != left.epoch + 1 or right.timestamp_ns <= left.timestamp_ns:
                raise ValueError("invalid temporal measurement sequence")
            elapsed = (right.timestamp_ns - left.timestamp_ns) / 1e9
            temporal.append({"footprint_bytes": size, "from_epoch": left.epoch,
                             "to_epoch": right.epoch, "elapsed_seconds": elapsed,
                             "d_log_latency_per_elapsed_second":
                             math.log(right.chase_ns_per_access / left.chase_ns_per_access) / elapsed})
    return {"spatial_edges": space, "temporal_edges": temporal,
            "derivative_definition": "finite difference of log chase ns/access over log working-set bytes",
            "cache_interpretation": "crossings are topology overlays, not cache-miss observations"}


def _measure_ns(call, *args) -> int:
    t0 = time.perf_counter_ns()
    call(*args)
    return max(1, time.perf_counter_ns() - t0)


def collect_memory_lattice(*, sizes_kib=(4, 32, 256, 2048, 16384), epochs=2,
                           repeats=5, target_accesses=750_000, seed=131) -> list[MemoryCell]:
    if _memprobe is None:
        raise RuntimeError("native memory probe unavailable; build with python -m pip install -e .")
    if (not sizes_kib or len(set(sizes_kib)) != len(sizes_kib)
            or any(type(k) is not int or k < 1 or k > 65536 for k in sizes_kib)):
        raise ValueError("sizes_kib must be distinct integers in [1, 65536]")
    if (not 1 <= epochs <= 20 or not 1 <= repeats <= 99
            or not 1000 <= target_accesses <= 20_000_000):
        raise ValueError("epochs/repeats/target_accesses outside bounded protocol")
    rings = {}
    for k in sorted(sizes_kib):
        nodes = k * 1024 // 4
        ring = _memprobe.make_ring(nodes, seed)
        # Independent deterministic identity/oracle checks before measurement.
        assert _memprobe.nodes(ring) == nodes
        if _memprobe.walk(ring, nodes) != 0:
            raise ValueError("invalid cyclic successor structure")
        if _memprobe.scan(ring, 1) != nodes*(nodes-1)//2:
            raise ValueError("invalid contiguous scan checksum")
        rings[k] = ring
    records = []
    for epoch in range(epochs):
        ordered = sorted(rings, reverse=bool(epoch % 2))
        for k in ordered:
            ring = rings[k]
            nodes = _memprobe.nodes(ring)
            steps = max(nodes, target_accesses)
            rounds = max(1, math.ceil(target_accesses / nodes))
            scan_reads = nodes * rounds
            # Pre-touch: allocation and first-touch page faults are not timed.
            _memprobe.walk(ring, min(nodes, 100_000))
            _memprobe.scan(ring, 1)
            chase, scan = [], []
            stamp = time.monotonic_ns()
            for r in range(repeats):
                # AB/BA measurement order counterbalances immediate DVFS drift.
                if (epoch + r) % 2:
                    scan.append(_measure_ns(_memprobe.scan, ring, rounds))
                    chase.append(_measure_ns(_memprobe.walk, ring, steps))
                else:
                    chase.append(_measure_ns(_memprobe.walk, ring, steps))
                    scan.append(_measure_ns(_memprobe.scan, ring, rounds))
            chase_ns = statistics.median(chase) / steps
            bandwidth = scan_reads * 4e9 / statistics.median(scan)
            records.append(MemoryCell(epoch, nodes*4, stamp, chase_ns, bandwidth,
                                      steps, scan_reads, tuple(chase), tuple(scan)))
    return records


class _PerfEventAttr(ctypes.Structure):
    # perf_event_attr ABI v0 requires >= 64 bytes. No event is emulated.
    _fields_ = [("type", ctypes.c_uint32), ("size", ctypes.c_uint32),
                ("config", ctypes.c_uint64), ("sample_period", ctypes.c_uint64),
                ("sample_type", ctypes.c_uint64), ("read_format", ctypes.c_uint64),
                ("flags", ctypes.c_uint64), ("wakeup_events", ctypes.c_uint32),
                ("bp_type", ctypes.c_uint32), ("config1", ctypes.c_uint64)]


def _open_perf_event(type_id: int, config: int) -> int:
    arch = platform.machine().lower()
    numbers = {"x86_64": 298, "amd64": 298, "aarch64": 241, "arm64": 241}
    if not sys.platform.startswith("linux") or arch not in numbers:
        raise OSError(errno.ENOTSUP, "perf_event_open requires supported Linux architecture")
    attr = _PerfEventAttr(type=type_id, size=ctypes.sizeof(_PerfEventAttr),
                          config=config, flags=1 | (1 << 5) | (1 << 6))
    lib = ctypes.CDLL(None, use_errno=True)
    fd = lib.syscall(numbers[arch], ctypes.byref(attr), 0, -1, -1, 0)
    if fd < 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return fd


def try_hardware_counters(ring, *, steps: int = 2_000_000) -> dict:
    """Real optional Linux perf counters; return explicit NA, never estimated misses.

    Each event uses a separate run. They are not simultaneously sampled, so
    do NOT derive L2 misses by subtraction. L2 needs model-specific counters.
    """
    event_ids = {
        "l1d_read_misses": (3, (0 | (0 << 8) | (1 << 16))),
        "llc_read_misses": (3, (2 | (0 << 8) | (1 << 16))),
        "generic_cache_misses": (0, 3),
    }
    result = {"source": "linux_perf_event_open", "counter_runs_are_separate": True,
              "l2_read_misses": {"status": "UNAVAILABLE_MODEL_SPECIFIC_EVENT_REQUIRED", "count": None}}
    if _memprobe is None:
        return {**result, "status": "UNAVAILABLE_NATIVE_EXTENSION"}
    if steps < 1 or steps > 200_000_000:
        raise ValueError("invalid counter walk steps")
    if not sys.platform.startswith("linux"):
        return {**result, "status": "UNAVAILABLE_NOT_LINUX"}
    try:
        import fcntl
    except ImportError:
        return {**result, "status": "UNAVAILABLE_NOT_LINUX"}
    for name, (type_id, config) in event_ids.items():
        fd = None
        try:
            fd = _open_perf_event(type_id, config)
            fcntl.ioctl(fd, 0x2403, 0)  # PERF_EVENT_IOC_RESET
            fcntl.ioctl(fd, 0x2400, 0)  # PERF_EVENT_IOC_ENABLE
            _memprobe.walk(ring, steps)
            fcntl.ioctl(fd, 0x2401, 0)  # PERF_EVENT_IOC_DISABLE
            raw = os.read(fd, 8)
            if len(raw) != 8:
                raise OSError(errno.EIO, "incomplete perf counter read")
            result[name] = {"status": "MEASURED", "count": int.from_bytes(raw, sys.byteorder),
                            "steps": steps}
        except OSError as ex:
            result[name] = {"status": "UNAVAILABLE", "count": None,
                            "errno": ex.errno, "reason": str(ex)}
        finally:
            if fd is not None:
                os.close(fd)
    result["status"] = ("PARTIAL_OR_FULL_COUNTERS_MEASURED" if any(
        v.get("status") == "MEASURED" for v in result.values() if isinstance(v, dict)
    ) else "UNAVAILABLE_NO_HARDWARE_COUNTERS")
    return result


def make_report(cells: list[MemoryCell], caches: list[dict], *, counters=None) -> dict:
    if not cells:
        raise ValueError("empty memory evidence")
    return {
        "protocol": "ALEXPY-MEMORY-06/1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": {"os": platform.platform(), "python": sys.version.split()[0],
                 "machine": platform.machine(), "cpu_model_reported": cpu_description(),
                 "affinity_cpu_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                 "hardware_environment": "may_be_virtualized_or_shared"},
        "cache_topology_reported": caches,
        "cells": [asdict(c) for c in cells],
        "transitions": analyze_transitions(cells, caches),
        "hardware_events": counters if counters is not None else {"status": "NOT_REQUESTED", "counts": None},
        "evidence": "EXPLORATORY_NATIVE_MEMORY_ACCESS_ONLY_NO_AUTOPROMOTION",
        "notes": [
            "Pointer chase in native C measures latency/access for a random dependent index chain, not Python bytecode speed.",
            "Sequential scan reports effective indexed bytes per second, not DRAM bus bandwidth.",
            "Cache topology comes from OS sysfs; sizes are metadata, not measured cache occupancy.",
            "Cache boundary crossings and derivatives show correlation, never prove cache causality.",
            "Linux perf cache misses appear only when counter access succeeds; L2 needs a specific event.",
            "No tuning, candidate promotion, or automatic code-path changes are made from these data.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes-kib", nargs="+", type=int, default=[4,32,256,2048,16384])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--target-accesses", type=int, default=750_000)
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--hardware-counters", action="store_true")
    parser.add_argument("--json", type=Path, default=Path("memory-lab.json"))
    args = parser.parse_args(argv)
    records = collect_memory_lattice(sizes_kib=tuple(args.sizes_kib),
                                     epochs=args.epochs, repeats=args.repeats,
                                     target_accesses=args.target_accesses, seed=args.seed)
    cache = linux_cache_topology() if sys.platform == "linux" else []
    counters = None
    if args.hardware_counters:
        biggest = max(args.sizes_kib)
        ring = _memprobe.make_ring(biggest*1024//4, args.seed)
        counters = try_hardware_counters(ring, steps=max(args.target_accesses, biggest*1024//4))
    result = make_report(records, cache, counters=counters)
    args.json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"AlexPy memory lab: {len(records)} cells; {len(result['transitions']['spatial_edges'])} spatial edges; "
          f"{len(result['transitions']['temporal_edges'])} temporal edges")
    for cell in sorted((c for c in records if c.epoch == 0), key=lambda c:c.footprint_bytes):
        print(f"  {cell.footprint_bytes//1024:>6} KiB: random {cell.chase_ns_per_access:>8.2f} ns/access; "
              f"sequential {cell.sequential_effective_bytes_per_sec/1e9:>6.2f} GB/s effective")
    if counters is not None:
        print(f"Hardware counters: {counters['status']}")
    print(f"Evidence saved to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
