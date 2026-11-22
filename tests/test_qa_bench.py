import tempfile
import unittest
from pathlib import Path
from alexpy.qa_bench import Case, generate, replay, verify, worker_run, run, SMOKE_CASES


class TestQABench(unittest.TestCase):
    def test_generated_replay_mirrors(self):
        case=Case("unit", 23, 4, 92, 2, 3)
        suites,events=generate(case)
        reference=verify(case,suites,events)
        for route in ("strong_python", "alexpy_graph", "scan_witness"):
            self.assertEqual(replay(route,case,suites,events),reference)

    def test_reproducibility(self):
        case=SMOKE_CASES[0]
        self.assertEqual(generate(case),generate(case))

    def test_worker_smoke(self):
        report=worker_run(1,True)
        self.assertEqual(len(report["cases"]),2)
        for case in report["cases"]:
            self.assertEqual(len(case["pairs"]),1)
            self.assertGreater(case["python_peak_tracemalloc_bytes"],0)
            self.assertGreater(case["alexpy_peak_tracemalloc_bytes"],0)

    def test_process_isolation_schema(self):
        report=run(sessions=1,pairs=1,smoke=True)
        self.assertEqual(report["schema"],"ALEXPY-QA-LAB/1")
        self.assertEqual(len(report["summary"]),2)
        self.assertIn("workers",report)

    def test_reject_unknown_route(self):
        case=SMOKE_CASES[0]
        s,e=generate(case)
        with self.assertRaises(ValueError): replay("nonexistent",case,s,e)

    def test_bad_configuration(self):
        with self.assertRaises(ValueError): run(sessions=0)
        with self.assertRaises(ValueError): run(pairs=0)


if __name__ == "__main__": unittest.main()
