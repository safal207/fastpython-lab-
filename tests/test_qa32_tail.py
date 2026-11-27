import unittest
from qa32_tail_latency import CASES, ORDERS, setup, worker, memory_worker

class QA32TailTests(unittest.TestCase):
    def test_orders_and_cases(self):
        self.assertEqual(len(set(ORDERS)), 6)
        self.assertEqual(len(set(x[2] for x in CASES)), 4)
    def test_fallback_is_semantically_identical(self):
        _, _, _, _, funcs = setup(CASES[0])
        self.assertEqual(len({f() for f in funcs.values()}), 1)
    def test_paired_raw_and_memory_schema(self):
        result = worker(0, 6, 0)
        self.assertEqual(sorted(result["order_codes"]), list(range(6)))
        for v in result["raw_ns"].values():
            self.assertEqual(len(v), 6)
            self.assertTrue(all(x > 0 for x in v))
        m = memory_worker(0, "dispatch")
        self.assertEqual(m["fingerprint"], result["fingerprint"])
        self.assertGreater(m["peak_traced_bytes"], 0)
