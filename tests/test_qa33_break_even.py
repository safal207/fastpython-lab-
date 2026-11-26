import unittest
from qa33_break_even import BASE, ALT, PLANS, choose, fixture, measure, select

class QA33Tests(unittest.TestCase):
    def test_exact_plan_and_unknown_fallback(self):
        self.assertEqual(len(set(PLANS)), 12)
        self.assertEqual(choose({PLANS[0]: ALT}, PLANS[0]), ALT)
        self.assertEqual(choose({PLANS[0]: ALT}, (128, 32, 2, 1)), BASE)
    def test_conservative_gates(self):
        a = dict(median_ns=80, p95_ns=100, p99_ns=120, peak_bytes=100)
        b = dict(median_ns=100, p95_ns=120, p99_ns=150, peak_bytes=100)
        row = {ALT: a, BASE: b}
        self.assertEqual(select([row]*3, 5), ALT)
        self.assertEqual(select([row]*2, 5), BASE)
        self.assertEqual(select([row]*3, 20), BASE)
        for field in ("p95_ns", "p99_ns", "peak_bytes"):
            bad = {ALT: dict(a, **{field: 1000}), BASE: b}
            self.assertEqual(select([row, row, bad], 5), BASE)
    def test_fixture_horizon_and_fresh_seed(self):
        plan = (128, 32, 8, 8)
        l, g = fixture(plan, 1)
        self.assertEqual(len(l), 128)
        self.assertEqual(len(g), 8)
        self.assertTrue(all(len(x) == 32 for x in g))
        self.assertNotEqual(g, fixture(plan, 2)[1])
    def test_smoke_exact_reports(self):
        row = measure((128, 32, 8, 8), 339999, 6, 0, "holdout", ALT)
        self.assertTrue(row["verified"])
        self.assertEqual(row["reports"], 1)
        self.assertEqual(sorted(row["order_codes"]), list(range(6)))
        for result in row["results"].values():
            self.assertEqual(len(result["samples_ns"]), 6)
