import unittest

from alexpy.qa_batch_bench import (Case, FULL_CASES, generate, replay,
                                   verify, run, worker, ROUTES)


class TestQABatchBenchmark(unittest.TestCase):
    def test_generate_reproducible(self):
        c = Case("x", 13, 3, 90, 7, 3, 2)
        self.assertEqual(generate(c), generate(c))
        self.assertTrue(all(e.test_id < 7 for e in generate(c)[1]))

    def test_semantics_and_digest_all_four_routes(self):
        c=Case("mirror", 16, 4, 97, 3, 2, 2)
        suites, events=generate(c)
        for b in (1, 2, 8, 64, 256):
            witness=verify(c,b,suites,events)
            self.assertEqual({replay(r,c,b,suites,events) for r in ROUTES},{witness})

    def test_bounded_smoke(self):
        r=worker(pairs=2,smoke=True,batch_sizes=(1,8))
        self.assertEqual(len(r["cases"]),4)
        self.assertEqual(len(r["cases"][0]["pairs"]),2)
        self.assertTrue(all(x["median_speedup_vs_strong_batch"]>0 for x in r["cases"]))

    def test_validation(self):
        for opts in ({"sessions":0},{"pairs":0},{"batch_sizes":()},{"batch_sizes":(0,)},
                     {"batch_sizes":(True,)}):
            with self.subTest(opts=opts),self.assertRaises(ValueError):run(**opts)


if __name__ == "__main__":unittest.main()
