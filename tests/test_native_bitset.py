"""Exact bitmap witness index, transactional semantics and non-power-of-two edges."""
import random
import unittest

from alexpy.native_bitset import BitsetNativeQA, WitnessBits
from alexpy.native_transition_tree import CTreeNativeQA
from alexpy.transition_tree import SetNativeQA
from alexpy.qa_lab import QAEvent, NOT_RUN, PASS, FAIL, SKIP, scan_reference, MAX_DURATION


@unittest.skipIf(WitnessBits is None, "native bitset extension not built")
class TestNativeBitset(unittest.TestCase):
    def test_crosscheck_random_batches_and_checkpoints(self):
        for n in (1, 2, 7, 63, 64, 65, 127, 128, 129, 257, 4097):
            suites = [i % 5 for i in range(n)]
            engines = [cls(suites, 5, limit=4) for cls in
                       (BitsetNativeQA, CTreeNativeQA, SetNativeQA)]
            statuses = [NOT_RUN] * n
            durations = [0] * n
            rng = random.Random(n + 8732)
            for tick in range(20):
                batch = []
                for _ in range(rng.randrange(0, min(n, 79) + 1)):
                    idx = rng.randrange(n)
                    st = rng.choice((PASS, FAIL, SKIP))
                    ms = 0 if st == SKIP else rng.randrange(10001)
                    batch.append(QAEvent(idx, st, ms))
                    statuses[idx], durations[idx] = st, ms
                transitions = [e.apply_batch(batch) for e in engines]
                self.assertEqual(transitions, [transitions[0]] * 3)
                expected = scan_reference(suites, 5, statuses, durations)
                witnesses = tuple(i for i, st in enumerate(statuses) if st == FAIL)[:4]
                for e in engines:
                    self.assertEqual(e.snapshot(), expected)
                    self.assertEqual(e.witnesses(), witnesses)
                    self.assertTrue(e.verify())

    def test_uninitialized_and_argument_guards(self):
        for n, limit in ((0, 4), (True, 4), (3, True), (-1, 4), (4, 0)):
            with self.assertRaises((ValueError, OverflowError)):
                WitnessBits(n, limit)
        bare = WitnessBits.__new__(WitnessBits)
        for f, arg in ((bare.stage, {}), (bare.commit, b"")):
            with self.assertRaises(RuntimeError): f(arg)
        for f in (bare.witnesses, bare.verify, bare.dump):
            with self.assertRaises(RuntimeError): f()

    def test_prepare_commit_preflight_and_order(self):
        b = WitnessBits(130, 3)
        for bad in ({-1:(FAIL, 0)}, {130:(FAIL, 0)}, {1:(0, 0)},
                    {1:(FAIL,)}, {1:([FAIL], 0)}):
            with self.assertRaises((ValueError, TypeError, IndexError)):
                b.stage(bad)
            self.assertEqual(b.witnesses(), ())
        for bad in (None, object(), b"invalidlen"):
            with self.assertRaises((ValueError, TypeError)):
                b.commit(bad)
        staged = b.stage({129:(FAIL, 3), 64:(FAIL, 0), 0:(FAIL, 1), 65:(FAIL, 5)})
        self.assertEqual(b.failed_count, 0)
        b.commit(staged)
        self.assertEqual(b.witnesses(), (0, 64, 65))
        self.assertEqual(b.failed_count, 4)
        self.assertTrue(b.verify())
        b.commit(b.stage({0:(PASS, 7), 65:(SKIP, 0), 129:(FAIL, 2)}))
        self.assertEqual(b.witnesses(), (64, 129))
        self.assertTrue(b.verify())

    def test_atomic_invalid_mid_batch(self):
        eng = BitsetNativeQA([0, 1, 0, 1, 0], 2)
        eng.apply_batch([QAEvent(4, FAIL, MAX_DURATION)])
        snapshot = (eng.snapshot(), eng.witnesses(), eng.engine.tick,
                    eng.engine.version, eng.index.dump())
        bad_batches = (
            [QAEvent(1, FAIL, 7), QAEvent(2, SKIP, 17)],
            [QAEvent(1, FAIL, 7), QAEvent(99, FAIL, 0)],
            [QAEvent(4, PASS, 0), object()],
            [QAEvent(1, FAIL, 7), QAEvent(2, False, 0)],
        )
        for bad in bad_batches:
            with self.assertRaises((ValueError, IndexError, TypeError)):
                eng.apply_batch(bad)
            self.assertEqual(snapshot, (eng.snapshot(), eng.witnesses(),
                                         eng.engine.tick, eng.engine.version,
                                         eng.index.dump()))
            self.assertTrue(eng.verify())

    def test_sparse_high_ids_and_empty(self):
        eng = BitsetNativeQA([0] * 16385, 1, limit=1)
        self.assertEqual(eng.witnesses(), ())
        eng.apply_batch([QAEvent(16384, FAIL, 4)])
        self.assertEqual(eng.witnesses(), (16384,))
        eng.apply_batch([QAEvent(64, FAIL, 3), QAEvent(16384, PASS, 4)])
        self.assertEqual(eng.witnesses(), (64,))
        eng.apply_batch([])
        self.assertTrue(eng.verify())

    def test_repeated_all_fail_and_all_pass(self):
        n = 131
        eng = BitsetNativeQA([0] * n, 1)
        for status in (FAIL, PASS, FAIL, SKIP):
            eng.apply_batch([QAEvent(i, status, 0) for i in range(n)])
            self.assertEqual(eng.index.failed_count, n if status == FAIL else 0)
            self.assertTrue(eng.verify())
