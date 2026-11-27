"""M21 atomic batch, retry coalescing and mirror tests."""
import random
import unittest

from alexpy.qa_lab import (AlexPyQAGraph, StrongPythonQA, QAEvent,
                           scan_reference, PASS, FAIL, SKIP, NOT_RUN,
                           MAX_DURATION)
from alexpy.qa_batch import AlexPyBatchQAGraph, StrongPythonBatchQA


class TestQABatches(unittest.TestCase):
    suites = [0, 1, 0, 2, 2, 1, 0]

    def engines(self):
        return (StrongPythonBatchQA(self.suites, 4),
                AlexPyBatchQAGraph(self.suites, 4))

    def test_final_retry_wins_and_events_not_counted_as_ticks(self):
        a, b = self.engines()
        events = (QAEvent(0, PASS, 20), QAEvent(2, FAIL, 70),
                  QAEvent(0, SKIP, 0), QAEvent(0, PASS, 4))
        for engine in (a, b):
            transition = engine.apply_batch(events)
            self.assertEqual(transition.tick, 1)
            self.assertEqual(transition.version, 1)
            self.assertEqual(transition.received_events, 4)
            self.assertEqual(transition.distinct_test_ids, 2)
            self.assertEqual(transition.changed_indices, (0, 2))
            self.assertEqual(transition.affected_suites, (0,))
        self.assertEqual(a.snapshot(), b.snapshot())
        self.assertEqual(a.snapshot().total, (5, 1, 1, 0, 74))
        self.assertTrue(b.verify())
        self.assertEqual(b._buffer.tick, 1)

    def test_validation_is_atomic_even_when_overwritten(self):
        invalid_batches = [
            [QAEvent(0, PASS, 20), QAEvent(2, FAIL, 70), QAEvent(8, PASS, 1)],
            [QAEvent(0, PASS, 20), QAEvent(0, SKIP, 8), QAEvent(0, PASS, 1)],
            [QAEvent(1, PASS, 8), QAEvent(True, FAIL, 8)],
            [QAEvent(0, PASS, 7), QAEvent(1, PASS, MAX_DURATION + 1)],
            [QAEvent(0, PASS, 7), object()],
            (QAEvent(0, PASS, 7),) + (QAEvent(0, NOT_RUN, 0),),
            "not a batch",
            (event for event in (QAEvent(0, PASS, 2),)),
        ]
        for engine in self.engines():
            engine.apply_batch([QAEvent(0, PASS, 10)])
            for batch in invalid_batches:
                with self.subTest(engine=type(engine).__name__, batch=str(batch)[:30]):
                    before = (engine.snapshot(), engine.tick, engine.version,
                              engine.last_batch)
                    if isinstance(engine, AlexPyBatchQAGraph):
                        before += (engine._buffer.tick, engine._buffer.version,
                                   engine._buffer.to_list())
                    with self.assertRaises((ValueError, TypeError, IndexError, OverflowError)):
                        engine.apply_batch(batch)
                    after = (engine.snapshot(), engine.tick, engine.version,
                             engine.last_batch)
                    if isinstance(engine, AlexPyBatchQAGraph):
                        after += (engine._buffer.tick, engine._buffer.version,
                                  engine._buffer.to_list())
                        self.assertTrue(engine.verify())
                    self.assertEqual(before, after)

    def test_empty_and_noop_and_status_only(self):
        a,b = self.engines()
        for batch in ([], [QAEvent(0, PASS, 0)],
                      [QAEvent(0, PASS, 0), QAEvent(0, PASS, 0)],
                      [QAEvent(0, PASS, 1), QAEvent(0, PASS, 0)]):
            ta = a.apply_batch(batch)
            tb = b.apply_batch(batch)
            self.assertEqual(ta,tb)
            self.assertEqual(a.snapshot(),b.snapshot())
            self.assertTrue(b.verify())
        self.assertEqual(a.tick, 4)
        self.assertEqual(a.version, 1)
        self.assertEqual(b._buffer.tick, 4)
        # Status-only update does not increment OwnedQBuffer data version.
        self.assertEqual(b._buffer.version, 0)

    def test_big_int_overflow_safe_aggregate(self):
        a,b = self.engines()
        batch = [QAEvent(i, PASS, MAX_DURATION) for i in range(len(self.suites))]
        for engine in (a,b): engine.apply_batch(batch)
        self.assertEqual(a.snapshot(),b.snapshot())
        self.assertEqual(b.snapshot().total[4], len(self.suites)*MAX_DURATION)
        self.assertTrue(b.verify())

    def test_aborted_batch_preserves_prior_trace(self):
        a,b = self.engines()
        for e in (a,b):e.apply_batch([QAEvent(0,PASS,11)])
        prev = b.last_affected_suites
        with self.assertRaises(IndexError):
            b.apply_batch([QAEvent(1,PASS,2),QAEvent(999,PASS,2)])
        self.assertEqual(b.last_affected_suites,prev)
        self.assertTrue(b.verify())

    def test_randomized_mirrors_with_batch_boundaries(self):
        rng=random.Random(203812)
        for n, batches in ((7,150),(64,75)):
            suites=[(i*7+i//3)%5 for i in range(n)]
            a=StrongPythonBatchQA(suites,5)
            b=AlexPyBatchQAGraph(suites,5)
            sa=StrongPythonQA(suites,5)
            sb=AlexPyQAGraph(suites,5)
            states=[NOT_RUN]*n
            times=[0]*n
            for step in range(batches):
                batch=[]
                for _ in range(rng.randrange(0, 24)):
                    idx=rng.randrange(n)
                    status=rng.choice((PASS,FAIL,SKIP))
                    duration=0 if status==SKIP else rng.randrange(0,100001)
                    event=QAEvent(idx,status,duration)
                    batch.append(event)
                    sa.apply(event);sb.apply(event)
                    states[idx]=status;times[idx]=duration
                pa=a.apply_batch(batch);pb=b.apply_batch(tuple(batch))
                with self.subTest(n=n,step=step):
                    self.assertEqual(pa,pb)
                    expected=scan_reference(suites,5,states,times)
                    self.assertEqual(a.snapshot(),expected)
                    self.assertEqual(b.snapshot(),expected)
                    self.assertEqual(sa.snapshot(),expected)
                    self.assertEqual(sb.snapshot(),expected)
                    self.assertTrue(b.verify())


if __name__ == "__main__":
    unittest.main()
