"""Cross-language invariants and exact-state checks for the C witness tree."""
import random
import unittest

from alexpy.qa_batch import _prepare
from alexpy.qa_lab import QAEvent, NOT_RUN, PASS, FAIL, SKIP, scan_reference, MAX_DURATION
from alexpy.native_transition_tree import CTreeNativeQA, WitnessTree
from alexpy.transition_tree import SetNativeQA, TreeNativeQA


@unittest.skipIf(WitnessTree is None, "native witness extension absent")
class TestNativeWitnessTree(unittest.TestCase):
    def test_exact_first_k_and_retries(self):
        for n in (1, 2, 7, 9, 64, 255, 4096):
            suites=[i%3 for i in range(n)]
            c=CTreeNativeQA(suites,3)
            s=SetNativeQA(suites,3)
            rng=random.Random(n)
            statuses=[NOT_RUN]*n
            durations=[0]*n
            for tick in range(31):
                batch=[]
                for _ in range(rng.randrange(0,37)):
                    i=rng.randrange(n)
                    st=rng.choice((PASS,FAIL,SKIP))
                    dur=0 if st==SKIP else rng.randrange(0,100000)
                    batch.append(QAEvent(i,st,dur))
                    statuses[i]=st
                    durations[i]=dur
                self.assertEqual(c.apply_batch(batch),s.apply_batch(batch))
                self.assertEqual(c.snapshot(),s.snapshot())
                self.assertEqual(c.witnesses(),s.witnesses())
                self.assertTrue(c.verify())
                self.assertEqual(c.snapshot(),scan_reference(suites,3,statuses,durations))
            self.assertTrue(c.engine.native)

    def test_invalid_mid_batch_is_atomic(self):
        c=CTreeNativeQA([0,1,0],2)
        c.apply_batch([QAEvent(1,FAIL,MAX_DURATION)])
        before=(c.witnesses(),c.snapshot(),c.engine.tick,c.engine.version,c.index.dump())
        for bad in (
            [QAEvent(2,FAIL,17), QAEvent(1,SKIP,10), QAEvent(0,PASS,20)],
            [QAEvent(2,FAIL,17), QAEvent(12,PASS,20)],
            [QAEvent(2,FAIL,17), QAEvent(1,False,20)],
            [QAEvent(2,FAIL,17), object()],
        ):
            with self.assertRaises((TypeError, ValueError, IndexError)):
                c.apply_batch(bad)
            self.assertEqual(before,(c.witnesses(),c.snapshot(),c.engine.tick,c.engine.version,c.index.dump()))
        self.assertTrue(c.verify())

    def test_native_commit_stage_guards(self):
        w=WitnessTree(5,3)
        for bad in ({0:(2,)}, {-1:(2,0)}, {0:(0,0)}, {5:(2,0)}, {0:([2],0)}):
            with self.assertRaises((ValueError,TypeError,IndexError)):
                w.stage(bad)
            self.assertEqual(w.witnesses(),())
        for bad in (b"x", None, object()):
            with self.assertRaises((ValueError,TypeError)):
                w.commit(bad)
            self.assertEqual(w.witnesses(),())
        staged=w.stage({1:(FAIL,123),4:(FAIL,4)})
        self.assertEqual(w.witnesses(),())
        w.commit(staged)
        self.assertEqual(w.witnesses(),(1,4))
        self.assertEqual(w.failed_count,2)
        self.assertTrue(w.verify())
        w.commit(w.stage({1:(PASS,34)}))
        self.assertEqual(w.witnesses(),(4,))

    def test_uninitialized_class_fails_closed(self):
        c = WitnessTree.__new__(WitnessTree)
        for method in (c.witnesses,c.verify,c.dump):
            with self.assertRaises(RuntimeError):method()
        with self.assertRaises(RuntimeError):c.stage({})
        with self.assertRaises(RuntimeError):c.commit(b"")

    def test_dimension_and_bool_guards(self):
        for n,k in ((0,4),(True,4),(3,True),(3,0),(-1,4)):
            with self.assertRaises((ValueError, OverflowError)):
                WitnessTree(n,k)

    def test_non_powers_of_two_and_limit(self):
        n=257
        c=CTreeNativeQA([0]*n,1,limit=2)
        c.apply_batch([QAEvent(i,FAIL,i) for i in (256,128,0,44,9)])
        self.assertEqual(c.witnesses(),(0,9))
        c.apply_batch([QAEvent(0,SKIP,0),QAEvent(9,PASS,1)])
        self.assertEqual(c.witnesses(),(44,128))
        self.assertTrue(c.verify())

    def test_noop_and_long_duration(self):
        c=CTreeNativeQA([0,0,0],1)
        for batch in ([],[QAEvent(0,FAIL,MAX_DURATION),QAEvent(2,FAIL,MAX_DURATION)],
                      [QAEvent(0,FAIL,MAX_DURATION)],[]):
            c.apply_batch(batch)
            self.assertTrue(c.verify())
        self.assertEqual(c.snapshot().total[4],2*MAX_DURATION)
