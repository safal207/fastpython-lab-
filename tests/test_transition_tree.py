"""Independent scan and strong-set mirrors for the Bardo-inspired tree."""
import random
import unittest

from alexpy.transition_tree import TransitionTree, TreeNativeQA, SetNativeQA
from alexpy.qa_lab import QAEvent, NOT_RUN, PASS, FAIL, SKIP, MAX_DURATION, scan_reference


class TestTransitionTree(unittest.TestCase):
    def test_staging_does_not_mutate(self):
        tree = TransitionTree(5)
        before = tree.root, tree.statuses[:], tree.durations[:]
        staged = tree.prepare({0:(FAIL,11), 4:(PASS,42)})
        self.assertEqual((tree.root,tree.statuses,tree.durations),before)
        tree.commit(staged)
        self.assertEqual(tree.root.counts, (3,1,1,0,53))
        self.assertEqual(tree.root.witnesses, (0,))
        self.assertTrue(tree.verify())

    def test_single_leaf_and_large_integer_sum(self):
        t=TransitionTree(1)
        t.commit(t.prepare({0:(FAIL,MAX_DURATION)}))
        self.assertEqual(t.root.counts, (0,0,1,0,MAX_DURATION))
        self.assertTrue(t.verify())

    def test_exact_earliest_witnesses(self):
        t=TransitionTree(12,3)
        t.commit(t.prepare({i:(FAIL,i) for i in (9,5,1,7,2,0)}))
        self.assertEqual(t.root.witnesses,(0,1,2))
        t.commit(t.prepare({1:(PASS,7), 0:(SKIP,0)}))
        self.assertEqual(t.root.witnesses,(2,5,7))
        self.assertTrue(t.verify())

    def test_randomized_with_native_and_strong_set(self):
        rng=random.Random(45617)
        for n in (1,5,31,128):
            suites=[i%7 for i in range(n)]
            tree=TreeNativeQA(suites,7,limit=4)
            control=SetNativeQA(suites,7,limit=4)
            status=[NOT_RUN]*n
            durations=[0]*n
            for tick in range(70):
                batch=[]
                for _ in range(rng.randrange(0,21)):
                    idx=rng.randrange(n)
                    st=rng.choice((PASS,FAIL,SKIP))
                    dur=0 if st==SKIP else rng.randrange(0,100000)
                    batch.append(QAEvent(idx,st,dur))
                    status[idx]=st
                    durations[idx]=dur
                self.assertEqual(tree.apply_batch(batch),control.apply_batch(batch))
                self.assertEqual(tree.snapshot(), control.snapshot())
                self.assertEqual(tree.witnesses(),control.witnesses())
                if tick % 17==0:
                    self.assertEqual(tree.snapshot(),scan_reference(suites,7,status,durations))
                    self.assertTrue(tree.verify())
                    self.assertTrue(control.verify())
            self.assertTrue(tree.verify())

    def test_errors_do_not_change_tree_or_native(self):
        tree=TreeNativeQA([0,1,0,1],2)
        tree.apply_batch([QAEvent(2,FAIL,44)])
        before=(tree.snapshot(),tree.witnesses(),tree.tree.root,tree.engine.tick,tree.engine.version)
        invalid=[QAEvent(0,PASS,1),QAEvent(2,SKIP,19),QAEvent(2,FAIL,10)]
        with self.assertRaises(ValueError): tree.apply_batch(invalid)
        self.assertEqual(before,(tree.snapshot(),tree.witnesses(),tree.tree.root,tree.engine.tick,tree.engine.version))
        with self.assertRaises(TypeError):tree.apply_batch((x for x in invalid))
        self.assertTrue(tree.verify())

    def test_noop_batch_and_duplicate(self):
        tree=TreeNativeQA([0,1,0],2)
        ctrl=SetNativeQA([0,1,0],2)
        for batch in ([], [QAEvent(1,FAIL,10),QAEvent(1,PASS,2),QAEvent(1,FAIL,10)],
                      [QAEvent(1,FAIL,10)], []):
            self.assertEqual(tree.apply_batch(batch),ctrl.apply_batch(batch))
            self.assertEqual(tree.witnesses(),ctrl.witnesses())
        self.assertEqual(tree.witnesses(),(1,))

    def test_bad_constructor(self):
        for n,k in ((0,2),(True,2),(2,0),(2,True)):
            with self.assertRaises(ValueError):TransitionTree(n,k)
