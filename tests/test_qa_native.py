"""Native QA batch correctness and atomicity; independent strong-Python mirror."""
import random
import unittest

from alexpy.qa_batch import QABatchTransition, StrongPythonBatchQA, AlexPyBatchQAGraph
from alexpy.qa_native import NativeBatchQA, _QAState
from alexpy.qa_lab import QAEvent, PASS, FAIL, SKIP, NOT_RUN, MAX_DURATION, scan_reference


class TestNativeBatchQA(unittest.TestCase):
    def engines(self):
        suites=[0,1,2,0,3,2,1,1]
        return (NativeBatchQA(suites,4),StrongPythonBatchQA(suites,4),AlexPyBatchQAGraph(suites,4))

    def test_compiled_module_available(self):
        self.assertIsNotNone(_QAState)
        self.assertTrue(self.engines()[0].native)

    def test_initial_snapshot(self):
        n,p,a=self.engines()
        self.assertEqual(n.snapshot(),p.snapshot())
        self.assertEqual(n.snapshot(),a.snapshot())
        self.assertEqual(n.snapshot().total,(8,0,0,0,0))

    def test_duplicate_ids_and_transaction(self):
        n,p,a=self.engines()
        for batch in ([QAEvent(0,PASS,20),QAEvent(3,FAIL,70),QAEvent(0,SKIP,0),QAEvent(0,PASS,4)],
                      [],[QAEvent(0,PASS,20),QAEvent(0,PASS,4)],
                      [QAEvent(0,PASS,7),QAEvent(0,PASS,4)]):
            ns=[obj.apply_batch(batch) for obj in (n,p,a)]
            self.assertEqual(ns[0],ns[1]);self.assertEqual(ns[1],ns[2])
            self.assertEqual(n.snapshot(),p.snapshot());self.assertEqual(n.snapshot(),a.snapshot())
            self.assertTrue(n.verify())
        self.assertEqual(n.tick,4);self.assertEqual(n.version,1)

    def test_status_only_version(self):
        n,p,a=self.engines()
        for batch in ([QAEvent(0,PASS,0)], [QAEvent(0,FAIL,0)], [QAEvent(0,FAIL,0)]):
            self.assertEqual(n.apply_batch(batch),p.apply_batch(batch))
        self.assertEqual(n.version,2)

    def test_large_duration_aggregates_arbitrary_precision(self):
        n,p,a=self.engines()
        events=[QAEvent(i,PASS,MAX_DURATION) for i in range(8)]
        for e in (n,p,a):e.apply_batch(events)
        self.assertEqual(n.snapshot(),p.snapshot())
        self.assertEqual(n.snapshot().total[4],8*MAX_DURATION)
        n.apply_batch([QAEvent(3,SKIP,0)])
        p.apply_batch([QAEvent(3,SKIP,0)])
        self.assertEqual(n.snapshot(),p.snapshot())
        self.assertEqual(n.snapshot().total[4],7*MAX_DURATION)
        self.assertTrue(n.verify())

    def test_invalid_intermediate_never_commits(self):
        n,p,a=self.engines()
        n.apply_batch([QAEvent(1,PASS,15)])
        invalid=[
            [QAEvent(0,PASS,2),QAEvent(0,SKIP,1),QAEvent(0,PASS,3)],
            [QAEvent(0,PASS,2),QAEvent(True,PASS,8)],
            [QAEvent(0,PASS,2),QAEvent(8,FAIL,8)],
            [QAEvent(0,PASS,2),QAEvent(2,FAIL,-1)],
            [QAEvent(0,PASS,2),QAEvent(2,FAIL,MAX_DURATION+1)],
            [QAEvent(0,PASS,2),object()],
            [QAEvent(0,PASS,2),QAEvent(2,NOT_RUN,0)],
            [QAEvent(0,PASS,2),QAEvent(2,FAIL,True)],
            (x for x in [QAEvent(0,PASS,2)]),
            'not a list',
        ]
        for batch in invalid:
            with self.subTest(batch=str(batch)[:50]):
                snap=n.snapshot();state=(n.tick,n.version,n.last_batch)
                with self.assertRaises((TypeError,ValueError,IndexError,OverflowError)):
                    n.apply_batch(batch)
                self.assertEqual(snap,n.snapshot())
                self.assertEqual(state,(n.tick,n.version,n.last_batch))
                self.assertTrue(n.verify())

    def test_explicit_python_fallback_is_not_reported_as_native(self):
        import alexpy.qa_native as m
        original=m._QAState
        try:
            m._QAState=None
            obj=m.NativeBatchQA([0,1,0],2)
            self.assertFalse(obj.native)
            self.assertEqual(obj.apply_batch([QAEvent(0,PASS,5)]).changed_indices,(0,))
            self.assertEqual(obj.snapshot().total,(2,1,0,0,5))
            self.assertTrue(obj.verify())
        finally:
            m._QAState=original

    def test_layout_invalid(self):
        for suites,count in (([],1),([0],0),([True],1),([3],2),([0],True),(1,1)):
            with self.subTest(suites=suites,count=count),self.assertRaises((TypeError,ValueError,OverflowError)):
                NativeBatchQA(suites,count)

    def test_randomized_mirrors(self):
        rng=random.Random(70241)
        for size,steps in ((8,180),(256,140)):
            suites=[i%11 for i in range(size)]
            engines=[NativeBatchQA(suites,11),StrongPythonBatchQA(suites,11),AlexPyBatchQAGraph(suites,11)]
            state=[NOT_RUN]*size;durations=[0]*size
            for _ in range(steps):
                batch=[]
                for j in range(rng.randrange(0,128)):
                    idx=rng.randrange(size);status=rng.choice([PASS,FAIL,SKIP]);dur=rng.randrange(1000000) if status!=SKIP else 0
                    batch.append(QAEvent(idx,status,dur))
                    state[idx]=status;durations[idx]=dur
                transitions=[obj.apply_batch(batch) for obj in engines]
                self.assertEqual(len(set(transitions)),1)
                expected=scan_reference(suites,11,state,durations)
                self.assertEqual(engines[0].snapshot(),expected)
                self.assertEqual(engines[1].snapshot(),expected)
                self.assertEqual(engines[2].snapshot(),expected)
            self.assertTrue(engines[0].verify())


if __name__=='__main__':unittest.main()
