import random
import unittest

from alexpy.qa_lab import (
    QAEvent, StrongPythonQA, AlexPyQAGraph, scan_reference,
    PASS, FAIL, SKIP, NOT_RUN, MAX_DURATION,
)


class TestQALab(unittest.TestCase):
    def setUp(self):
        self.suites = [0, 0, 1, 2, 2, 1]

    def engines(self):
        return StrongPythonQA(self.suites, 4), AlexPyQAGraph(self.suites, 4)

    def test_initial(self):
        a,b=self.engines()
        self.assertEqual(a.snapshot(),b.snapshot())
        self.assertEqual(b.snapshot().total,(6,0,0,0,0))
        self.assertEqual(b.snapshot().suites[3],(0,0,0,0,0))
        self.assertTrue(b.verify())

    def test_updates_retries_and_sparse_invalidation(self):
        a,b=self.engines()
        for event, affected in [(QAEvent(1,PASS,12),(0,)),
                                (QAEvent(1,FAIL,40),(0,)),
                                (QAEvent(1,FAIL,40),()),
                                (QAEvent(5,SKIP,0),(1,)),
                                (QAEvent(1,PASS,8),(0,))]:
            a.apply(event);b.apply(event)
            self.assertEqual(b.last_affected_suites,affected)
            self.assertEqual(a.snapshot(),b.snapshot())
            self.assertTrue(b.verify())
        self.assertEqual(b.snapshot().total,(4,1,0,1,8))
        self.assertEqual(b.tick,5)
        self.assertEqual(b.version,4)
        self.assertEqual(b._buffer.aggregates()[0],8)

    def test_different_duration_same_status(self):
        a,b=self.engines()
        for event in (QAEvent(0,PASS,5),QAEvent(0,PASS,25)):
            a.apply(event);b.apply(event)
        self.assertEqual(b.snapshot().total,(5,1,0,0,25))
        self.assertEqual(a.snapshot(),b.snapshot())
        self.assertTrue(b.verify())

    def test_reject_atomicity(self):
        for engine in self.engines():
            engine.apply(QAEvent(0,PASS,30))
            prev=(engine.snapshot(),engine.tick,engine.version)
            for bad in (QAEvent(-1,PASS,3),QAEvent(True,PASS,3),
                        QAEvent(0,True,3),QAEvent(0,NOT_RUN,0),
                        QAEvent(1,SKIP,20),QAEvent(0,FAIL,-1),
                        QAEvent(0,FAIL,MAX_DURATION+1),
                        QAEvent(0,FAIL,1.0), (0,PASS,9)):
                with self.subTest(bad=bad):
                    with self.assertRaises((ValueError,TypeError,IndexError)):
                        engine.apply(bad)
                    self.assertEqual((engine.snapshot(),engine.tick,engine.version),prev)

    def test_layout_strict(self):
        for typ in (StrongPythonQA,AlexPyQAGraph):
            with self.assertRaises(ValueError): typ([],1)
            with self.assertRaises(ValueError): typ([True],1)
            with self.assertRaises(ValueError): typ([1],1)
            with self.assertRaises(ValueError): typ([0],True)

    def test_source_alias_copied(self):
        suites=[0,1,0]
        graph=AlexPyQAGraph(suites,2)
        suites[0]=1
        self.assertEqual(graph.suites,(0,1,0))
        self.assertTrue(graph.verify())

    def test_big_int_aggregate_does_not_overflow(self):
        a=StrongPythonQA(self.suites,4)
        b=AlexPyQAGraph(self.suites,4)
        for j in range(len(self.suites)):
            event=QAEvent(j,PASS,MAX_DURATION)
            a.apply(event);b.apply(event)
        self.assertEqual(a.snapshot(),b.snapshot())
        self.assertTrue(b.verify())
        self.assertEqual(b.snapshot().total[4],MAX_DURATION*6)

    def test_randomized_mirror(self):
        rng=random.Random(271828)
        for n,suites_count in ((1,1),(17,5),(257,9)):
            suites=[rng.randrange(suites_count) for _ in range(n)]
            strong=StrongPythonQA(suites,suites_count)
            graph=AlexPyQAGraph(suites,suites_count)
            statuses=[NOT_RUN]*n
            durations=[0]*n
            for k in range(1600):
                i=rng.randrange(n)
                state=rng.choice((PASS,FAIL,SKIP))
                duration=0 if state==SKIP else rng.randrange(60000)
                event=QAEvent(i,state,duration)
                strong.apply(event)
                graph.apply(event)
                statuses[i],durations[i]=state,duration
                if k%23==0:
                    expected=scan_reference(suites,suites_count,statuses,durations)
                    self.assertEqual(graph.snapshot(),expected)
                    self.assertEqual(strong.snapshot(),expected)
                    self.assertTrue(graph.verify())
            self.assertEqual(graph.snapshot(),strong.snapshot())

    def test_empty_suite_and_trace_boundary(self):
        engine=AlexPyQAGraph([0,2,2],4)
        engine.apply(QAEvent(1,FAIL,42))
        self.assertEqual(engine.last_affected_suites,(2,))
        self.assertEqual(engine.snapshot().suites[1],(0,0,0,0,0))
        self.assertEqual(engine.snapshot().suites[3],(0,0,0,0,0))
        self.assertEqual(engine._buffer.tick,1)


if __name__ == "__main__": unittest.main()
