"""QA compact native C scratch: sparse/dense parity and transaction safety."""
import random
import unittest
from alexpy.qa_lab import QAEvent, PASS, FAIL, SKIP, MAX_DURATION
from alexpy.qa_native import NativeBatchQA
from alexpy.qa_batch import StrongPythonBatchQA

class TestCompactNativeBatch(unittest.TestCase):
    def test_wide_sparse_batches(self):
        suites=[i%17 for i in range(20000)]
        a,b=NativeBatchQA(suites,17),StrongPythonBatchQA(suites,17)
        rng=random.Random(174203)
        for tick in range(100):
            es=[QAEvent(rng.randrange(20000),PASS if tick%2 else FAIL,
                        rng.randrange(50000)) for _ in range(3)]
            self.assertEqual(a.apply_batch(es),b.apply_batch(es))
        self.assertEqual(a.snapshot(),b.snapshot())
        self.assertTrue(a.verify())

    def test_duplicates_more_events_than_tests(self):
        suites=[0,1,1]
        a,b=NativeBatchQA(suites,2),StrongPythonBatchQA(suites,2)
        es=[QAEvent(i%3,PASS if i%2 else FAIL,i) for i in range(2501)]
        es.extend((QAEvent(0,SKIP,0),QAEvent(1,PASS,MAX_DURATION)))
        self.assertEqual(a.apply_batch(es),b.apply_batch(es))
        self.assertEqual(a.snapshot(),b.snapshot())
        self.assertTrue(a.verify())

    def test_hash_collision(self):
        suites=[i%4 for i in range(65)]
        a,b=NativeBatchQA(suites,4),StrongPythonBatchQA(suites,4)
        for tick in range(30):
            es=[QAEvent(i,PASS if tick%2 else FAIL,tick+i) for i in (0,8,16)]
            es.append(QAEvent(8,PASS,tick+8))
            self.assertEqual(a.apply_batch(es),b.apply_batch(es))
        self.assertEqual(a.snapshot(),b.snapshot())

    def test_dense_switch_boundary(self):
        suites=[i%7 for i in range(2048)]
        a,b=NativeBatchQA(suites,7),StrongPythonBatchQA(suites,7)
        for length in (1024,1025,2048,4000):
            es=[QAEvent(i%2048,PASS if i%3 else FAIL,i+length) for i in range(length)]
            self.assertEqual(a.apply_batch(es),b.apply_batch(es))
            self.assertEqual(a.snapshot(),b.snapshot())
        snap,tick,version=a.snapshot(),a.tick,a.version
        es=[QAEvent(i,PASS,i) for i in range(1200)]
        es.extend((QAEvent(1,SKIP,1),QAEvent(1,PASS,3)))
        with self.assertRaises(ValueError):a.apply_batch(es)
        self.assertEqual((a.snapshot(),a.tick,a.version),(snap,tick,version))
        self.assertTrue(a.verify())

    def test_invalid_intermediate_rollback(self):
        a=NativeBatchQA([0,1,0,1],2)
        a.apply_batch([QAEvent(0,PASS,11)])
        prior=(a.snapshot(),a.tick,a.version,a.last_batch)
        for es in (
            [QAEvent(1,PASS,6),QAEvent(1,SKIP,1),QAEvent(1,PASS,7)],
            [QAEvent(1,PASS,6),QAEvent(1,PASS,MAX_DURATION+1)],
            [QAEvent(1,PASS,6),QAEvent(True,PASS,1)],
            [QAEvent(1,PASS,6),QAEvent(1,FAIL,-2)],
        ):
            with self.subTest(es=es),self.assertRaises((ValueError,IndexError,OverflowError)):
                a.apply_batch(es)
            self.assertEqual((a.snapshot(),a.tick,a.version,a.last_batch),prior)

    def test_empty_batch_reuse(self):
        a,b=NativeBatchQA([0,0,1],2),StrongPythonBatchQA([0,0,1],2)
        for _ in range(300):
            self.assertEqual(a.apply_batch([]),b.apply_batch([]))
        with self.assertRaises(ValueError):
            a.apply_batch([QAEvent(1,PASS,3),QAEvent(2,SKIP,5)])
        es=[QAEvent(1,PASS,MAX_DURATION),QAEvent(2,PASS,MAX_DURATION)]
        self.assertEqual(a.apply_batch(es),b.apply_batch(es))
        self.assertEqual(a.snapshot(),b.snapshot())
        self.assertTrue(a.verify())

if __name__=='__main__':unittest.main()
