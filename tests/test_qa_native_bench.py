import unittest
from alexpy.qa_native_bench import Case,ROUTES,generate,replay,verify,run,worker

class TestQANativeBench(unittest.TestCase):
    def test_digests(self):
        case=Case('mirror',32,4,106,11,3,2)
        suites,events=generate(case)
        for size in (1,8,64,256):
            digest=verify(case,size,suites,events)
            self.assertEqual({replay(r,case,size,suites,events) for r in ROUTES},{digest})

    def test_small_smoke(self):
        x=worker(2,True,(1,8))
        self.assertEqual(len(x['cells']),4)
        self.assertTrue(x['native'])
        self.assertTrue(all(c['median_vs_python']>0 for c in x['cells']))

    def test_reject_invalid_config(self):
        for kw in ({'sessions':0},{'pairs':0},{'batches':(0,)},{'batches':()},{'batches':(True,)}):
            with self.subTest(kw=kw),self.assertRaises(ValueError):run(**kw)

if __name__=='__main__':unittest.main()
