"""Independent range-query witnesses, edge cases, and API fail-closed tests."""
import random
import unittest

from alexpy.range_routes import (ROUTES, Direct, Fenwick, RangeWorkload,
                                 checked_routes, make_backend, make_case, run_workflow)


class TestRangeRoutes(unittest.TestCase):
    def test_random_patches_and_windows(self):
        rng=random.Random(12016)
        for n in (1,2,3,8,31,64,257):
            initial=[rng.randrange(-(10**15),10**15) for _ in range(n)]
            backends=[make_backend(r,initial) for r in ROUTES]
            mirror=list(initial)
            for _ in range(10):
                i=rng.randrange(n); val=rng.choice((0,2**63,-2**64,rng.randrange(-1000,1000)))
                mirror[i]=val
                for backend in backends:
                    backend.update(i,val)
                for _ in range(5):
                    a,b=sorted((rng.randrange(n+1),rng.randrange(n+1)))
                    expected=(sum(mirror[a:b]),sum(x*x for x in mirror[a:b]))
                    for backend in backends:
                        self.assertEqual(backend.query(a,b),expected)

    def test_empty_windows_and_zero_values(self):
        for route in ROUTES:
            backend=make_backend(route, [0,0,0,0])
            self.assertEqual(backend.query(0,0),(0,0))
            self.assertEqual(backend.query(4,4),(0,0))
            backend.update(0,9)
            self.assertEqual(backend.query(0,1),(9,81))
            self.assertEqual(backend.query(1,4),(0,0))

    def test_bad_ranges_never_loop(self):
        for route in ROUTES:
            backend=make_backend(route, [1,2,3])
            for start,stop in [(-1,1),(0,4),(2,1),(True,2),(0,False),(0,3.0)]:
                with self.subTest(route=route,start=start,stop=stop):
                    with self.assertRaises(IndexError):backend.query(start,stop)
            for index in (-1,3,1.0,True):
                with self.assertRaises(IndexError):backend.update(index,4)
            self.assertEqual(backend.query(0,3),(6,14))

    def test_exact_int_gate(self):
        class Altered(int):
            def __mul__(self,other):return 999
        for route in ROUTES:
            with self.assertRaises(TypeError):make_backend(route,[Altered(1)])
            with self.assertRaises(TypeError):make_backend(route,[True])
            with self.assertRaises(TypeError):make_backend(route,[1.5])
            backend=make_backend(route,[1,2])
            with self.assertRaises(TypeError):backend.update(0,Altered(2))
            self.assertEqual(backend.query(0,2),(3,5))

    def test_same_checksum_four_plans(self):
        for n in (16,128,256):
            for changes in (1,8):
                for q in (1,8,32):
                    for span in ("short","wide"):
                        w=RangeWorkload(n,changes,q,3,span)
                        expected=checked_routes(w)
                        for route in ROUTES:
                            self.assertEqual(run_workflow(route,w),expected)

    def test_bad_workload(self):
        for args in ((0,1,1,1,'wide'),(8,9,1,1,'short'),
                     (8,1,0,1,'wide'),(8,1,1,1,'unknown')):
            with self.assertRaises(ValueError):RangeWorkload(*args)

    def test_fenwick_full_boundaries(self):
        x=Fenwick([-5,2**90,0,31,-2**92])
        self.assertEqual(x.query(0,5),(-5+2**90+31-2**92,25+2**180+961+2**184))
        self.assertEqual(x.query(2,2),(0,0))


if __name__=='__main__':unittest.main()
