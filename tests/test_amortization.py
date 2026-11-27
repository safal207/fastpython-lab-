"""Read-only workflow semantics, admission, and state-machine edge cases."""
from array import array
import unittest
from alexpy.amortization import (
    reference_batch, packed_batch, repacked_batch, _pack,
)
from alexpy import sum_squares


class IteratingList(list):
    def __init__(self, values):
        super().__init__(values)
        self.calls=0
    def __iter__(self):
        self.calls+=1
        yield 8
        yield 9

class FancyInt(int):
    def __mul__(self, other):
        return 5

class TestAmortization(unittest.TestCase):
    def test_exact_results_empty_singleton_large_and_signs(self):
        examples=([],[0],[-1],[2**40, -(2**40), 3],
                 [-2**63,2**63-1], [2**40]*40, list(range(-30,30)))
        for source in examples:
            for container in (list,tuple):
                values=container(source)
                for k in (1,2,8):
                    for ret in (False,True):
                        with self.subTest(n=len(values),k=k,ret=ret):
                            expected=reference_batch(values,k,materialize=ret)
                            self.assertEqual(packed_batch(values,k,materialize=ret),expected)
                            self.assertEqual(repacked_batch(values,k,materialize=ret),expected)
                            if ret:
                                self.assertIsNot(packed_batch(values,k,materialize=True)[1],values)
    def test_unsupported_is_rejected_without_user_iteration(self):
        weird=IteratingList([1,2,3])
        self.assertIs(_pack(weird),NotImplemented)
        self.assertIs(packed_batch(weird,2),NotImplemented)
        self.assertEqual(weird.calls,0)
        for values in ([FancyInt(3)],[True],[2**100], [1.5],iter([1,2,3]),array('q',[2,3])):
            self.assertIs(packed_batch(values,1),NotImplemented)
            self.assertIs(repacked_batch(values,1),NotImplemented)
    def test_reference_preserves_iteration_semantics(self):
        weird=IteratingList([20,30])
        self.assertEqual(reference_batch(weird,2),2*(64+81))
        self.assertEqual(weird.calls,2)
        self.assertEqual(reference_batch([FancyInt(4)],2),10)
    def test_original_not_mutated_and_results_are_exact(self):
        values=[1,2**40,-3]
        before=values.copy()
        self.assertEqual(packed_batch(values,4),(1+2**80+9)*4)
        self.assertEqual(values,before)
        result, output=packed_batch(values,3,materialize=True)
        self.assertEqual(result,3*(1+2**80+9))
        self.assertEqual(output,values)
        self.assertIsNot(output,values)
    def test_bad_controls(self):
        for k in (0,-1):
            with self.assertRaises(ValueError):packed_batch([1],k)
        for k in (True,1.2,None):
            with self.assertRaises(TypeError):packed_batch([1],k)
        with self.assertRaises(TypeError):reference_batch([1],1,materialize=1)

if __name__=='__main__':unittest.main()
