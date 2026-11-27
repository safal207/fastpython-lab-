"""Evidence and fail-closed statistical gates for reuse experiments."""
import unittest
from alexpy.amortization_bench import (
    sample_values,mirrors,paired,report,summarize,empirical_break_even,
)

class TestAmortizationBench(unittest.TestCase):
    def test_input_shapes(self):
        for n in (0,1,8,128):
            for profile in ('small','large'):
                v=sample_values(n,profile)
                self.assertEqual(len(v),n)
                self.assertEqual(mirrors(v,2,False),2*sum(x*x for x in v))
                self.assertEqual(mirrors(v,2,True)[1],v)
        with self.assertRaises(ValueError):sample_values(-1,'small')
        with self.assertRaises(ValueError):sample_values(1,'other')
    def test_check_correctness_before_timing(self):
        with self.assertRaises(AssertionError):
            paired(lambda:14,lambda:15,expected=14,pairs=1,target_seconds=.00001,epoch=0)
    def test_ab_ba_and_report_digest(self):
        obs=paired(lambda:14,lambda:14,expected=14,pairs=3,
                   target_seconds=.00001,epoch=0)
        self.assertEqual([x['order'] for x in obs['pairs']],['AB','BA','AB'])
        self.assertTrue(all(x['reference_ns']>0 for x in obs['pairs']))
        r=report(sizes=(0,8),repetitions=(1,2),profiles=('small',),
                 epochs=2,pairs=2,target_seconds=.00001)
        self.assertEqual(len(r['cases']),16)
        self.assertEqual(len(r['summary']),16)
        self.assertEqual(len(r['empirical_break_even']),4)
        self.assertIn('BLOCKED',r['promotion_gate'])
        self.assertEqual(len(r['sha256_without_digest']),64)
    def test_break_even_monotonic_guard(self):
        rows=[]
        for k,speeds in ((1,[.7,.8]),(2,[1.3,1.4]),(4,[.99,1.6]),(8,[1.2,1.3])):
            rows.append({'candidate':'pack_once','size':8,'profile':'small',
              'repetitions':k,'materialize':False,'epochs':2,
              'epoch_speedups':speeds})
        self.assertEqual(empirical_break_even(rows)[0]['first_measured_repetitions_at_least_10pct_gain'],8)
        rows[-1]['epoch_speedups']=[.8,.9]
        self.assertIsNone(empirical_break_even(rows)[0]['first_measured_repetitions_at_least_10pct_gain'])
    def test_summary_flags_require_multiple_epochs(self):
        case={'epoch':0,'size':8,'profile':'small','repetitions':1,
              'materialize':False,'observations':{'pack_once':{'speedup':2}}}
        self.assertFalse(summarize([case])[0]['repeated_gain_10pct'])

if __name__=='__main__':unittest.main()
