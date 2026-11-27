"""Check no hidden future, no NaNs, and hardware-profile isolation."""
import unittest
from alexpy.calibration_gate import TrustEvidence,trust_score,probe_decision

class TestCalibrationGate(unittest.TestCase):
    def test_trust_monotonic_and_cross_host(self):
        fresh=TrustEvidence(200,0,0.,0.)
        old=TrustEvidence(200,1024,.2,.1)
        self.assertGreater(trust_score(fresh),trust_score(old))
        self.assertEqual(trust_score(TrustEvidence(200,0,0.,0.,False)),0.)
        self.assertEqual(trust_score(TrustEvidence(0,0,0.,0.)),0.)

    def test_risk_increases_probe_frequency(self):
        safe=probe_decision(TrustEvidence(200,0,.0,0.0))
        risk=probe_decision(TrustEvidence(200,600,.7,.4))
        self.assertGreater(safe.interval,risk.interval)
        self.assertFalse(safe.fallback)
        self.assertTrue(risk.fallback)

    def test_mismatched_host_fails_closed(self):
        d=probe_decision(TrustEvidence(100000,0,0.,0.,False))
        self.assertTrue(d.fallback)

    def test_invalid_numbers_and_inputs(self):
        for kwargs in ({'count':-1,'age':0,'drift':0.,'error':0.},
                       {'count':1,'age':-1,'drift':0.,'error':0.},
                       {'count':1,'age':0,'drift':float('nan'),'error':0.},
                       {'count':1,'age':0,'drift':0.,'error':float('inf')},
                       {'count':1,'age':0,'drift':0.,'error':2.}):
            with self.assertRaises(ValueError):TrustEvidence(**kwargs)
        e=TrustEvidence(1,0,0.,0.)
        for kwargs in ({'prior':0.}, {'half_life':float('nan')}):
            with self.assertRaises(ValueError):trust_score(e,**kwargs)
        with self.assertRaises(ValueError):probe_decision(e,probe_cost=-1)
        with self.assertRaises(ValueError):probe_decision(e,min_interval=20,max_interval=8)
