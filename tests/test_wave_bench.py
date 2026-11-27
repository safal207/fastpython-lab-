"""Holdout integrity and smoke tests for the wavefront experiment."""
import unittest
from alexpy.wave_bench import run_session, POLICIES


class TestWaveBench(unittest.TestCase):
    def test_blind_split_and_frozen_routes(self):
        r=run_session(train_sizes=(32,),holdout_sizes=(64,),
                      changes=(1,),reads=(1,),width=8,ticks=2,pairs=1)
        self.assertEqual(len(r['training']),1)
        self.assertEqual(len(r['holdout']),1)
        sample=r['holdout'][0]
        self.assertEqual(set(sample['frozen']),set(POLICIES))
        self.assertEqual(sample['state']['size'],64)
        self.assertEqual(len(r['memory']['light']['observations']),4)
        self.assertGreater(r['training_ns'],0)
        self.assertGreater(r['freeze_ns'],0)
        for p in POLICIES:self.assertGreater(sample['online']['median_ns'][p],0)

    def test_same_coarse_cell_rejected(self):
        with self.assertRaises(ValueError):
            run_session(train_sizes=(64,),holdout_sizes=(65,),
                        changes=(1,),reads=(1,),width=8,ticks=2,pairs=1)


if __name__=='__main__':unittest.main()
