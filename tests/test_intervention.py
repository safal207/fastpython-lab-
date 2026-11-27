import unittest
from alexpy.graph_search import State
from alexpy.intervention import STATES, verify, parse_states, worker, experiment
from alexpy.graph_search import make_values


class TestInterventions(unittest.TestCase):
    def test_state_set_has_controls(self):
        self.assertEqual(len(STATES), len(set(STATES)))
        self.assertIn(State('tuple', 1, 'huge'), STATES)
        self.assertIn(State('list_subclass', 0, 'small'), STATES)
        self.assertIn(State('list', 10000, 'mixed'), STATES)

    def test_all_candidate_mirrors(self):
        for state in STATES:
            with self.subTest(state=state.id):
                self.assertEqual(len(verify(make_values(state))), 64)

    def test_parsing_rejects_duplicate_as_experiment(self):
        self.assertEqual(parse_states('list|1|small|steady'), [State('list', 1, 'small')])
        with self.assertRaises(ValueError):
            parse_states('list|1|small')
        with self.assertRaises(ValueError):
            experiment(states=[STATES[0], STATES[0]], sessions=1, repeats=1)

    def test_small_worker(self):
        rows = worker([State('tuple', 1, 'huge'), State('list', 0, 'small')],
                      repeats=2, target_seconds=.0001, session=0)
        self.assertEqual(len(rows['rows']), 2)
        for row in rows['rows']:
            self.assertEqual(set(row['ratios']), {'bigint_native', 'small_dispatch'})

    def test_validation(self):
        for kw in ({'sessions': 0}, {'repeats': 0}, {'target_seconds': -1}):
            with self.assertRaises(ValueError):
                experiment(**kw)


if __name__ == '__main__':
    unittest.main()

class TestPromotionGate(unittest.TestCase):
    def test_guardset_regressions_block_promotion(self):
        # A single real subprocess is enough to validate shape; multi-process
        # policy is exercised through the 3-session study, not a CI speed gate.
        report = experiment(states=[State('list', 0, 'small')],
                            sessions=1, repeats=1, target_seconds=.0001)
        self.assertEqual(report['promotion_gate']['bigint_native'],
                         'NOT_PROMOTED_SECOND_MACHINE_REQUIRED')
        self.assertEqual(report['promotion_gate']['small_dispatch'],
                         'NOT_PROMOTED_SECOND_MACHINE_REQUIRED')
