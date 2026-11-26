"""Offline negative controls for the read-only AlexPy QA 3.1 evidence auditor."""
import copy
import hashlib
import math
import statistics
import unittest

from alexpy.qa_crosshost import audit, nearest_rank

SOURCES = ('ubuntu-latest', 'macos-latest', 'windows-latest')
PYTHONS = ('3.13', '3.15')
RUNS = (45717781984, 45717790340)
PATHS = (('small', 'single'), ('small', 'small'), ('medium', 'wide'))


def result(time_ns=1000, peak_bytes=1000):
    samples = [time_ns + i % 3 for i in range(60)]
    peak_samples = [peak_bytes, peak_bytes + 2, peak_bytes + 4]
    return {'n': len(samples), 'samples_ns': samples,
            'median_ns': statistics.median(samples),
            'p95_ns': nearest_rank(samples, .95),
            'p99_ns': nearest_rank(samples, .99),
            'peak_bytes': statistics.median(peak_samples),
            'peak_samples': peak_samples}


def rows():
    return {'c_tree': result(1000), 'bitset': result(1150),
            'python': result(1500), 'native_set': result(1400)}


def make_record(os_name, py, run):
    platform = {'ubuntu-latest': 'Linux', 'macos-latest': 'macOS',
                'windows-latest': 'Windows'}[os_name]
    train = [{'n': n, 'batch': b, 'seed': 1000 + n, 'results': rows()}
             for n, b in ((19, 1), (23, 8), (384, 64))]
    sessions = [{'pid': 100 + i, 'train': copy.deepcopy(train)} for i in range(3)]
    holdouts = []
    for n, b in ((29, 1), (31, 13), (769, 96), (81, 1)):
        r = rows()
        r['adaptive'] = result(1010)
        holdouts.append({'n': n, 'batch': b, 'seed': 9000 + n, 'route': 'c_tree',
                         'reason': 'unknown_workload' if n == 81 else 'risk_or_gain_gate',
                         'results': r})
    report = {'schema': 'alexpy-qa30/1',
              'environment': f'{platform}-test|{py}.77|{run}|tests|12345',
              'overhead_p99_ns': 10, 'sessions': sessions,
              'decisions': [{'key': list(k), 'route': 'c_tree', 'reason': 'risk_or_gain_gate'}
                            for k in PATHS],
              'holdouts': holdouts, 'experimental': True, 'default_promotion': False,
              'switch_scope': 'fresh engine only; no live state migration'}
    return {'artifact_id': 123, 'name': f'alexpy-qa30-{os_name}-py{py}-{run}',
            'sha256': hashlib.sha256(b'test').hexdigest(), 'run_id': run,
            'source_sha': '3'*40, 'report': report}


def fixture():
    return [make_record(os_name, py, run) for run in RUNS
            for os_name in SOURCES for py in PYTHONS]


class QA31AuditTests(unittest.TestCase):
    def test_complete_unmodified_frozen_controls(self):
        s = audit(fixture())
        self.assertEqual(s['configs'], 6)
        self.assertEqual(s['artifact_count'], 12)
        self.assertEqual(s['train_cells'], 36)
        self.assertEqual(s['holdout_count'], 48)
        self.assertTrue(s['all_fallback_to_c_tree'])
        self.assertFalse(s['auto_promotion_justified'])
        self.assertGreater(s['median_adaptive_over_tree']['median_ns'], 1.0)

    def test_forged_training_decision_rejected(self):
        a = fixture()
        a[0]['report']['decisions'][0]['route'] = 'bitset'
        with self.assertRaisesRegex(ValueError, 'not reproducible'):
            audit(a)

    def test_holdout_leakage_or_unmatched_route_rejected(self):
        a = fixture()
        a[0]['report']['holdouts'][0]['route'] = 'bitset'
        with self.assertRaisesRegex(ValueError, 'holdout route'):
            audit(a)

    def test_forged_p99_rejected(self):
        a = fixture()
        a[0]['report']['holdouts'][0]['results']['bitset']['p99_ns'] = 1
        with self.assertRaisesRegex(ValueError, 'inconsistent p99'):
            audit(a)

    def test_nan_and_zero_samples_rejected(self):
        for bad in (float('nan'), 0):
            a = fixture()
            a[0]['report']['sessions'][0]['train'][0]['results']['c_tree']['samples_ns'][0] = bad
            with self.assertRaises(ValueError):
                audit(a)

    def test_missing_run_rejected(self):
        with self.assertRaisesRegex(ValueError, 'missing or unmatched'):
            audit(fixture()[:-1])

    def test_duplicate_artifact_rejected(self):
        a = fixture()
        a[-1] = a[0]
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            audit(a)

    def test_mixed_commit_rejected(self):
        a = fixture()
        a[-1]['source_sha'] = 'f' * 40
        with self.assertRaisesRegex(ValueError, 'different source commits'):
            audit(a)

    def test_wrong_runner_rejected(self):
        a = fixture()
        a[0]['report']['environment'] = a[0]['report']['environment'].replace('Linux', 'Windows')
        with self.assertRaisesRegex(ValueError, 'platform mismatch'):
            audit(a)

    def test_default_activation_rejected(self):
        a = fixture()
        a[0]['report']['default_promotion'] = True
        with self.assertRaisesRegex(ValueError, 'fail closed'):
            audit(a)

    def test_real_raw_zip_digest_is_checked(self):
        from pathlib import Path
        import json
        import tempfile
        import zipfile
        from alexpy.qa_crosshost import records_from_manifest
        with tempfile.TemporaryDirectory() as d:
            base=Path(d)
            (base/'raw.zip').write_bytes(b'not really a zip')
            mf={'required_runs': 2, 'source_sha': '3'*40, 'artifacts': [
                {'zip_file': 'raw.zip','sha256': '0'*64,'json_sha256': '0'*64}]}
            (base/'manifest.json').write_text(json.dumps(mf))
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                records_from_manifest(base/'manifest.json', base)

    def test_missing_five_percent_gain_no_promotion(self):
        a = fixture()
        # Even if adaptation looks faster on the holdout, frozen training
        # prohibits selecting an unqualified alternative.
        a[0]['report']['holdouts'][0]['results']['adaptive'] = result(500)
        s = audit(a)
        self.assertEqual(s['alternative_selections'], 0)
        self.assertFalse(s['auto_promotion_justified'])


if __name__ == '__main__':
    unittest.main()
