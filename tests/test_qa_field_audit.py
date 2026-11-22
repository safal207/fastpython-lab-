"""Smoke tests for isolated RAM auditing and anonymized public CI adapter."""
import json
import tempfile
from pathlib import Path
import unittest

from alexpy.qa_field_audit import (
    audit, recorded_case, resident_peak, measure_worker, RECORDED, DEFAULT_FIXTURE,
)
from alexpy.qa_actions import extract_jobs
from alexpy.qa_lab import PASS
from alexpy.qa_native import NativeBatchQA
from alexpy.qa_batch import StrongPythonBatchQA, AlexPyBatchQAGraph
from alexpy.qa_native_bench import verify


class TestFieldAudit(unittest.TestCase):
    def test_public_ci_fixture_is_redacted_and_replayable(self):
        d=json.loads(DEFAULT_FIXTURE.read_text(encoding='utf-8'))
        self.assertEqual(len(d['jobs']),24)
        self.assertTrue(all(set(j)=={'name','status','conclusion','started_at','completed_at'} for j in d['jobs']))
        self.assertFalse(any('runner_name' in j or 'id' in j for j in d['jobs']))
        case,suites,events=recorded_case(DEFAULT_FIXTURE)
        self.assertEqual(case.events,24)
        self.assertEqual(case.tests,6)
        self.assertEqual(case.suites,3)
        self.assertTrue(all(e.status == PASS for e in events))
        for size in (1,4,16):
            self.assertIsInstance(verify(case,size,suites,events),int)
        engines=[cls(suites,3) for cls in (NativeBatchQA,StrongPythonBatchQA,AlexPyBatchQAGraph)]
        for eng in engines:
            eng.apply_batch(events)
        self.assertEqual(engines[0].snapshot(),engines[1].snapshot())
        self.assertEqual(engines[0].snapshot(),engines[2].snapshot())
        self.assertTrue(engines[0].verify())

    def test_native_memory_worker_measures_traced_heap(self):
        x=measure_worker(RECORDED,'native_batch',4,DEFAULT_FIXTURE)
        self.assertGreater(x['trace_peak_bytes'],0)
        self.assertGreater(x['rss_peak_bytes'],0)
        self.assertEqual(x['events'],24)
        self.assertEqual(x['route'],'native_batch')

    def test_isolated_audit_and_routes_agree(self):
        data=audit(cases=(RECORDED,),sizes=(4,),sessions=1)
        self.assertEqual(data['schema'],'ALEXPY-QA-FIELD-AUDIT/1')
        self.assertEqual(len(data['raw']),3)
        self.assertEqual(len({x['digest'] for x in data['raw']}),1)
        self.assertEqual(set(data['summary'][0]['routes']),{'native_batch','strong_batch','alexpy_batch'})
        self.assertIn('instrumented_time',data['measurement_contract'])

    def test_bad_recorded_fixture_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'invalid.json';p.write_text('{"jobs": []}',encoding='utf-8')
            with self.assertRaises(ValueError):recorded_case(p)

    def test_memory_stats_and_config_guard(self):
        self.assertGreater(resident_peak(),0)
        with self.assertRaises(ValueError):audit(cases=('unknown',))
        with self.assertRaises(ValueError):audit(sizes=(0,))
        with self.assertRaises(ValueError):measure_worker(RECORDED,'nope',1,DEFAULT_FIXTURE)


if __name__=='__main__':unittest.main()
