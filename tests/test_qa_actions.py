import json
from pathlib import Path
import unittest

from alexpy.qa_actions import extract_jobs, summarize_jobs
from alexpy.qa_lab import PASS, FAIL, SKIP

FIXTURE = Path(__file__).resolve().parents[1] / "evidence" / "qa-actions-jobs-sample.json"


class TestActionsAdapter(unittest.TestCase):
    def test_current_public_run(self):
        payload=json.loads(FIXTURE.read_text(encoding="utf-8"))
        suites,names,mapping,events=extract_jobs(payload)
        self.assertEqual(len(events),6)
        self.assertEqual(suites,("macos-latest","ubuntu-latest","windows-latest"))
        self.assertTrue(all(e.status==PASS for e in events))
        report=summarize_jobs(payload)
        self.assertEqual(report["total"],(0,6,0,0,165000))
        self.assertEqual(report["jobs"],6)
        self.assertEqual(set(report["suites"]),set(suites))

    def test_retry_latest_status_is_final(self):
        row=lambda outcome,t0,t1:{"name":"tests (ubuntu-latest, 3.13)",
            "status":"completed","conclusion":outcome,
            "started_at":t0,"completed_at":t1}
        payload={"jobs":[row("success","2026-10-01T01:00:00Z","2026-10-01T01:00:02Z"),
                        row("failure","2026-10-01T00:00:00Z","2026-10-01T00:00:01Z")]}
        r=summarize_jobs(payload)
        self.assertEqual(r["jobs"],1)
        self.assertEqual(r["events"],2)
        self.assertEqual(r["total"],(0,1,0,0,2000))

    def test_skip_and_incomplete_ignored(self):
        jobs=[{"name":"step 1","status":"completed","conclusion":"skipped",
               "started_at":"2026-01-01T00:00:00Z","completed_at":"2026-01-01T00:00:01Z"},
              {"name":"step 2","status":"in_progress","conclusion":None}]
        r=summarize_jobs({"jobs":jobs})
        self.assertEqual(r["total"],(0,0,0,1,0))
        self.assertEqual(r["jobs"],1)

    def test_malformed_timestamps_fail_closed(self):
        payload={"jobs":[{"name":"x","status":"completed","conclusion":"success",
            "started_at":"2026-01-01T00:00:10Z","completed_at":"2026-01-01T00:00:01Z"}]}
        with self.assertRaises(ValueError):summarize_jobs(payload)

    def test_no_completed_jobs(self):
        self.assertEqual(summarize_jobs({"jobs":[]})["jobs"],0)

    def test_payload_type_validation(self):
        with self.assertRaises(TypeError):summarize_jobs({"jobs":None})


if __name__=="__main__":unittest.main()
