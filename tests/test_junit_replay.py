"""JUnit replay tests including duplicate attempts and real format semantics."""
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree as ET

from alexpy.junit_replay import parse_junit, compare
from alexpy.qa_lab import PASS, FAIL, SKIP


class TestJUnitReplay(unittest.TestCase):
    def report(self, text):
        td = tempfile.TemporaryDirectory()
        p = Path(td.name) / "pytest-junit.xml"
        p.write_text(text, encoding="utf-8")
        self.addCleanup(td.cleanup)
        return p

    def test_mixed_real_junit_semantics(self):
        report=self.report('''<testsuites><testsuite name="pytest" tests="5">
          <testcase classname="tests.a" name="test_a" time="0.001" />
          <testcase classname="tests.b" name="test_b" time="0.100"><failure>bad</failure></testcase>
          <testcase classname="tests.b" name="test_c" time="0.5"><skipped /></testcase>
          <testcase classname="tests.a" name="test_a" time="0.003"><error>crash</error></testcase>
          <testcase classname="tests.a" name="test_a" time="0.004" />
        </testsuite></testsuites>''')
        suites,events,meta=parse_junit(report)
        self.assertEqual([e.status for e in events], [PASS,FAIL,SKIP,FAIL,PASS])
        self.assertEqual(meta["retried_tests"],2)
        self.assertEqual(suites,[0,1,1])
        self.assertEqual(events[2].duration_ms,0)
        result=compare(report,batch=2)
        self.assertTrue(result["all_routes_match"])
        self.assertEqual(result["failure_witness_ids"],[1])
        self.assertEqual(result["final_counts"],[0,1,1,1,104])

    def test_reject_empty_invalid_and_missing(self):
        for xml in ("", "<bad />", "<testsuite />", "<testsuite><testcase /></testsuite>",
                    '<testsuite><testcase name="a" time="-1" /></testsuite>',
                    '<testsuite><testcase name="a" time="nan" /></testsuite>'):
            with self.subTest(xml=xml):
                with self.assertRaises((ValueError,ET.ParseError)):
                    compare(self.report(xml))

    def test_reject_invalid_batch_and_unbounded(self):
        p=self.report('<testsuite><testcase name="a" /></testsuite>')
        for batch in (0,-1,False,1.5):
            with self.assertRaises(ValueError):compare(p,batch=batch)
        with self.assertRaises(ValueError):parse_junit(p,max_bytes=2)
