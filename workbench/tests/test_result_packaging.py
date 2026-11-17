"""CPU-only coverage for lossless mask report transport and bounded finalization."""
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from tools import run_evaluation_job as runner


class ResultPackagingTests(unittest.TestCase):
    def test_whitespace_compaction_preserves_coordinates_holes_and_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            data={'images':[{'segments':[{'hole':False,'points':[[1.23456789012345,2.0]]*80},{'hole':True,'points':[[3,4]]*80}]}]}
            raw=json.dumps(data,indent=2).encode();(directory/'report.json').write_bytes(raw)
            request=dict(job_id='a',dispatch_id='b',snapshot_sha256='c',dataset_sha256='d',return_token='e',kind='inference')
            with patch.object(runner,'MAX_ARTIFACT_BYTES',len(raw)//2):runner.pack_result(directory,request,{'status':'interrupted'})
            self.assertEqual((directory/'report.json').read_bytes(),raw)
            with zipfile.ZipFile(directory/'return-result.zip') as z:
                packed=z.read('report.json');receipt=json.loads(z.read('receipt.json'))
                self.assertEqual(json.loads(packed),data)
                self.assertEqual(receipt['artifacts']['report.json'],{'bytes':len(packed),'sha256':hashlib.sha256(packed).hexdigest()})
                self.assertEqual(receipt['kind'],'inference')

    def test_semantically_large_report_fails_without_publishing_or_truncating(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);raw=json.dumps({'predictions':['abc']*200}).encode();(directory/'report.json').write_bytes(raw)
            with patch.object(runner,'MAX_ARTIFACT_BYTES',100),self.assertRaises(RuntimeError):runner.pack_result(directory,{}, {})
            self.assertFalse((directory/'return-result.zip').exists());self.assertEqual((directory/'report.json').read_bytes(),raw)

    def test_source_report_bound_checked_before_parse(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);(directory/'report.json').write_bytes(b'x'*100)
            with patch.object(runner,'MAX_SOURCE_REPORT_BYTES',50),self.assertRaises(RuntimeError):runner.pack_result(directory,{}, {})

    def test_archive_and_receipt_budget_remain_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);(directory/'report.json').write_text('{}')
            req=dict(job_id='a',dispatch_id='b',snapshot_sha256='c',dataset_sha256='d',return_token='e')
            with patch.object(runner,'MAX_RETURN_BYTES',10),self.assertRaises(RuntimeError):runner.pack_result(directory,req,{})
            self.assertFalse((directory/'return-result.zip').exists())
