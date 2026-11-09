import pytest
import copy
import importlib.util
import json
from pathlib import Path
import unittest

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('tempcompass_data_gate',REPO/'experiment/scripts/operations/prepare_tempcompass_reviewed_data.py')
gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)
PACKAGE = REPO/'experiment/docs/operations/evidence/tempcompass-cpu-preparation-20261021/scoped-evaluation-v1'


@pytest.mark.private_evidence
class TempCompassDataGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((PACKAGE/'DATA_CONFIG.json').read_text())
        cls.bundle = gate.Bundle(cls.config,PACKAGE)
        cls.documents = {key:cls.bundle.data(key) for key in gate.PINS}

    def manifest_check(self,records,source_map=None):
        return gate.validate_manifest(records,source_map or self.documents['source_map'],self.documents['rows'],self.documents['meta'])

    def semantic_bundle(self,**replacements):
        outer = self
        class SemanticBundle(gate.Bundle):
            def data(self,name):
                return replacements.get(name,copy.deepcopy(outer.documents[name]))
        return SemanticBundle(self.config,PACKAGE)

    def test_actual_frozen_1580_rows_and_both_reviews_pass(self):
        result = gate.validate_documents(self.bundle)
        self.assertEqual(result['counts'],gate.COUNTS)
        self.assertTrue(result['review']['passed'])
        self.assertFalse(result['review']['original_audit_passed'])

    def test_component_is_not_unique_media_key(self):
        rows = self.documents['manifest']
        groups = {}
        for row in rows: groups.setdefault(row['source_video_id'],set()).add(row['video_id'])
        self.assertEqual(len(groups),188)
        self.assertTrue(any(len(clips)>1 for clips in groups.values()))
        changed = copy.deepcopy(rows)
        first = next(r for r in changed if len(groups[r['source_video_id']])>1)
        other = next(r for r in changed if r['source_video_id']==first['source_video_id'] and r['video_id']!=first['video_id'])
        first['image_paths'] = other['image_paths']
        with self.assertRaises(ValueError): self.manifest_check(changed)

    def test_duplicate_and_missing_qa_fail(self):
        rows = self.documents['manifest']
        for changed in (rows[:-1],rows[:-1]+[rows[0]]):
            with self.subTest(size=len(changed)), self.assertRaises(ValueError): self.manifest_check(changed)

    def test_gold_or_original_text_change_fails(self):
        for key,value in [('answer','Z'),('original_answer','A. altered'),('question','altered')]:
            rows=copy.deepcopy(self.documents['manifest']);rows[0][key]=value
            with self.subTest(key=key), self.assertRaises(ValueError):self.manifest_check(rows)

    def test_underlying_source_and_component_changes_fail(self):
        for key,value in [('underlying_source_video_ids',['shutterstock:0']),('source_video_id','tempcompass-source-component:0')]:
            rows=copy.deepcopy(self.documents['manifest']);rows[0][key]=value
            with self.subTest(key=key), self.assertRaises(ValueError):self.manifest_check(rows)

    def test_frame_order_and_extra_model_media_fail(self):
        rows=copy.deepcopy(self.documents['manifest']);rows[0]['image_paths'].reverse()
        with self.assertRaises(ValueError):self.manifest_check(rows)
        rows=copy.deepcopy(self.documents['manifest']);rows[0]['audio_path']='injected.wav'
        with self.assertRaises(ValueError):self.manifest_check(rows)

    def test_incomplete_decode_and_changed_sampling_fail(self):
        for change in ('decode','indices'):
            cache=copy.deepcopy(self.documents['cache'])
            if change=='decode':cache['entries'][0]['full_decode_passes']=1
            else:cache['entries'][0]['selected_indices'][0]=1
            with self.subTest(change=change),self.assertRaises(ValueError):gate.validate_documents(self.semantic_bundle(cache=cache))

    def test_review_disagreement_and_nonindependence_fail(self):
        for key,value in [('passed',False),('reviewer','/root'),('independent_of_root_visual_judgment',False)]:
            review=copy.deepcopy(self.documents['independent_review']);review[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):gate.validate_documents(self.semantic_bundle(independent_review=review))

    def test_original_candidate_addition_cannot_be_silently_accepted(self):
        audit=copy.deepcopy(self.documents['audit']);audit['perceptual_duplicate_pairs'].append(copy.deepcopy(audit['perceptual_duplicate_pairs'][0]))
        with self.assertRaises(ValueError):gate.validate_documents(self.semantic_bundle(audit=audit))

    def test_confirmed_byte_overlap_and_incomplete_scan_block_review(self):
        for key,value in [('exact_duplicate_pairs',[{'sha256':'0'*64}]),('scan_complete',False),('search_completed_without_truncation',False)]:
            audit=copy.deepcopy(self.documents['audit']);audit[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):gate.validate_documents(self.semantic_bundle(audit=audit))

    def test_visual_payload_tampering_is_detected(self):
        files=[*self.config['inputs'].values(),*self.config['visual_files'].values()]
        blobs={x['path']:(PACKAGE/x['path']).read_bytes() for x in files}
        target=next(iter(self.config['visual_files'].values()))['path'];blobs[target]=blobs[target][:-1]
        with self.assertRaises(ValueError):gate.validate_documents(gate.Bundle(self.config,blobs=blobs))

    def test_changed_annotation_pin_or_path_traversal_fails(self):
        config=copy.deepcopy(self.config);config['inputs']['rows']['sha256']='0'*64
        with self.assertRaises(ValueError):gate.Bundle(config,PACKAGE)
        config=copy.deepcopy(self.config);config['inputs']['rows']['path']='../escape.json'
        with self.assertRaises(ValueError):gate.Bundle(config,PACKAGE).data('rows')

    def test_no_partial_or_unbound_remote_proof(self):
        for proof in ({},{'status':'TEMP_COMPASS_REMOTE_AUDIT_PASSED','png_sha256_rgb_fingerprint_count':3279}):
            with self.assertRaises(ValueError):gate.validate_remote_proof(proof,self.bundle)


if __name__=='__main__':unittest.main()
