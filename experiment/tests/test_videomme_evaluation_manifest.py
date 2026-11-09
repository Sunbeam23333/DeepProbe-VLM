import copy
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("prepare_vm_eval", Path(__file__).resolve().parents[1] /
                                             "scripts/operations/prepare_videomme_evaluation.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def source_rows():
    rows = []
    for group in range(1, 901):
        for question in (1, 2, 3):
            rows.append({"video_id": f"{group:03d}", "question_id": f"{group:03d}-{question}",
                         "videoID": "azZZZbSwLQght" if group == 490 else f"v{group:010d}",
                         "question": f"Question {group}/{question}?", "options": ["A. One", "B. Two", "C. Three", "D. Four"],
                         "answer": "B", "duration": "short" if group <= 300 else "medium" if group <= 600 else "long",
                         "task_type": "Counting Problem", "domain": "Knowledge"})
    return rows


def registry(rows):
    selected, _, _ = module.gate.build_videomme_subset(rows, module.REVISION, module.ANNOTATION_SHA)
    return {"entries": selected}


def test_annotation_binding_rejects_changed_gold_and_option_order():
    rows = source_rows()
    accepted = registry(rows)
    result, bindings, exclusions = module.build_qa(rows, accepted)
    assert len(result) == len(bindings) == 2697
    assert [r["question_id"] for r in exclusions] == ["490-1", "490-2", "490-3"]
    assert result[0]["question"] == rows[0]["question"]
    assert result[0]["choices"] == ["One", "Two", "Three", "Four"]
    changed = copy.deepcopy(rows)
    changed[0]["answer"] = "D"
    with pytest.raises(ValueError, match="pretraining source registry"):
        module.build_qa(changed, accepted)
    changed = copy.deepcopy(rows)
    changed[0]["options"][0:2] = list(reversed(changed[0]["options"][0:2]))
    with pytest.raises(ValueError, match="original A/B/C/D order"):
        module.build_qa(changed, registry(changed))


def test_frame_mapping_refuses_permuted_or_missing_source_and_preserves_qa():
    rows = source_rows()
    records, _, _ = module.build_qa(rows, registry(rows))
    entries = []
    for source in dict.fromkeys(r["source_video_id"] for r in records):
        paths = [f"frames/{source.split(':')[1]}/{i:02d}.png" for i in range(8)]
        entries.append({"source_video_id": source, "decode_complete": True,
                        "frames": [{"relative_path": p} for p in paths], "image_paths": paths})
    receipt = {"complete": True, "kind": "videomme_uniform8_frame_cache", "num_frames": 8,
               "revision": module.REVISION, "entries": entries}
    converted = module.attach_frames(records, receipt)
    assert len(converted) == 2697
    for before, after in zip(records, converted):
        assert {k: v for k, v in before.items() if k != "video_path"} == {k: v for k, v in after.items() if k != "image_paths"}
    changed = copy.deepcopy(receipt)
    changed["entries"][0]["image_paths"].reverse()
    with pytest.raises(ValueError, match="Ordered source"):
        module.attach_frames(records, changed)
    changed = copy.deepcopy(receipt)
    changed["entries"].pop()
    with pytest.raises(ValueError, match="all 899"):
        module.attach_frames(records, changed)
