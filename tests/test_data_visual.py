import tempfile
from pathlib import Path
import unittest

import torch
from PIL import Image

from deepprobe_vlm.data import MultimodalCollator, load_frames, uniform_indices
from deepprobe_vlm.evaluate import score_answer


class FakeTokenizer:
    eos_token_id = 2
    bos_token_id = 1
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(c)+3 for c in text]


class DataVisualTest(unittest.TestCase):
    def test_uniform_sampling_repeats_short_clips_deterministically(self):
        self.assertEqual(uniform_indices(2, 4), [0, 0, 1, 1])
        self.assertEqual(uniform_indices(10, 3), [0, 4, 9])
        with self.assertRaises(ValueError):
            uniform_indices(0, 3)

    def test_answer_only_labels_and_inference_no_answer(self):
        record = {"sample_id": "x", "question": "Which?", "answer": "ZEBRA",
                  "choices": ["Yes", "No"]}
        sample = {"record": record, "pixel_values": torch.zeros(1, 3, 8, 8)}
        train = MultimodalCollator(FakeTokenizer(), num_visual_tokens=2, max_text_tokens=256)([sample])
        target = train["labels"][train["labels"] != -100].tolist()
        self.assertEqual(target, FakeTokenizer().encode(" ZEBRA")+[2])
        self.assertEqual(train["visual_mask"].sum().item(), 2)
        infer = MultimodalCollator(FakeTokenizer(), num_visual_tokens=2, max_text_tokens=256,
                                  include_answer=False)([sample])
        self.assertNotIn("labels", infer)
        self.assertEqual(infer["input_ids"].shape[1], train["input_ids"].shape[1]-len(target))

    def test_no_silent_text_truncation(self):
        sample = {"record": {"sample_id": "long", "question": "Q"*100, "answer": "yes"},
                  "pixel_values": torch.zeros(1, 3, 8, 8)}
        with self.assertRaises(ValueError):
            MultimodalCollator(FakeTokenizer(), num_visual_tokens=1, max_text_tokens=20)([sample])

    def test_image_order(self):
        with tempfile.TemporaryDirectory() as directory:
            for i in range(3):
                Image.new("RGB", (8, 8), (i*100, 0, 0)).save(Path(directory)/f"{i}.png")
            frames = load_frames({"image_paths": ["0.png", "1.png", "2.png"]}, directory, 2)
            self.assertEqual([x.getpixel((0, 0))[0] for x in frames], [0, 200])

    def test_video_bounds_are_respected(self):
        import av
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"clip.mp4"
            with av.open(str(path), "w") as container:
                stream = container.add_stream("mpeg4", rate=10)
                stream.width = stream.height = 16
                stream.pix_fmt = "yuv420p"
                for i in range(8):
                    frame = av.VideoFrame.from_ndarray(np.full((16, 16, 3), i*25, dtype=np.uint8), format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            frames = load_frames({"video_path": "clip.mp4", "start_seconds": .2,
                                  "end_seconds": .5}, directory, 2)
            self.assertAlmostEqual(frames[0].getpixel((0, 0))[0], 50, delta=5)
            self.assertAlmostEqual(frames[1].getpixel((0, 0))[0], 100, delta=5)

    def test_strict_mcq_does_not_parse_arbitrary_first_letter(self):
        self.assertEqual(score_answer("Answer: B", "B", ["yes", "no"])["correct"], 1)
        self.assertEqual(score_answer("Actually B or C", "A", ["yes", "no"])["correct"], 0)


if __name__ == "__main__":
    unittest.main()
