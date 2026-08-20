from __future__ import annotations

import unittest
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from ocr_backends import OcrResult, _paddleocr_vl_min_pixels
from screen_time_extractor import (
    ScreenTimeReading,
    extract_screen_times,
    parse_swedish_date,
    parse_total_minutes,
)


ROOT = Path(__file__).resolve().parents[1]


class FakeOcr:
    def __init__(self, texts: list[str]) -> None:
        self._texts = texts

    def recognize(self, image: np.ndarray) -> OcrResult:
        return self.recognize_many([image])[0]

    def recognize_many(self, images: list[np.ndarray]) -> list[OcrResult]:
        if len(images) != len(self._texts):
            raise AssertionError("Unexpected fake OCR batch size")
        return [OcrResult(lines=text.splitlines()) for text in self._texts]


class ParsingTests(unittest.TestCase):
    def test_swedish_date_variants(self) -> None:
        self.assertEqual(parse_swedish_date("söndag, 22 mars"), "22 mars")
        self.assertEqual(parse_swedish_date("Igår, 19 mars"), "19 mars")
        self.assertEqual(parse_swedish_date("22 februar1"), "22 februari")

    def test_duration_variants(self) -> None:
        self.assertEqual(parse_total_minutes("3 h 51 m"), 231)
        self.assertEqual(parse_total_minutes("2h 43m"), 163)
        self.assertEqual(parse_total_minutes("59 m"), 59)

    def test_schema_rejects_wrong_sum(self) -> None:
        with self.assertRaises(ValueError):
            ScreenTimeReading(
                date="22 mars",
                total_minutes=10,
                hourly_minutes=[0] * 24,
            )


class ExtractionTests(unittest.TestCase):
    def test_all_sample_layouts_with_fake_ocr(self) -> None:
        paths = [ROOT / "images" / f"image{index}.png" for index in range(1, 6)]
        ocr = FakeOcr(
            [
                "söndag, 22 mars\n3 h 51 m",
                "Igår, 19 mars\n2 h 43 m",
                "söndag, 22 februari\n3 h 4 m",
                "Igår, 19 mars\n5 h 13 m",
                "måndag, 23 mars\n4 h 19 m",
            ]
        )
        with TemporaryDirectory() as boxes_dir:
            readings = extract_screen_times(paths, ocr, Path(boxes_dir))
            self.assertEqual(
                sorted(path.name for path in Path(boxes_dir).iterdir()),
                [f"image{index}.png" for index in range(1, 6)],
            )
        self.assertEqual([reading.total_minutes for reading in readings], [231, 163, 184, 313, 259])
        self.assertTrue(all(len(reading.hourly_minutes) == 24 for reading in readings))
        self.assertTrue(
            all(sum(reading.hourly_minutes) == reading.total_minutes for reading in readings)
        )


class GpuBackendTests(unittest.TestCase):
    def test_transformers_processor_uses_size_configuration(self) -> None:
        image_processor = SimpleNamespace(
            size={"shortest_edge": 384 * 384}
        )
        self.assertEqual(_paddleocr_vl_min_pixels(image_processor), 384 * 384)


if __name__ == "__main__":
    unittest.main()
