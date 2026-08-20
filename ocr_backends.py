from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class OcrResult:
    lines: list[str]
    confidences: list[float] | None = None

    @property
    def text(self) -> str:
        return "\n".join(line for line in self.lines if line.strip())


class OcrBackend(Protocol):
    def recognize(self, image: np.ndarray) -> OcrResult: ...

    def recognize_many(self, images: list[np.ndarray]) -> list[OcrResult]: ...


def _configure_paddle_cache() -> None:
    workspace_cache = Path(__file__).resolve().parent / ".cache" / "paddlex"
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(workspace_cache))
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


def _result_json(result: Any) -> dict[str, Any]:
    payload = getattr(result, "json", None)
    if callable(payload):
        payload = payload()
    if not isinstance(payload, dict):
        raise RuntimeError("PaddleOCR returned an unsupported result object")
    nested = payload.get("res")
    return nested if isinstance(nested, dict) else payload


def _find_value(data: Any, key: str) -> Any | None:
    if isinstance(data, dict):
        if key in data:
            return data[key]
        for value in data.values():
            found = _find_value(value, key)
            if found is not None:
                return found
    elif isinstance(data, (list, tuple)):
        for value in data:
            found = _find_value(value, key)
            if found is not None:
                return found
    return None


class PaddleMobileOcr:
    """CPU OCR using PP-OCRv5 detection and Latin-script recognition."""

    def __init__(self, device: str = "cpu") -> None:
        _configure_paddle_cache()
        try:
            from paddleocr import PaddleOCR
        except ImportError as error:
            raise RuntimeError(
                "Install the CPU OCR profile with `uv sync --extra ocr-cpu`"
            ) from error

        self._pipeline = PaddleOCR(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="latin_PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device=device,
        )

    def recognize(self, image: np.ndarray) -> OcrResult:
        return self.recognize_many([image])[0]

    def recognize_many(self, images: list[np.ndarray]) -> list[OcrResult]:
        outputs: list[OcrResult] = []
        for result in self._pipeline.predict(images):
            lines: list[str] = []
            scores: list[float] = []
            payload = _result_json(result)
            texts = _find_value(payload, "rec_texts")
            confidences = _find_value(payload, "rec_scores")
            if texts is None:
                texts = []
            if confidences is None:
                confidences = []
            lines.extend(str(text) for text in texts)
            scores.extend(float(score) for score in confidences)
            outputs.append(OcrResult(lines=lines, confidences=scores or None))
        if len(outputs) != len(images):
            raise RuntimeError("PaddleOCR returned a different number of results than inputs")
        return outputs


class TransformersVlOcr:
    """PaddleOCR-VL through PyTorch/Transformers for ARM64 NVIDIA GPUs."""

    def __init__(self, device: str = "cuda") -> None:
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as error:
            raise RuntimeError(
                "Install the DGX profile with `uv sync --extra ocr-dgx`"
            ) from error

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available to PyTorch")

        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(
            "PaddlePaddle/PaddleOCR-VL-1.6"
        )
        self._model = AutoModelForImageTextToText.from_pretrained(
            "PaddlePaddle/PaddleOCR-VL-1.6",
            torch_dtype=torch.bfloat16,
        ).to(device).eval()

    def recognize(self, image: np.ndarray) -> OcrResult:
        from PIL import Image

        rgb = np.ascontiguousarray(image[:, :, ::-1])
        pil_image = Image.fromarray(rgb)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": "OCR:"},
                ],
            }
        ]
        max_pixels = 1280 * 28 * 28
        # Transformers 5.15 stores PaddleOCR-VL's pixel limits in `size`;
        # older versions exposed the lower limit as `min_pixels` directly.
        # Read the supported processor configuration instead of assuming the
        # legacy attribute exists.
        min_pixels = _paddleocr_vl_min_pixels(self._processor.image_processor)
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            images_kwargs={
                "size": {
                    "shortest_edge": min_pixels,
                    "longest_edge": max_pixels,
                }
            },
        ).to(self._model.device)
        with self._torch.inference_mode():
            outputs = self._model.generate(**inputs, max_new_tokens=128)
        prompt_length = inputs["input_ids"].shape[-1]
        text = self._processor.decode(
            outputs[0, prompt_length:],
            skip_special_tokens=True,
        )
        return OcrResult(lines=text.splitlines())

    def recognize_many(self, images: list[np.ndarray]) -> list[OcrResult]:
        # Sequential generation keeps memory use predictable on unified-memory
        # DGX Spark systems. CLI batch size still bounds decoded image memory.
        return [self.recognize(image) for image in images]


def create_ocr_backend(profile: str) -> OcrBackend:
    if profile == "cpu":
        return PaddleMobileOcr()
    if profile == "gpu":
        return TransformersVlOcr()
    raise ValueError(f"Unknown OCR profile: {profile}")
