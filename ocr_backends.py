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


class PaddleVlOcr:
    """GPU OCR using PaddleOCR-VL-1.6 locally or through a vLLM server."""

    def __init__(
        self,
        *,
        server_url: str | None = None,
        device: str = "gpu:0",
    ) -> None:
        _configure_paddle_cache()
        try:
            from paddleocr import PaddleOCRVL
        except ImportError as error:
            raise RuntimeError(
                "Install the VL client with `uv sync --extra ocr-vl-client`"
            ) from error

        arguments: dict[str, Any] = {
            "pipeline_version": "v1.6",
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_layout_detection": False,
        }
        if server_url:
            endpoint = server_url.rstrip("/")
            if not endpoint.endswith("/v1"):
                endpoint += "/v1"
            arguments.update(
                vl_rec_backend="vllm-server",
                vl_rec_server_url=endpoint,
                vl_rec_api_model_name="PaddlePaddle/PaddleOCR-VL-1.6",
            )
        else:
            arguments["device"] = device
        self._pipeline = PaddleOCRVL(**arguments)

    def recognize(self, image: np.ndarray) -> OcrResult:
        return self.recognize_many([image])[0]

    def recognize_many(self, images: list[np.ndarray]) -> list[OcrResult]:
        outputs: list[OcrResult] = []
        output = self._pipeline.predict(
            images,
            prompt_label="ocr",
            temperature=0.0,
            max_new_tokens=128,
        )
        for result in output:
            lines: list[str] = []
            payload = _result_json(result)
            blocks = payload.get("parsing_res_list", [])
            for block in blocks:
                content = block.get("block_content")
                if content:
                    lines.extend(str(content).splitlines())
            # PaddleOCR-VL does not expose calibrated confidence scores.
            outputs.append(OcrResult(lines=lines))
        if len(outputs) != len(images):
            raise RuntimeError("PaddleOCR-VL returned a different number of results than inputs")
        return outputs


def create_ocr_backend(
    profile: str,
    *,
    server_url: str | None = None,
) -> OcrBackend:
    if profile == "cpu":
        return PaddleMobileOcr()
    if profile == "vl":
        return PaddleVlOcr(server_url=server_url)
    raise ValueError(f"Unknown OCR profile: {profile}")
