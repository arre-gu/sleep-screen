from __future__ import annotations

import argparse
import difflib
import glob
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from ocr_backends import OcrBackend


@dataclass(frozen=True)
class ChartGeometry:
    left: int
    right: int
    top: int
    baseline: int


class ScreenTimeReading(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date: str
    total_minutes: int = Field(ge=0, le=24 * 60)
    hourly_minutes: list[int] = Field(min_length=24, max_length=24)

    @model_validator(mode="after")
    def validate_hourly_total(self) -> "ScreenTimeReading":
        if any(value < 0 or value > 60 for value in self.hourly_minutes):
            raise ValueError("hourly_minutes values must be between 0 and 60")
        if sum(self.hourly_minutes) != self.total_minutes:
            raise ValueError("hourly_minutes must sum to total_minutes")
        return self


SWEDISH_MONTHS = (
    "januari",
    "februari",
    "mars",
    "april",
    "maj",
    "juni",
    "juli",
    "augusti",
    "september",
    "oktober",
    "november",
    "december",
)


def _cluster(values: list[int], tolerance: int = 3) -> list[int]:
    if not values:
        return []
    groups: list[list[int]] = [[values[0]]]
    for value in values[1:]:
        if value - groups[-1][-1] <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [int(round(float(np.median(group)))) for group in groups]


def _longest_true_run(mask: np.ndarray) -> tuple[int, int]:
    padded = np.pad(mask.astype(np.int8), (1, 1))
    changes = np.flatnonzero(np.diff(padded))
    if len(changes) < 2:
        return 0, 0
    starts, ends = changes[::2], changes[1::2]
    index = int(np.argmax(ends - starts))
    return int(starts[index]), int(ends[index])


def locate_hourly_chart(image: np.ndarray) -> ChartGeometry:
    """Locate the lower, 0--60 minute chart from its five horizontal grid lines."""
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 20, 70)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(50, width // 5),
        minLineLength=int(width * 0.55),
        maxLineGap=int(width * 0.04),
    )
    if lines is None:
        raise ValueError("Could not find chart grid lines")

    horizontal: list[tuple[int, int, int]] = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        if abs(y2 - y1) <= 2 and abs(x2 - x1) >= width * 0.55:
            horizontal.append((min(x1, x2), max(x1, x2), int(round((y1 + y2) / 2))))

    rows = _cluster(sorted(y1 for _, _, y1 in horizontal))
    candidates: list[tuple[float, list[int]]] = []
    for start in range(len(rows) - 4):
        group = rows[start : start + 5]
        gaps = np.diff(group)
        mean_gap = float(np.mean(gaps))
        if mean_gap < height * 0.012 or mean_gap > height * 0.08:
            continue
        error = float(np.max(np.abs(gaps - mean_gap)))
        if error <= max(3.0, mean_gap * 0.08):
            # Prefer the lower group when both weekly and hourly charts are visible.
            score = group[-1] / height - error / max(mean_gap, 1.0)
            candidates.append((score, group))

    if not candidates:
        raise ValueError(f"Could not identify five equally spaced grid lines; rows={rows}")

    _, grid_rows = max(candidates, key=lambda item: item[0])
    top, baseline = grid_rows[0], grid_rows[-1]

    # The top 60-minute line is unobscured in normal Screen Time charts. Find
    # its exact continuous run instead of trusting Hough endpoints, which can
    # extend into the adjacent "60 m" label.
    best_run = (0, 0)
    for y in range(max(4, top - 3), min(height - 4, top + 4)):
        current = image[y].astype(np.int16)
        before = image[y - 4].astype(np.int16)
        after = image[y + 4].astype(np.int16)
        contrast = np.maximum(
            np.max(np.abs(current - before), axis=1),
            np.max(np.abs(current - after), axis=1),
        )
        run = _longest_true_run(contrast > 5)
        if run[1] - run[0] > best_run[1] - best_run[0]:
            best_run = run
    left, right_exclusive = best_run
    right = right_exclusive - 1
    if right - left < width * 0.55:
        raise ValueError("Could not determine chart's horizontal extent")
    return ChartGeometry(left=left, right=right, top=top, baseline=baseline)


def crop_summary_text(image: np.ndarray, chart: ChartGeometry) -> np.ndarray:
    """Crop the date and daily-total text using the hourly chart as an anchor."""
    height, width = image.shape[:2]
    top = max(0, int(round(chart.top - width * 0.58)))
    bottom = min(height, int(round(chart.top - width * 0.37)))
    left = max(0, chart.left)
    right = min(width, chart.right + 1)
    if bottom <= top or right <= left:
        raise ValueError("Could not derive the Screen Time summary text region")
    return image[top:bottom, left:right]


def parse_swedish_date(text: str) -> str:
    normalized = text.casefold().replace("\n", " ")
    for match in re.finditer(r"(?<!\d)(\d{1,2})\s+([a-zåäö]+)", normalized):
        day = int(match.group(1))
        token = match.group(2)
        month = difflib.get_close_matches(token, SWEDISH_MONTHS, n=1, cutoff=0.72)
        if 1 <= day <= 31 and month:
            return f"{day} {month[0]}"
    raise ValueError(f"Could not parse a Swedish date from OCR text: {text!r}")


def parse_total_minutes(text: str) -> int:
    normalized = " ".join(text.casefold().split())
    hours = re.search(r"(?<!\d)(\d{1,2})\s*h\s*(?:(\d{1,2})\s*m)?", normalized)
    if hours:
        return int(hours.group(1)) * 60 + int(hours.group(2) or 0)
    minutes = re.search(r"(?<!\d)(\d{1,3})\s*m\b", normalized)
    if minutes:
        value = int(minutes.group(1))
        if value <= 24 * 60:
            return value
    raise ValueError(f"Could not parse a daily duration from OCR text: {text!r}")


def measure_hourly_bars(image: np.ndarray, chart: ChartGeometry) -> list[float]:
    """Read the total height of each of the 24 stacked bars."""
    left, right, top, baseline = (
        chart.left,
        chart.right,
        chart.top,
        chart.baseline,
    )
    chart_width = right - left + 1
    step = chart_width / 24.0
    roi = image[top:baseline, left : right + 1].astype(np.int16)

    # At every y coordinate, most of the chart is empty background (or a grid
    # line). Its median color is therefore a robust light/dark-mode reference.
    row_background = np.median(roi, axis=1)
    values: list[float] = []
    for hour in range(24):
        center = (hour + 0.5) * step
        half_width = max(1, int(step * 0.16))
        x1 = max(0, int(round(center)) - half_width)
        x2 = min(chart_width, int(round(center)) + half_width + 1)
        patch = roi[:, x1:x2]
        distance = np.max(np.abs(patch - row_background[:, None, :]), axis=2)
        occupied = np.mean(distance > 14, axis=1) >= 0.55

        # A real bar is a nearly continuous run ending at the baseline. Allow
        # two missed antialiasing rows, but reject isolated text/grid pixels.
        misses = 0
        bar_top = baseline - top
        for y in range((baseline - top) - 1, -1, -1):
            if occupied[y]:
                bar_top = y
                misses = 0
            else:
                misses += 1
                if misses > 2:
                    break
        pixel_height = (baseline - top) - bar_top
        minutes = pixel_height * 60.0 / (baseline - top)
        values.append(max(0.0, min(60.0, minutes)))
    return values


def _remove_zero_line_artifacts(values: list[float]) -> list[float]:
    """Remove a repeated sub-2-minute baseline artifact seen in light mode."""
    small = [value for value in values if 0.25 <= value < 2.0]
    if len(small) < 3:
        return values
    bins: dict[float, int] = {}
    for value in small:
        key = round(value, 1)
        bins[key] = bins.get(key, 0) + 1
    artifact, count = max(bins.items(), key=lambda item: item[1])
    if count < 3:
        return values
    return [0.0 if abs(value - artifact) <= 0.15 else value for value in values]


def _reconcile_total(values: list[float], total_minutes: int) -> list[int]:
    rounded = [int(round(value)) for value in values]
    difference = total_minutes - sum(rounded)
    limit = max(3, int(round(total_minutes * 0.02)))
    if abs(difference) > limit:
        raise ValueError(
            f"Bar sum {sum(rounded)} is too far from OCR total {total_minutes}"
        )
    while difference:
        direction = 1 if difference > 0 else -1
        candidates = [
            index
            for index, value in enumerate(values)
            if value > 0 and (direction > 0 or rounded[index] > 0)
        ]
        if not candidates:
            raise ValueError("Could not reconcile hourly bars with the daily total")
        index = min(
            candidates,
            key=lambda i: abs((rounded[i] + direction) - values[i])
            - abs(rounded[i] - values[i]),
        )
        rounded[index] += direction
        difference -= direction
    return rounded


def read_hourly_bars(
    image: np.ndarray,
    chart: ChartGeometry,
    total_minutes: int | None = None,
) -> list[int]:
    values = _remove_zero_line_artifacts(measure_hourly_bars(image, chart))
    if total_minutes is not None:
        return _reconcile_total(values, total_minutes)
    return [int(round(value)) for value in values]


def extract_chart(
    image_path: str | Path,
    total_minutes: int | None = None,
) -> tuple[ChartGeometry, list[int]]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    chart = locate_hourly_chart(image)
    return chart, read_hourly_bars(image, chart, total_minutes=total_minutes)


def extract_screen_time(
    image_path: str | Path,
    ocr: "OcrBackend",
) -> ScreenTimeReading:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    chart = locate_hourly_chart(image)
    ocr_result = ocr.recognize(crop_summary_text(image, chart))
    date = parse_swedish_date(ocr_result.text)
    total_minutes = parse_total_minutes(ocr_result.text)
    hourly = read_hourly_bars(image, chart, total_minutes=total_minutes)
    return ScreenTimeReading(
        date=date,
        total_minutes=total_minutes,
        hourly_minutes=hourly,
    )


def extract_screen_times(
    image_paths: list[str | Path],
    ocr: "OcrBackend",
) -> list[ScreenTimeReading]:
    prepared: list[tuple[np.ndarray, ChartGeometry]] = []
    crops: list[np.ndarray] = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        chart = locate_hourly_chart(image)
        prepared.append((image, chart))
        crops.append(crop_summary_text(image, chart))

    readings: list[ScreenTimeReading] = []
    for (image, chart), ocr_result in zip(prepared, ocr.recognize_many(crops)):
        date = parse_swedish_date(ocr_result.text)
        total_minutes = parse_total_minutes(ocr_result.text)
        hourly = read_hourly_bars(image, chart, total_minutes=total_minutes)
        readings.append(
            ScreenTimeReading(
                date=date,
                total_minutes=total_minutes,
                hourly_minutes=hourly,
            )
        )
    return readings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+")
    parser.add_argument(
        "--ocr-profile",
        choices=("none", "cpu", "vl"),
        default="none",
        help="CPU PP-OCRv5, GPU PaddleOCR-VL, or bars only",
    )
    parser.add_argument(
        "--vl-server-url",
        help="Base URL of a remote PaddleOCR-VL vLLM server, e.g. http://host:8080",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Images per OCR batch (default: 16)",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Emit an error JSON object and continue with the remaining images",
    )
    parser.add_argument(
        "--total-minutes",
        type=int,
        help="OCR/displayed daily total; reconciles per-bar pixel rounding",
    )
    args = parser.parse_args()
    expanded_images: list[str] = []
    for pattern in args.images:
        matches = glob.glob(pattern)
        expanded_images.extend(matches or [pattern])
    args.images = expanded_images
    if args.total_minutes is not None and len(args.images) != 1:
        parser.error("--total-minutes can only be used with one image")
    if args.ocr_profile != "none" and args.total_minutes is not None:
        parser.error("--total-minutes cannot be combined with an OCR profile")
    if args.vl_server_url and args.ocr_profile != "vl":
        parser.error("--vl-server-url requires --ocr-profile vl")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    ocr = None
    if args.ocr_profile != "none":
        from ocr_backends import create_ocr_backend

        ocr = create_ocr_backend(
            args.ocr_profile,
            server_url=args.vl_server_url,
        )

    if ocr is not None:
        for start in range(0, len(args.images), args.batch_size):
            paths = args.images[start : start + args.batch_size]
            try:
                readings = extract_screen_times(paths, ocr)
            except Exception:
                if not args.continue_on_error:
                    raise
                for image_path in paths:
                    try:
                        reading = extract_screen_time(image_path, ocr)
                        payload = {"image": image_path, **reading.model_dump()}
                    except Exception as error:
                        payload = {
                            "image": image_path,
                            "error": type(error).__name__,
                            "message": str(error),
                        }
                    print(json.dumps(payload, ensure_ascii=False))
                continue
            for image_path, reading in zip(paths, readings):
                print(
                    json.dumps(
                        {"image": image_path, **reading.model_dump()},
                        ensure_ascii=False,
                    )
                )
        return

    for image_path in args.images:
        geometry, hourly = extract_chart(
            image_path,
            total_minutes=args.total_minutes,
        )
        print(
            json.dumps(
                {
                    "image": image_path,
                    "geometry": asdict(geometry),
                    "hourly_minutes": hourly,
                    "hourly_sum": sum(hourly),
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
