# Screen Time screenshot extraction

The extractor uses OCR only for the Swedish date and displayed daily total. The
24 hourly values are measured from the chart pixels and reconciled against the
daily total. This keeps the output deterministic across light/dark mode and
different screenshot sizes.

## Laptop profile (CPU)

Install the mobile PaddleOCR detector and Swedish-aware Latin recognizer:

```powershell
uv sync --extra ocr-cpu
```

Run one image or a batch. Output is JSON Lines, one object per image:

```powershell
.\.venv\Scripts\python.exe screen_time_extractor.py images\image1.png --ocr-profile cpu

.\.venv\Scripts\python.exe screen_time_extractor.py images\*.png --ocr-profile cpu --batch-size 16 --continue-on-error
```

The first run downloads the official `PP-OCRv5_mobile_det` and
`latin_PP-OCRv5_mobile_rec` weights into `.cache/paddlex`. Subsequent runs use
the local cache.

## Remote GPU profile

The high-accuracy profile uses `PaddlePaddle/PaddleOCR-VL-1.6`, a 1B-parameter
BF16 document/OCR model. The recommended deployment is its official vLLM
server on the Linux GPU host:

```bash
docker run --rm --gpus all --network host \
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-nvidia-gpu \
  paddleocr genai_server \
  --model_name PaddleOCR-VL-1.6-0.9B \
  --host 0.0.0.0 \
  --port 8080 \
  --backend vllm
```

Install the VL client. If it runs on the same Windows laptop, retain the CPU
Paddle runtime as well:

```powershell
uv sync --extra ocr-cpu --extra ocr-vl-client
```

Then point the extractor at the remote host:

```powershell
.\.venv\Scripts\python.exe screen_time_extractor.py images\*.png `
  --ocr-profile vl `
  --vl-server-url http://REMOTE_HOST:8080 `
  --batch-size 32
```

For direct GPU inference instead of a server, install the PaddlePaddle GPU wheel
matching the remote machine's CUDA version, install `ocr-vl-client`, and omit
`--vl-server-url`.

## Validation behavior

Each result is validated with a Pydantic model:

- `date` must be a recognized Swedish calendar date without an inferred year.
- `total_minutes` must be between 0 and 1440.
- `hourly_minutes` must contain exactly 24 values between 0 and 60.
- The hourly values must sum to the displayed daily total.

Images that fail OCR, chart detection, or reconciliation raise an error rather
than silently returning plausible-looking data. For a large dataset, capture
those failures into a review queue instead of discarding them.
