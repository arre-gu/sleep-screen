# Screen Time screenshot extraction

This project is intended to run directly on a Linux GPU machine containing the
code and an `images/` directory:

```text
sleep-screen/
├── images/
│   ├── image1.png
│   └── ...
├── ocr_backends.py
├── screen_time_extractor.py
└── pyproject.toml
```

OCR reads only the Swedish date and displayed daily total. The 24 hourly values
are measured from the chart pixels and reconciled against that total. The
result is deterministic across light/dark mode and different screenshot sizes.

## GPU setup

The GPU profile uses `PaddlePaddle/PaddleOCR-VL-1.6`, a 1B-parameter BF16 OCR
and document-understanding model. If `uv` is not already installed on the Linux
machine, install it with the official standalone installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

The installer normally updates the shell profile automatically. The explicit
`PATH` export also makes `uv` available immediately in the current SSH session.

Install the project and its VL dependencies:

```bash
cd /path/to/sleep-screen
uv sync --extra ocr-vl-client
```

Install the PaddlePaddle GPU wheel matching the machine's CUDA version. For
CUDA 12.6:

```bash
uv pip install paddlepaddle-gpu==3.2.1 \
  --index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

Use the corresponding PaddlePaddle package index for a different CUDA version.
Verify that the environment can see the GPU:

```bash
.venv/bin/python -c "import paddle; print(paddle.device.cuda.device_count()); paddle.utils.run_check()"
```

The first extraction downloads the official model weights into
`.cache/paddlex`. Later jobs reuse that cache.

## Run the extraction job over SSH

SSH to the GPU machine and enter the project directory:

```bash
ssh USER@GPU_HOST
cd /path/to/sleep-screen
mkdir -p output logs
```

Run a small smoke test first:

```bash
.venv/bin/python screen_time_extractor.py images/image1.png \
  --ocr-profile vl \
  --batch-size 1
```

Run the complete `images/` directory as a background job:

```bash
nohup .venv/bin/python screen_time_extractor.py 'images/*.png' \
  --ocr-profile vl \
  --batch-size 32 \
  --continue-on-error \
  > output/screen-time.jsonl \
  2> logs/screen-time.log \
  < /dev/null &

echo $! > output/screen-time.pid
```

The image glob is quoted deliberately. The extractor expands it internally,
avoiding the shell's command-length limit when the dataset contains many files.

To select a specific physical GPU:

```bash
CUDA_VISIBLE_DEVICES=1 nohup .venv/bin/python screen_time_extractor.py 'images/*.png' \
  --ocr-profile vl \
  --batch-size 32 \
  --continue-on-error \
  > output/screen-time.jsonl \
  2> logs/screen-time.log \
  < /dev/null &
```

Reduce `--batch-size` if the process runs out of GPU memory. Increase it only
after measuring throughput and memory use on the target machine.

## Monitor the job

The SSH connection can be closed after starting the `nohup` job. On the next
login:

```bash
tail -f logs/screen-time.log
ps -p "$(cat output/screen-time.pid)"
wc -l output/screen-time.jsonl
grep '"error"' output/screen-time.jsonl
```

Standard output is JSON Lines: one result or error object per image. For
example:

```json
{"image":"images/image1.png","date":"22 mars","total_minutes":231,"hourly_minutes":[4,0,0,0,0,0,0,16,36,24,11,6,1,3,23,24,0,2,16,9,7,42,7,0]}
```

## Validation and failed images

Each successful result is validated with Pydantic:

- `date` must contain a recognized Swedish calendar date. No year is inferred.
- `total_minutes` must be between 0 and 1440.
- `hourly_minutes` must contain exactly 24 values between 0 and 60.
- The hourly values must sum to the displayed daily total.

With `--continue-on-error`, an image that fails OCR, chart detection, or total
reconciliation produces an error object and does not stop the dataset job:

```json
{"image":"images/unusual.png","error":"ValueError","message":"Could not find chart grid lines"}
```

Review these images separately rather than treating them as valid zero-valued
records.

## Tests

The automated tests do not load the GPU OCR model:

```bash
.venv/bin/python -m unittest discover -s tests -v
```
