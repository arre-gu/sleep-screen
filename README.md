# Screen Time screenshot extraction

Extract three values from each Apple Screen Time screenshot:

- Swedish date
- Total screen time in minutes
- 24 hourly screen-time values in minutes

The images must be in `images/`. Each machine processes its own local files;
there is no OCR server or communication between machines.

Choose one setup:

1. Windows laptop, CPU
2. Linux ARM GPU, NVIDIA DGX Spark

## Windows laptop — CPU

### Install

Open PowerShell in the project directory. Install `uv` if needed:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart PowerShell if `uv` is not immediately available, then install the CPU
OCR environment:

```powershell
uv sync --extra ocr-cpu
```

This installs PaddleOCR with the mobile detector and Swedish-aware Latin text
recognizer. The first processing run downloads their weights into
`.cache/paddlex`.

### Process images

Create output directories and process every PNG in `images/`:

```powershell
New-Item -ItemType Directory -Force output, logs

uv run python screen_time_extractor.py 'images\*.png' `
  --ocr-profile cpu `
  --batch-size 16 `
  --continue-on-error `
  2> logs\screen-time.log
```

For a quick test with one image:

```powershell
uv run python screen_time_extractor.py images\image1.png `
  --ocr-profile cpu `
  --batch-size 1
```

## Linux ARM GPU — NVIDIA DGX Spark

The DGX Spark path uses PyTorch with CUDA 13 and the Transformers implementation
of `PaddlePaddle/PaddleOCR-VL-1.6`. It does not install the PaddlePaddle GPU
wheel, because the standard wheel is not built for the DGX Spark's ARM64
architecture.

### Verify the machine

Run these commands on the DGX Spark:

```bash
uname -m
nvidia-smi
```

`uname -m` should report `aarch64` and `nvidia-smi` should show the GPU.

### Install

Open a shell in the project directory. Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

Install the DGX environment. The project configuration selects PyTorch's CUDA
13.0 package index automatically:

```bash
uv sync --extra ocr-dgx
```

Verify that PyTorch can use the GPU:

```bash
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

The first processing run downloads `PaddleOCR-VL-1.6` into the Hugging Face
cache under `~/.cache/huggingface/`.

### Process images

Create output directories and process every PNG in `images/`:

```bash
mkdir -p output logs

uv run python screen_time_extractor.py 'images/*.png' \
  --ocr-profile gpu \
  --batch-size 8 \
  --continue-on-error \
  2> logs/screen-time.log
```

For a quick test with one image:

```bash
uv run python screen_time_extractor.py images/image1.png \
  --ocr-profile gpu \
  --batch-size 1
```

To keep a long job running after disconnecting from SSH:

```bash
nohup uv run python screen_time_extractor.py 'images/*.png' \
  --ocr-profile gpu \
  --batch-size 8 \
  --continue-on-error \
  > /dev/null \
  2> logs/screen-time.log \
  < /dev/null &

echo $! > output/screen-time.pid
```

Check the background job later with:

```bash
ps -p "$(cat output/screen-time.pid)"
tail -f logs/screen-time.log
```

## Output

Each run creates a timestamped file such as
`output/screen-time-20260820-143052.jsonl`. It contains one JSON object per
image. The timestamp uses local machine time in `yyyyMMdd-HHmmss` format and is
captured by `screen_time_extractor.py` immediately before processing starts.
Use `--output-dir PATH` to save the timestamped file in another directory.

```json
{"image":"images/image1.png","date":"22 mars","total_minutes":231,"hourly_minutes":[4,0,0,0,0,0,0,16,36,24,11,6,1,3,23,24,0,2,16,9,7,42,7,0]}
```

With `--continue-on-error`, failed images are recorded instead of stopping the
job:

```json
{"image":"images/unusual.png","error":"ValueError","message":"Could not find chart grid lines"}
```

Find failed images with:

```bash
grep '"error"' output/screen-time-*.jsonl
```

### Detection sanity checks

After an image is parsed and validated successfully, the script saves an
annotated copy with the same filename in `boxes/`:

- Green rectangle: detected 24-hour bar chart
- Magenta rectangle: derived date and total screen-time text region

Images that fail parsing do not get an annotated copy. The script creates the
directory automatically. Use `--boxes-dir PATH` to save these images elsewhere.

## Validation

Successful records are validated before they are written:

- The date must contain a recognized Swedish day and month.
- Total minutes must be between 0 and 1440.
- The hourly array must contain exactly 24 values between 0 and 60.
- The hourly values must sum to the total minutes.

## Tests

Windows PowerShell:

```powershell
uv run python -m unittest discover -s tests -v
```

Linux:

```bash
uv run python -m unittest discover -s tests -v
```
