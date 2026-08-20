import json
from pathlib import Path

file_path = max(Path("output").glob("screen-time-*.jsonl"), key=lambda path: path.stat().st_mtime)
data = []
with file_path.open("r", encoding="utf-8") as file:
    for line in file:
        # Parse each line individually and append to the list
        data.append(json.loads(line))
        data[-1]
        assert int(data[-1]["total_minutes"]) == sum([int(x) for x in data[-1]["hourly_minutes"]])
