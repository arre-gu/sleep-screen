import json
from pathlib import Path

file_path = max(Path("output").glob("screen-time-*.jsonl"), key=lambda path: path.stat().st_mtime)
failed = []
with file_path.open("r", encoding="utf-8") as file:
    for line in file:
        # Parse each line individually and append to the list
        line_data =  json.loads(line)
        try:
            assert int(line_data["total_minutes"]) == sum([int(x) for x in line_data["hourly_minutes"]])
        except:
            failed.append(line_data["image"])



if len(failed):
    print(f"{len(failed)} images failed parsing")
elif len(failed) == 0:
    print("Successfully parsed all data")
else:
    print("Failed images",failed)
