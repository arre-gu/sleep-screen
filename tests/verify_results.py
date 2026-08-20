import os
import json

file_path = os.path.join("output","screen-time.jsonl")
data = []
with open(file_path, "r", encoding="utf-16") as file:
    for line in file:
        # Parse each line individually and append to the list
        data.append(json.loads(line))
        data[-1]
        assert int(data[-1]["total_minutes"]) == sum([int(x) for x in data[-1]["hourly_minutes"]])