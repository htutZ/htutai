import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOLDER = os.path.join(BASE, "databases")


def load_responses_from_database():
    all_queries, all_responses, mapping = [], [], {}
    index = 0
    for file_name in sorted(os.listdir(FOLDER)):
        if not file_name.endswith(".json"):
            continue
        try:
            with open(os.path.join(FOLDER, file_name), "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if not isinstance(data, dict) or "queries" not in data or "responses" not in data:
            continue
        qs, rs = data["queries"], data["responses"]
        n = min(len(qs), len(rs))
        all_queries.extend(qs[:n])
        all_responses.extend(rs[:n])
        for i in range(n):
            mapping[index + i] = file_name.replace(".json", "")
        index += n
    return all_queries, all_responses, mapping