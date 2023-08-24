import json
import os

def load_responses_from_database():
    all_queries = []
    all_responses = []
    mapping = {}
    index = 0
    
    files = os.listdir("databases")
    for file_name in files:
        if file_name.endswith(".json"):
            with open(os.path.join("databases", file_name), "r", encoding="utf-8") as file:
                data = json.load(file)
                all_queries.extend(data["queries"])
                all_responses.extend(data["responses"])
                for i, response in enumerate(data["responses"]):
                    mapping[index + i] = file_name.replace(".json", "")
                index += len(data["responses"])

    return all_queries, all_responses, mapping
