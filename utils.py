import json

def get_hypers_model():
    with open("./config.json") as f:
        data = json.load(f)["model"]
    return data

