import json

def get_hypers_model():
    with open("/home/nhqcs/Desktop/Github/DrugDesign/config.json") as f:
        data = json.load(f)["model"]
    return data

