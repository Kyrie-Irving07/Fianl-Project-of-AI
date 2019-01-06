import json
import numpy as np

def get_dict(path):
    list = json.load(open(path), 'r')
    return list
