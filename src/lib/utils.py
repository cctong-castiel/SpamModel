# coding=utf-8
# func.py

import hashlib
import json
import pandas as pd
import numpy as np
import datetime


def get_digest(file_path):
    h = hashlib.sha256()

    with open(file_path, 'rb') as file:
        while True:
            # Reading is buffered, so we can read smaller chunks.
            chunk = file.read(h.block_size)
            if not chunk:
                break
            h.update(chunk)

    return h.hexdigest()

def get_dData(file_path, flag, sheetname=None):

    if flag == "json":
        with open(file_path, "r", encoding="utf-8") as f:
            dData = json.load(f)
    elif flag == "xlsx":
        dData = pd.read_excel(file_path, sheet_name=sheetname)

    return dData

def fake_ylabel(dData):

    old_y = dData["ylabel"]
    new_y = [1 if i == "positive" else 0 for i in old_y]

    dData["ylabel"] = new_y

    return dData

def myconverter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime.datetime):
        return obj.__str__()
