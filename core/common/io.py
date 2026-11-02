"""Shared I/O helpers: JSON read/write, CSV append."""

import csv
import json
import os
from collections import OrderedDict


def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def save_json(data, json_path, indent=4):
    parent = os.path.dirname(str(json_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=indent)


def write_csv_line(result_file_path, result):
    """Write a line in a csv file; create the file and write the header if it does not already exist."""
    result = OrderedDict(result)
    file_exists = os.path.exists(result_file_path)
    with open(result_file_path, 'a') as csv_file:
        writer = csv.DictWriter(csv_file, result.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


__all__ = ["read_json", "save_json", "write_csv_line"]

