import json


def read_jsonl(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows, mode="w"):
    with open(path, mode) as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)


def append_jsonl(path, rows):
    write_jsonl(path, rows, "a")


def load_jsonl(path, key):
    return {row[key]: row for row in read_jsonl(path) if "error" not in row}
