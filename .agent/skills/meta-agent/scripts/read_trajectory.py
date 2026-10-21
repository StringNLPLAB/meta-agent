#!/usr/bin/env python3
"""Print complete trajectory text from an explicit dump file."""
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    with open(args.path, encoding="utf-8") as stream:
        data = json.load(stream)
    rows = data.get("trajectories", [data]) if isinstance(data, dict) else data
    row = rows[args.index]
    print(json.dumps(row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
