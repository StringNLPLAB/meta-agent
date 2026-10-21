"""Validate the eight-column review table, including escaped Markdown pipes."""
import argparse
from check_mandate import CHANGE_LOG, _review_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=CHANGE_LOG)
    args = parser.parse_args()
    # Missing logs are errors here, rather than a claim that a table is valid.
    with open(args.path, encoding="utf-8"):
        pass
    rows = _review_rows(path=args.path)
    bad = [row for row in rows if len(row["cells"]) != 8]
    for row in bad:
        print(f"[TABLE] line {row['line']}: {row['run']} step {row['step']} "
              f"has {len(row['cells'])} columns, expected 8")
    print(f"[TABLE] {len(rows)} review/history rows, {len(bad)} malformed")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
