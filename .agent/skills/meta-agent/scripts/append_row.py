"""Append and verify an eight-column review record without changing history."""
import re
import os
import stat
import tempfile
from pathlib import Path
from check_mandate import CHANGE_LOG, _review_rows, table_cells
from eau_paths import canonical_run

COLUMNS = ("run", "step", "tier", "knob", "change", "evidence", "prediction", "verification")


def replace_text(path, text):
    """Keep an interrupted automated append from truncating existing history."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            if path.exists():
                os.fchmod(handle.fileno(), stat.S_IMODE(path.stat().st_mode))
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _clean(text):
    """Preserve displayed pipes while preventing extra cells and lines."""
    return " ".join(str(text).replace("|", "&#124;").split())


def append_row(*, run, step, tier, knob, change, evidence, prediction,
               verification="—", anchor=None, path=CHANGE_LOG, decision=None):
    """Insert after this run's last row, or the existing eight-column table header.

    `decision` optionally prefixes a review-v2 tag in the existing change cell.
    Verify visibility in the supplied path, including prediction-free reviews.
    """
    canonical_run(run)  # Resolve identity errors before writing the record.
    if decision is not None:
        change = f"decision: {decision}; {change}"
    cells = [run, step, tier, knob, change, evidence, prediction, verification]
    row = "| " + " | ".join(_clean(cell) for cell in cells) + " |"
    with open(path, encoding="utf-8") as handle:
        original = handle.read()
    if anchor:
        index = original.rfind(anchor)
        if index < 0:
            raise ValueError(f"Anchor not found: {anchor!r}")
        cut = original.find("\n", index)
        cut = len(original) if cut < 0 else cut + 1
    else:
        cut, header_cut, offset = None, None, 0
        for line in original.splitlines(keepends=True):
            offset += len(line)
            parsed = table_cells(line)
            if len(parsed) == 8 and parsed[0] == str(run) and parsed[1].isdigit():
                cut = offset
            if len(parsed) == 8 and all(re.fullmatch(r":?-+:?", cell) for cell in parsed):
                header_cut = offset
        cut = cut if cut is not None else header_cut
        if cut is None:
            raise ValueError("Initialize the eight-column review table before appending")
    prefix = original[:cut]
    if prefix and not prefix.endswith("\n"):
        prefix += "\n"
    updated = prefix + row + "\n" + original[cut:]
    replace_text(path, updated)
    if not any(r["raw"] == row and len(r["cells"]) == 8 for r in _review_rows(run, path)):
        replace_text(path, original)
        raise ValueError("New row is not visible in the supplied table; file rolled back")
    print(f"[append_row] {run} step {step}: written and verified visible (not a semantic review check)")
