"""Summarize a hyperparameter search: one row per run, best validation NDCG@10 first.

    python summarize_runs.py multirun/            # or outputs/, or any folder of runs
    python summarize_runs.py multirun/ --category Toys_and_Games --csv toys_hpo.csv

Reads every ``test_metrics.json`` below the given folders. Model selection should use the
validation columns (``val_*``, metrics of the checkpoint selected during training); the
``test_*`` columns are reported for the selected configuration.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

COLUMNS = [
    ("category", "dataset"),
    ("m", "m"),
    ("min_share_prob", "rho"),
    ("use_item_mass", "mass"),
    ("val_ndcg@10", "val N@10"),
    ("test_recall@5", "R@5"),
    ("test_recall@10", "R@10"),
    ("test_ndcg@5", "N@5"),
    ("test_ndcg@10", "N@10"),
    ("best_epoch", "epoch"),
]


def fmt(key: str, value) -> str:
    if value is None:
        return "-"
    if key.startswith(("val_", "test_")):
        return f"{100 * float(value):.2f}"
    if key == "best_epoch":
        return f"{float(value):.0f}"
    return str(value)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+", type=Path)
    ap.add_argument("--category", help="keep only this data.category")
    ap.add_argument("--csv", type=Path, help="also write all rows to this CSV file")
    args = ap.parse_args()

    rows = []
    for folder in args.folders:
        for f in sorted(folder.rglob("test_metrics.json")):
            try:
                r = json.loads(f.read_text())
            except (OSError, json.JSONDecodeError) as e:
                print(f"skipping {f}: {e}", file=sys.stderr)
                continue
            r["run_dir"] = str(f.parent)
            if args.category is None or r.get("category") == args.category:
                rows.append(r)
    if not rows:
        raise SystemExit("no test_metrics.json found")
    rows.sort(key=lambda r: (r.get("category", ""), -float(r.get("val_ndcg@10", float("-inf")))))

    print("| " + " | ".join(h for _, h in COLUMNS) + " |")
    print("|" + "---|" * len(COLUMNS))
    for r in rows:
        print("| " + " | ".join(fmt(k, r.get(k)) for k, _ in COLUMNS) + " |")
    print("\nMetrics in %. Rows are sorted by validation NDCG@10 within each dataset; pick the top row.")

    if args.csv:
        keys = sorted({k for r in rows for k in r})
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
