"""Amazon Reviews 2014 (5-core) loading, leave-last-out splits, and item text.

Files are the standard SNAP 5-core release (McAuley et al.) used by TIGER and
most semantic-ID papers. They are downloaded on first use into ``<root>/amazon``.
"""
from __future__ import annotations

import ast
import gzip
import hashlib
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import requests
from datasets import Dataset, DatasetDict, Features, Sequence, Value, load_from_disk
from tqdm.auto import tqdm

_SNAP = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles"

AMAZON2014_CATEGORIES = (
    "Beauty",
    "Toys_and_Games",
    "Sports_and_Outdoors",
    "Health_and_Personal_Care",
    "Home_and_Kitchen",
)


def _urls(category: str) -> Tuple[str, str]:
    if category not in AMAZON2014_CATEGORIES:
        raise ValueError(f"Unknown category={category!r}. Choose one of: {list(AMAZON2014_CATEGORIES)}")
    return (
        f"{_SNAP}/reviews_{category}_5.json.gz",
        f"{_SNAP}/meta_{category}.json.gz",
    )


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def _parse_gz_json_lines(path: str):
    """SNAP files are one Python-literal dict per line (meta files are not strict JSON)."""
    with gzip.open(path, "rb") as g:
        for line in g:
            line = line.replace(b"true", b"True").replace(b"false", b"False")
            yield ast.literal_eval(line.decode("utf-8"))


def _download_file(url: str, path: str, *, timeout: int = 60) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)


def _ensure_file(url: str, amazon_dir: str, download_if_missing: bool) -> str:
    fp = os.path.join(amazon_dir, url.split("/")[-1])
    if not os.path.exists(fp):
        if not download_if_missing:
            raise FileNotFoundError(f"Missing {fp}. Set data.download_if_missing=true or place the file there.")
        print(f"{os.path.basename(fp)} not found, downloading...")
        _download_file(url, fp)
    return fp


# ---------------------------------------------------------------------------
# Interactions -> sequential splits
# ---------------------------------------------------------------------------
def _build_user_sequences_from_snap_reviews(
    reviews_gz_path: str,
    *,
    rating_score: float = 0.0,
) -> Dict[str, List[str]]:
    """user -> [asin, ...] sorted by unixReviewTime (keeps interactions with overall > rating_score)."""
    tmp: Dict[str, List[Tuple[int, str]]] = {}
    for inter in _parse_gz_json_lines(reviews_gz_path):
        if float(inter["overall"]) <= rating_score:
            continue
        tmp.setdefault(inter["reviewerID"], []).append((int(inter["unixReviewTime"]), inter["asin"]))

    user_items: Dict[str, List[str]] = {}
    for u, pairs in tmp.items():
        pairs.sort(key=lambda x: x[0])
        user_items[u] = [it for _, it in pairs]
    return user_items


def _build_leave_last_out_splits(user_items: Dict[str, List[str]]) -> DatasetDict:
    """
    Leave-last-out protocol:
      test       = (seq[:-1] -> seq[-1])
      validation = (seq[:-2] -> seq[-2])
      train      = every earlier prefix (seq[:t] -> seq[t]) for t = 1 .. n-3
    Sequences of length 2/3 (absent under 5-core) contribute to train/validation only.
    """
    splits = {s: {"user_id": [], "history": [], "label": []} for s in ("train", "validation", "test")}

    def add(split, user, hist, label):
        splits[split]["user_id"].append(user)
        splits[split]["history"].append(hist)
        splits[split]["label"].append(label)

    for user, seq in user_items.items():
        n = len(seq)
        if n < 2:
            continue
        if n == 2:
            add("train", user, [seq[0]], seq[1])
            continue
        if n == 3:
            add("train", user, [seq[0]], seq[1])
            add("validation", user, seq[:2], seq[2])
            continue
        for t in range(1, n - 2):
            add("train", user, seq[:t], seq[t])
        add("validation", user, seq[:-2], seq[-2])
        add("test", user, seq[:-1], seq[-1])

    return DatasetDict({s: Dataset.from_dict(cols) for s, cols in splits.items()})


def print_split_stats(dd: DatasetDict) -> None:
    print("\n=== Sequence length statistics ===")
    for split, ds in dd.items():
        lengths = np.fromiter((len(h) for h in ds["history"]), dtype=np.int32)
        if len(lengths) == 0:
            print(f"{split:>10} | empty")
            continue
        print(
            f"{split:>10} | n={len(lengths):,}  min={lengths.min():>3d}  max={lengths.max():>3d}  "
            f"mean={lengths.mean():>6.2f}  median={np.median(lengths):>6.2f}"
        )


def build_amazon2014_seq_splits(
    category: str,
    *,
    root: str = "./raw_data",
    rating_score: float = 0.0,
    download_if_missing: bool = True,
) -> Tuple[DatasetDict, Dict[str, int], Dict[int, str]]:
    """Returns (splits with ASIN histories/labels, asin -> item_id, item_id -> asin)."""
    amazon_dir = os.path.join(os.path.abspath(root), "amazon")
    os.makedirs(amazon_dir, exist_ok=True)

    reviews_url, meta_url = _urls(category)
    reviews_gz = _ensure_file(reviews_url, amazon_dir, download_if_missing)
    _ensure_file(meta_url, amazon_dir, download_if_missing)

    user_items = _build_user_sequences_from_snap_reviews(reviews_gz, rating_score=rating_score)

    uniques = sorted({it for seq in user_items.values() for it in seq})
    item2token = {asin: i for i, asin in enumerate(uniques, start=1)}
    token2item = {i: asin for asin, i in item2token.items()}

    dd = _build_leave_last_out_splits(user_items)
    print(f"[Amazon2014:{category}] users={len(user_items):,} items={len(uniques):,} "
          f"splits={ {k: len(v) for k, v in dd.items()} }")
    return dd, item2token, token2item


# ---------------------------------------------------------------------------
# Metadata -> item text
# ---------------------------------------------------------------------------
def _coerce_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)):
        return " ".join(str(t) for t in x if t)
    return str(x)


def _coerce_categories(v: Any):
    """SNAP meta uses list[list[str]] category paths; a flat list[str] is read as one path."""
    if isinstance(v, list):
        if all(isinstance(x, list) for x in v):
            return [[str(s) for s in path if s is not None and str(s).strip() != ""] for path in v]
        if all(isinstance(x, str) for x in v):
            path = [s.strip() for s in v if s and s.strip()]
            return [path] if path else []
    return []


def _coerce_price(v: Any):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if not s or s.lower() in ("none", "nan"):
            return None
        try:
            return float(s.replace("$", "").replace(",", ""))
        except ValueError:
            return None
    return None


def load_amazon2014_meta(
    category: str,
    *,
    root: str = "./raw_data",
    only_asins: Optional[Set[str]] = None,
    download_if_missing: bool = True,
    keep_columns: Optional[Iterable[str]] = None,
) -> Dataset:
    """HF Dataset over meta_<category>.json.gz, restricted to `only_asins` (cached on disk)."""
    amazon_dir = os.path.join(os.path.abspath(root), "amazon")
    os.makedirs(amazon_dir, exist_ok=True)
    meta_path = _ensure_file(_urls(category)[1], amazon_dir, download_if_missing)

    keep_columns = list(dict.fromkeys(keep_columns or ["asin", "title", "description", "categories", "price"]))
    if "asin" not in keep_columns:
        keep_columns = ["asin"] + keep_columns

    only = None if only_asins is None else set(only_asins)
    target = 0 if only is None else len(only)

    # cache key: (category, requested ASIN set, kept columns)
    cols_digest = hashlib.sha1("|".join(keep_columns).encode("utf-8")).hexdigest()[:16]
    if only is None:
        asin_key = "all"
    else:
        h = hashlib.sha1()
        for asin in sorted(only):
            h.update(str(asin).encode("utf-8"))
            h.update(b"\0")
        asin_key = f"filtered_{target}_{h.hexdigest()[:16]}"
    cache_path = os.path.join(amazon_dir, f".meta_cache_{category}_{asin_key}_{cols_digest}")
    if os.path.isdir(cache_path):
        meta_ds = load_from_disk(cache_path)
        print(f"[Amazon2014:{category}] meta rows loaded from cache: {len(meta_ds):,}")
        return meta_ds

    def _feature_for(key: str):
        if key == "categories":
            return Sequence(Sequence(Value("string")))
        if key == "price":
            return Value("float32")
        return Value("string")

    def _coerce(key: str, v: Any):
        if key in ("title", "description"):
            return _coerce_text(v).strip() if v is not None else ""
        if key == "categories":
            return _coerce_categories(v)
        if key == "price":
            return _coerce_price(v)
        return v

    rows: List[Dict[str, Any]] = []
    matched: Set[str] = set()
    for info in _parse_gz_json_lines(meta_path):
        asin = info.get("asin")
        if not asin or (only is not None and asin not in only):
            continue
        row = {"asin": asin}
        for key in keep_columns:
            if key != "asin":
                row[key] = _coerce(key, info.get(key, None))
        rows.append(row)
        if only is not None:
            matched.add(asin)
            if len(matched) >= target:
                break

    meta_ds = Dataset.from_list(rows, features=Features({k: _feature_for(k) for k in keep_columns}))
    try:
        meta_ds.save_to_disk(cache_path)
    except Exception as e:  # cache is best-effort
        print(f"[Amazon2014:{category}] warning: failed to save meta cache ({e})")
    print(f"[Amazon2014:{category}] meta rows loaded: {len(meta_ds):,}")
    return meta_ds


def build_asin2text(meta: Dataset, asin_set: Set[str], cols: List[str]) -> Dict[str, str]:
    """asin -> 'title: ... | categories: a > b || c > d | description: ...'."""
    asin2text: Dict[str, str] = {}
    for ex in tqdm(meta, desc="meta->text"):
        asin = ex.get("asin")
        if not asin or asin not in asin_set:
            continue

        parts = []
        for col in cols:
            if col not in meta.column_names:
                continue
            val = ex.get(col)
            if val is None or val in ("", "None", [], {}):
                continue
            if col == "categories":
                if isinstance(val, list):
                    paths = [
                        " > ".join(str(x).strip() for x in path if str(x).strip())
                        for path in val
                        if isinstance(path, list) and path
                    ]
                    val = " || ".join(paths)
                else:
                    val = ""
            elif isinstance(val, list):
                val = ". ".join(x.strip() for x in val if isinstance(x, str) and x.strip())
            elif isinstance(val, dict):
                val = json.dumps(val, ensure_ascii=False)
            else:
                val = str(val).strip()
            if val:
                parts.append(f"{col}: {val}")

        asin2text[asin] = " | ".join(parts) if parts else "missing_meta"
        if len(asin2text) >= len(asin_set):
            break
    return asin2text
