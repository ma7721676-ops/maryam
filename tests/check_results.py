"""Check a load-test results file against the source data.

Usage: python tests/check_results.py [results/load_test_<stamp>.jsonl]
(defaults to the newest results file)

Checks:
  - exactly 20 lines, each with a unique question_id from the replay CSV
  - shop, SKU and size on each line match that question's row in the replay CSV
  - the stock rows sent to the model are the right catalog rows for that
    product_sku + variant_size (recomputed here independently of app/catalog.py)
  - every request finished with status "ok"
  - the 20 requests overlapped in time (they really ran at once)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
replay = pd.read_csv(ROOT / "data" / "shopper_question_replay.csv").set_index("question_id")
cat = pd.read_csv(ROOT / "data" / "product_stock_catalog_oct2026.csv")


def expected(shop: str, sku: str, size: str) -> tuple[set[str], str]:
    title = cat.loc[cat.sku == sku, "product_title"].iloc[0]
    product = cat[(cat.shop_domain == shop) & (cat.product_title == title)]
    if len(product):
        exact = product[product.variant_size == size]
        if len(exact):
            return set(exact.sku), "exact"
        return set(product.sku), "other_sizes"
    return {sku}, "sku_only"


def main(path: Path) -> int:
    lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    problems = []
    if len(lines) != 20:
        problems.append(f"expected 20 lines, found {len(lines)}")
    ids = [l.get("question_id") for l in lines]
    if len(set(ids)) != len(ids):
        problems.append("duplicate question_ids")

    for l in lines:
        qid = l.get("question_id")
        if qid not in replay.index:
            problems.append(f"{qid}: not in the replay CSV")
            continue
        src = replay.loc[qid]
        for col in ("shop_domain", "product_sku", "variant_size"):
            if l.get(col) != src[col]:
                problems.append(f"{qid}: {col} is {l.get(col)!r}, CSV says {src[col]!r}")
        if l.get("status") != "ok":
            problems.append(f"{qid}: status {l.get('status')} ({l.get('error')})")
            continue
        skus, kind = expected(src.shop_domain, src.product_sku, src.variant_size)
        got = set(l.get("stock_skus", []))
        if l.get("match_type") != kind:
            problems.append(f"{qid}: match_type {l.get('match_type')}, expected {kind}")
        if got != skus:
            problems.append(f"{qid}: stock rows {sorted(got)}, expected {sorted(skus)}")
        if kind == "exact":
            sizes = set(cat[cat.sku.isin(got)].variant_size)
            if sizes != {src.variant_size}:
                problems.append(f"{qid}: rows are sizes {sizes}, shopper asked {src.variant_size}")

    ok = [l for l in lines if l.get("status") == "ok"]
    if ok and max(l["started_at"] for l in ok) >= min(l["finished_at"] for l in ok):
        problems.append("requests did not all overlap: some started after others finished")

    print(f"Checked {path.relative_to(ROOT)}: {len(lines)} lines")
    for p in problems:
        print("  FAIL", p)
    print("ALL CHECKS PASSED" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = Path(sys.argv[1]).resolve()
    else:
        files = sorted((ROOT / "results").glob("load_test_*[0-9].jsonl"))
        if not files:
            sys.exit("No results files in results/")
        target = files[-1]
    sys.exit(main(target))
