"""Product stock catalog: loaded once, looked up per question.

Lookup rule (also used by tests/check_results.py, so both agree):
  1. Find the product the shopper is looking at: product_title of product_sku.
  2. "exact": rows in the shopper's shop for that product AND the asked variant_size.
  3. "other_sizes": no row for that size, so give the product's other sizes
     in that shop (lets the model say "we don't stock S, but M and L are...").
  4. "sku_only": the product isn't sold in that shop, so give the SKU's own row.

Why not just look up product_sku? In the replay data the SKU's own row is a
different size from the one the shopper asks about about 81% of the time,
so a plain SKU lookup returns the wrong stock row most of the time.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data" / "product_stock_catalog_oct2026.csv"

STOCK_COLUMNS = [
    "sku", "product_title", "variant_size", "variant_color", "price_usd",
    "stock_on_hand", "restock_date", "shipping_days", "returns_window_days",
]


class Catalog:
    def __init__(self, path: Path = DATA):
        df = pd.read_csv(path)
        df["restock_date"] = df["restock_date"].fillna("")
        self.df = df
        self.by_sku = df.set_index("sku")
        self.by_product = {k: g for k, g in df.groupby(["shop_domain", "product_title"])}

    def lookup(self, shop_domain: str, product_sku: str | None, variant_size: str | None):
        """Return (rows as list of dicts, match_type)."""
        if not product_sku or product_sku not in self.by_sku.index:
            return [], "none"
        title = self.by_sku.loc[product_sku, "product_title"]
        product = self.by_product.get((shop_domain, title))
        if product is not None:
            if variant_size:
                exact = product[product["variant_size"] == variant_size]
                if len(exact):
                    return self._rows(exact), "exact"
            return self._rows(product), "other_sizes"
        return self._rows(self.df[self.df["sku"] == product_sku]), "sku_only"

    @staticmethod
    def _rows(frame: pd.DataFrame) -> list[dict]:
        return frame[STOCK_COLUMNS].to_dict("records")


def format_row(r: dict) -> str:
    stock = f"{r['stock_on_hand']} in stock" if r["stock_on_hand"] > 0 else (
        f"out of stock, restock {r['restock_date']}" if r["restock_date"] else "out of stock"
    )
    return (
        f"{r['sku']} | {r['product_title']} | size {r['variant_size']} | {r['variant_color']} | "
        f"${r['price_usd']:.2f} | {stock} | ships in {r['shipping_days']} days | "
        f"{r['returns_window_days']}-day returns"
    )
