"""Send 20 replay questions to /ask at the same time and save every result.

Usage: python tests/load_test.py [--n 20] [--seed 42] [--url http://localhost:8000]
Writes results/load_test_<YYYYMMDD-HHMM>.jsonl (one line per question)
and    results/load_test_<YYYYMMDD-HHMM>_summary.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "data" / "shopper_question_replay.csv"


class GpuPeak(threading.Thread):
    """Polls nvidia-smi so the summary can report peak GPU memory."""

    def __init__(self):
        super().__init__(daemon=True)
        self.peak_mib, self.running = 0, True

    def run(self):
        while self.running:
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                ).stdout.strip().splitlines()
                self.peak_mib = max(self.peak_mib, int(out[0]))
            except Exception:
                pass
            time.sleep(0.5)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


async def one(client: httpx.AsyncClient, url: str, q: dict, run_id: str) -> dict:
    line = {
        "run_id": run_id,
        "question_id": q["question_id"],
        "session_id": int(q["session_id"]),
        "shop_domain": q["shop_domain"],
        "intent": q["intent"],
        "product_sku": q["product_sku"],
        "variant_size": q["variant_size"],
        "needs_stock_lookup": bool(q["needs_stock_lookup"]),
        "question_text": q["question_text"],
        "baseline_reply_seconds": float(q["baseline_reply_seconds"]),
        "started_at": now(),
    }
    t0 = time.perf_counter()
    try:
        r = await client.post(f"{url}/ask", json={
            "shop_domain": q["shop_domain"], "session_id": int(q["session_id"]),
            "question_text": q["question_text"], "product_sku": q["product_sku"],
            "variant_size": q["variant_size"],
        })
        if r.status_code == 503:
            line.update(status="timeout", error=r.text)
        else:
            r.raise_for_status()
            line.update(status="ok", error=None, **r.json())
    except Exception as e:
        line.update(status="error", error=repr(e))
    line["finished_at"] = now()
    line["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)  # end-to-end, overrides server value
    return line


async def main(n: int, seed: int, url: str):
    qs = pd.read_csv(QUESTIONS).sample(n=n, random_state=seed).to_dict("records")
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    run_id = f"run-{stamp}-seed{seed}"
    gpu = GpuPeak()
    gpu.start()

    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=60) as client:
        lines = await asyncio.gather(*(one(client, url, q, run_id) for q in qs))
    wall = time.perf_counter() - t0
    gpu.running = False

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    jsonl = out_dir / f"load_test_{stamp}.jsonl"
    with jsonl.open("w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

    ok = [l for l in lines if l["status"] == "ok"]
    lat = pd.Series([l["latency_ms"] for l in ok])
    reply_tokens = sum(l.get("reply_tokens", 0) for l in ok)
    summary = {
        "run_id": run_id,
        "model": os.environ.get("MODEL_NAME"),
        "max_model_len": 1024,
        "max_num_seqs": 20,
        "seed": seed,
        "questions": n,
        "ok": len(ok),
        "errors": n - len(ok),
        "p50_latency_ms": round(lat.quantile(0.5), 1) if len(lat) else None,
        "p95_latency_ms": round(lat.quantile(0.95), 1) if len(lat) else None,
        "wall_seconds": round(wall, 2),
        "reply_tokens_per_sec": round(reply_tokens / wall, 1),
        "peak_gpu_memory_mib": gpu.peak_mib or None,
    }
    (out_dir / f"load_test_{stamp}_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"{'question_id':<10} {'status':<8} {'latency_s':>9} {'baseline_s':>10}  match_type")
    for l in lines:
        print(f"{l['question_id']:<10} {l['status']:<8} {l['latency_ms']/1000:>9.2f} "
              f"{l['baseline_reply_seconds']:>10.2f}  {l.get('match_type', '-')}")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved {jsonl.relative_to(ROOT)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--url", default="http://localhost:8000")
    a = p.parse_args()
    asyncio.run(main(a.n, a.seed, a.url))
