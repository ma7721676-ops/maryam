"""FastAPI endpoint that answers shopper questions through a vLLM server.

Run:  uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
Env:  MODEL_NAME (must match the model vLLM is serving)
      VLLM_URL   (default http://localhost:8001/v1)
"""
from __future__ import annotations

import asyncio
import os
import time

import httpx
from fastapi import FastAPI, HTTPException
from openai import AsyncOpenAI
from pydantic import BaseModel
from transformers import AutoTokenizer

from app.catalog import Catalog, format_row

MODEL_NAME = os.environ["MODEL_NAME"]
VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8001/v1")

MAX_MODEL_LEN = 1024      # longest prompt 216 + longest reply 177 = 393; ~630 left for stock rows
MAX_REPLY_TOKENS = 200
MAX_CONCURRENT = 20       # matches vLLM --max-num-seqs 20
TIMEOUT_S = 30

SYSTEM_PROMPT = (
    "You are a helpful shop assistant. Answer the shopper's question in 1-3 short "
    "sentences using only the stock rows given. If the asked size isn't listed, say so "
    "and mention the sizes that are. Never invent stock, prices or dates."
)

app = FastAPI(title="Shopper question assistant")
client = AsyncOpenAI(base_url=VLLM_URL, api_key="not-needed")
gate = asyncio.Semaphore(MAX_CONCURRENT)
catalog = Catalog()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class Ask(BaseModel):
    shop_domain: str
    session_id: int | None = None
    question_text: str
    product_sku: str | None = None
    variant_size: str | None = None


def build_messages(q: Ask, rows: list[dict]) -> list[dict]:
    stock = "\n".join(format_row(r) for r in rows) or "No matching stock rows."
    user = (
        f"Shop: {q.shop_domain}\nViewing SKU: {q.product_sku}  Asked size: {q.variant_size}\n"
        f"Stock rows:\n{stock}\n\nQuestion: {q.question_text}"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def count_tokens(messages: list[dict]) -> int:
    return len(tokenizer.apply_chat_template(messages, add_generation_prompt=True))


def fit_budget(q: Ask, rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Drop the least relevant (last) rows until prompt + reply fits the context."""
    rows = list(rows)
    messages = build_messages(q, rows)
    while rows and count_tokens(messages) + MAX_REPLY_TOKENS > MAX_MODEL_LEN:
        rows.pop()
        messages = build_messages(q, rows)
    return messages, rows


@app.post("/ask")
async def ask(q: Ask):
    rows, match_type = catalog.lookup(q.shop_domain, q.product_sku, q.variant_size)
    messages, rows = fit_budget(q, rows)
    start = time.perf_counter()
    async with gate:
        try:
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=MODEL_NAME, messages=messages,
                    max_tokens=MAX_REPLY_TOKENS, temperature=0.2,
                ),
                timeout=TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            raise HTTPException(503, f"Model did not answer within {TIMEOUT_S}s")
    return {
        "reply": resp.choices[0].message.content,
        "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        "prompt_tokens": resp.usage.prompt_tokens,
        "reply_tokens": resp.usage.completion_tokens,
        "stock_rows_used": len(rows),
        "stock_skus": [r["sku"] for r in rows],
        "match_type": match_type,
    }


@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=5) as h:
            r = await h.get(f"{VLLM_URL}/models")
        r.raise_for_status()
        return {"status": "ok", "model": MODEL_NAME}
    except Exception as e:
        raise HTTPException(503, f"vLLM server not reachable: {e}")
