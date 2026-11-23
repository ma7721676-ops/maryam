# Shop assistant on a free Colab T4

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ma7721676-ops/maryam/blob/main/notebooks/serve_model.ipynb)

Answers shopper questions (stock, sizing, shipping, returns) with an LLM served by vLLM on one free Colab T4, behind a FastAPI endpoint that handles 20 shoppers at once.

## Layout

| Path | What it is |
|---|---|
| `notebooks/serve_model.ipynb` | Run top to bottom in Colab: starts vLLM, starts the API, runs the 20-shopper test, checks it, shows the saved results |
| `app/main.py` | FastAPI `POST /ask` and `GET /health` |
| `app/catalog.py` | Loads the stock catalog once and picks the rows for each question |
| `tests/load_test.py` | Sends 20 replay questions at once, saves `results/load_test_<date>.jsonl` and a summary |
| `tests/check_results.py` | Verifies the results file against the source CSVs |
| `data/` | Where the Shopper Question Replay and Product Stock Catalog CSVs go (see `data/README.md`) |
| `results/` | Saved load-test runs |

## Settings and why

- **Context 1,024 tokens.** Longest prompt 216 + longest reply 177 = 393 tokens, leaving about 630 for stock rows. The largest product in the catalog has 14 variants, about 560 tokens, so it fits.
- **fp16 (`--dtype half`).** The T4 has no bf16 support.
- **20 concurrent sequences.** `--max-num-seqs 20` in vLLM, matched by `asyncio.Semaphore(20)` in the API. A short context keeps each shopper's KV cache small, so all 20 fit in the T4's 16 GB at once instead of queuing.

## Key Claude Code prompt

```
Add a FastAPI app to the same repo (app/main.py) that answers shopper
questions using the vLLM server from notebooks/serve_model.ipynb.

Endpoint
- POST /ask with JSON: shop_domain, session_id, question_text,
  product_sku (optional), variant_size (optional)
- Returns: reply text, latency_ms, prompt_tokens, reply_tokens

Concurrency (20 shoppers at once)
- Fully async: use AsyncOpenAI pointed at the vLLM OpenAI-compatible
  server, never a blocking client or a sync def route
- Cap in-flight model calls with asyncio.Semaphore(20) to match
  max_num_seqs=20; extra requests wait instead of failing
- Per-request timeout of 30 s, returning a clear 503 if exceeded
- Run with one uvicorn worker (one GPU, one model server)

Stock lookup
- Load the Product Stock Catalog into memory once at startup
- Only include rows for the shopper's shop and product (max 14 variants,
  ~560 tokens), never the whole catalog

Token budget (context is 1,024)
- max_tokens = 200 for the reply
- If prompt + 200 would exceed 1,024, drop the least relevant stock rows first

Saving the run
- Write every result to results/load_test_<YYYYMMDD-HHMM>.jsonl, one line
  per question, with question_id, session_id, shop_domain, intent,
  product_sku, variant_size, question_text, reply, stock_rows_used,
  prompt/reply tokens, started_at, finished_at, latency_ms,
  baseline_reply_seconds and status
- Also write a summary with p50/p95 latency, error count and peak GPU memory
- Pick the 20 questions with a fixed seed so the run can be repeated
```

## What I checked and changed

**The stock lookup.** The obvious way to find a question's stock row is to look up its `product_sku`. When I checked that against the data, the SKU in a question points to one specific variant row, and that row's size differs from the size the shopper is asking about in about 81% of questions (for example, viewing SKU-1039, an XXL, while asking "Do you have this in S?"). A plain SKU lookup would have given the model the wrong stock row most of the time, so it would confidently answer about the wrong size.

I changed it to: take the product of that SKU, then find the rows in the shopper's own shop for the size they asked about. If that size isn't stocked, send the product's other sizes so the model can say what is available. `tests/check_results.py` recomputes this independently and fails the run if any line has the wrong rows. Run against the old SKU-only lookup, it flags 34 problems across the 20 questions.

## Results

| Run | Model | p50 latency | p95 latency | Errors | Peak GPU memory |
|---|---|---|---|---|---|
| [`run-20261123-1647-seed42`](results/load_test_20261123-1647.jsonl) | Qwen/Qwen2.5-1.5B-Instruct | 1,830 ms | 2,028 ms | 0 / 20 | 13,231 MiB |

Run on a free Colab T4 (fp16, `max_model_len=1024`, `max_num_seqs=20`). All 20 requests overlapped in time and finished in 2.45 s wall-clock (about 150 reply tokens/s). The largest prompt including stock rows was 368 tokens, well inside the 1,024 budget. Peak GPU memory is high mainly because vLLM pre-allocates 90% of the card for the KV cache.

`tests/check_results.py` passed on this file: 20 lines, every `question_id` found in the replay CSV, and the right catalog rows for each `product_sku` and `variant_size`. The two stock questions were answered correctly from those rows (Q-00939: 78 in stock in L; Q-01647: no L left, restock date given).

Full per-question results: [`results/load_test_20261123-1647.jsonl`](results/load_test_20261123-1647.jsonl). Summary: [`results/load_test_20261123-1647_summary.json`](results/load_test_20261123-1647_summary.json).
