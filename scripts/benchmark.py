"""
Performance benchmark for the Legal AI RAG system.

Tests:
  1. Retrieval accuracy  – does the search endpoint return relevant law sections?
  2. Retrieval latency   – embedding + vector search time via /search
  3. End-to-end latency  – full chat (streaming) via /chat/stream
  4. German detection    – does it skip translation for German queries?

Usage:
    python -m scripts.benchmark [--base-url http://localhost:8000]
"""
from __future__ import annotations

import argparse
import json
import time
import httpx
import asyncio
import sys

BASE = "http://localhost:8000"

# ── Test queries with expected top-1 law code ──────────────────────────
RETRIEVAL_TESTS = [
    # German queries (should skip translation)
    {"query": "Was ist ein Kaufvertrag nach dem BGB?", "expect_law": "BGB", "lang": "de"},
    {"query": "Welche Rechte hat ein Mieter bei Mängeln?", "expect_law": "BGB", "lang": "de"},
    {"query": "Wann liegt Mord vor und wann Totschlag?", "expect_law": "StGB", "lang": "de"},
    {"query": "Was regelt § 823 BGB?", "expect_law": "BGB", "lang": "de"},
    {"query": "Wie funktioniert die Handelsregistereintragung?", "expect_law": "HGB", "lang": "de"},
    {"query": "Was sind die Grundrechte im Grundgesetz?", "expect_law": "GG", "lang": "de"},
    {"query": "Wie läuft eine Zivilklage ab?", "expect_law": "ZPO", "lang": "de"},
    {"query": "Welche Rechte hat der Beschuldigte in der StPO?", "expect_law": "StPO", "lang": "de"},
    {"query": "Was ist ein Verwaltungsakt?", "expect_law": "VwGO", "lang": "de"},
    {"query": "Schadensersatz bei Vertragsverletzung", "expect_law": "BGB", "lang": "de"},
    # English queries (need translation)
    {"query": "What are the grounds for divorce in German law?", "expect_law": "BGB", "lang": "en"},
    {"query": "What is theft under German criminal law?", "expect_law": "StGB", "lang": "en"},
    {"query": "Freedom of speech in German constitution", "expect_law": "GG", "lang": "en"},
    {"query": "How does a commercial partnership work in Germany?", "expect_law": "HGB", "lang": "en"},
    {"query": "Rules of evidence in German civil procedure", "expect_law": "ZPO", "lang": "en"},
]


async def run_search(client: httpx.AsyncClient, query: str, top_k: int = 5) -> dict:
    """Call /api/v1/search and return results + timing."""
    t0 = time.perf_counter()
    resp = await client.post(
        f"{BASE}/api/v1/search",
        json={"query": query, "top_k": top_k, "similarity_threshold": 0.0},
        timeout=30,
    )
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()
    return {"results": data["results"], "elapsed": elapsed}


async def run_chat_stream(client: httpx.AsyncClient, query: str) -> dict:
    """Call /api/v1/chat/stream and measure time-to-first-token + total time."""
    t0 = time.perf_counter()
    ttft = None
    tokens = []
    sources = []

    async with client.stream(
        "POST",
        f"{BASE}/api/v1/chat/stream",
        json={"query": query, "top_k": 5, "similarity_threshold": 0.0},
        timeout=60,
    ) as resp:
        resp.raise_for_status()
        buf = ""
        async for chunk in resp.aiter_text():
            buf += chunk
            while "\n\n" in buf:
                raw_event, buf = buf.split("\n\n", 1)
                lines = raw_event.strip().split("\n")
                event_type = ""
                data = ""
                for line in lines:
                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                    elif line.startswith("data:"):
                        data = line[5:].strip()
                if event_type == "token":
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    tokens.append(data)
                elif event_type == "sources":
                    try:
                        sources = json.loads(data)
                    except json.JSONDecodeError:
                        pass

    total = time.perf_counter() - t0
    return {
        "ttft": ttft or total,
        "total": total,
        "token_count": len(tokens),
        "answer_chars": sum(len(t) for t in tokens),
        "sources": sources,
    }


def check_retrieval_accuracy(results: list[dict], expected_law: str, top_n: int) -> bool:
    """Check if expected law appears in top_n results."""
    for r in results[:top_n]:
        meta = r.get("metadata", {})
        abbrev = meta.get("law_abbreviation", "")
        if abbrev.upper() == expected_law.upper():
            return True
    return False


async def main(base_url: str):
    global BASE
    BASE = base_url

    print("=" * 70)
    print("  LEGAL AI RAG — PERFORMANCE BENCHMARK")
    print(f"  Embedding model: gemini-embedding-001")
    print(f"  Backend: {BASE}")
    print("=" * 70)

    # Check backend health
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(f"{BASE}/health/ready", timeout=5)
            info = r.json()
            print(f"\n  Status: {info.get('status')} | DB: {info.get('database')} | LLM: {info.get('llm_provider')}")
        except Exception as e:
            print(f"\n  ERROR: Backend not reachable at {BASE}: {e}")
            sys.exit(1)

    async with httpx.AsyncClient() as client:
        # ── 1. RETRIEVAL ACCURACY ─────────────────────────────
        print("\n" + "─" * 70)
        print("  1. RETRIEVAL ACCURACY")
        print("─" * 70)

        top1_hits = 0
        top3_hits = 0
        top5_hits = 0
        de_times = []
        en_times = []
        all_times = []

        for test in RETRIEVAL_TESTS:
            result = await run_search(client, test["query"], top_k=5)
            results = result["results"]
            elapsed = result["elapsed"]
            all_times.append(elapsed)

            if test["lang"] == "de":
                de_times.append(elapsed)
            else:
                en_times.append(elapsed)

            hit1 = check_retrieval_accuracy(results, test["expect_law"], 1)
            hit3 = check_retrieval_accuracy(results, test["expect_law"], 3)
            hit5 = check_retrieval_accuracy(results, test["expect_law"], 5)

            if hit1:
                top1_hits += 1
            if hit3:
                top3_hits += 1
            if hit5:
                top5_hits += 1

            status = "HIT" if hit1 else ("top3" if hit3 else ("top5" if hit5 else "MISS"))
            top_sim = f"{results[0]['similarity']:.3f}" if results else "N/A"
            top_law = results[0]["metadata"].get("law_abbreviation", "?") if results else "?"

            print(f"  [{status:>4}] {test['query'][:55]:<55} → {top_law} ({top_sim}) {elapsed:.2f}s")

        n = len(RETRIEVAL_TESTS)
        print(f"\n  Top-1 accuracy: {top1_hits}/{n} ({top1_hits/n*100:.0f}%)")
        print(f"  Top-3 accuracy: {top3_hits}/{n} ({top3_hits/n*100:.0f}%)")
        print(f"  Top-5 accuracy: {top5_hits}/{n} ({top5_hits/n*100:.0f}%)")

        # ── 2. RETRIEVAL LATENCY ──────────────────────────────
        print("\n" + "─" * 70)
        print("  2. RETRIEVAL LATENCY (embedding + vector search)")
        print("─" * 70)

        avg_all = sum(all_times) / len(all_times)
        avg_de = sum(de_times) / len(de_times) if de_times else 0
        avg_en = sum(en_times) / len(en_times) if en_times else 0

        print(f"  German queries (no translation):  avg {avg_de:.2f}s  (n={len(de_times)})")
        print(f"  English queries (w/ translation):  avg {avg_en:.2f}s  (n={len(en_times)})")
        print(f"  Overall:                           avg {avg_all:.2f}s  (n={len(all_times)})")
        print(f"  Min: {min(all_times):.2f}s  Max: {max(all_times):.2f}s")

        # ── 3. END-TO-END STREAMING LATENCY ───────────────────
        print("\n" + "─" * 70)
        print("  3. END-TO-END STREAMING LATENCY (chat/stream)")
        print("─" * 70)

        e2e_tests = [
            {"query": "Was ist Notwehr nach deutschem Recht?", "lang": "de"},
            {"query": "Erkläre § 433 BGB", "lang": "de"},
            {"query": "What are tenant rights under German law?", "lang": "en"},
        ]

        for test in e2e_tests:
            try:
                result = await run_chat_stream(client, test["query"])
                print(f"  [{test['lang'].upper()}] {test['query'][:50]:<50}")
                print(f"       TTFT: {result['ttft']:.2f}s | Total: {result['total']:.2f}s | Tokens: {result['token_count']} | Chars: {result['answer_chars']}")
                src_laws = [s.get("metadata", {}).get("law_abbreviation", "?") for s in result["sources"][:3]]
                print(f"       Sources: {', '.join(src_laws)}")
            except Exception as e:
                print(f"  [{test['lang'].upper()}] {test['query'][:50]:<50}")
                print(f"       ERROR: {e}")

        # ── 4. SIMILARITY SCORE DISTRIBUTION ──────────────────
        print("\n" + "─" * 70)
        print("  4. SIMILARITY SCORE DISTRIBUTION")
        print("─" * 70)

        # Re-use results from retrieval tests
        all_sims = []
        for test in RETRIEVAL_TESTS[:5]:  # sample 5
            result = await run_search(client, test["query"], top_k=10)
            sims = [r["similarity"] for r in result["results"]]
            all_sims.extend(sims)
            top3_sims = sims[:3]
            print(f"  {test['query'][:50]:<50}")
            print(f"    Top-3 sims: {', '.join(f'{s:.3f}' for s in top3_sims)}  |  Results: {len(sims)}")

        if all_sims:
            print(f"\n  Overall similarity stats (n={len(all_sims)}):")
            print(f"    Mean: {sum(all_sims)/len(all_sims):.3f}  |  Max: {max(all_sims):.3f}  |  Min: {min(all_sims):.3f}")

        # ── SUMMARY ───────────────────────────────────────────
        print("\n" + "=" * 70)
        print("  SUMMARY")
        print("=" * 70)
        print(f"  Embedding model:      gemini-embedding-001 (1536d)")
        print(f"  Retrieval accuracy:   Top-1={top1_hits/n*100:.0f}%  Top-3={top3_hits/n*100:.0f}%  Top-5={top5_hits/n*100:.0f}%")
        print(f"  Search latency:       DE={avg_de:.2f}s  EN={avg_en:.2f}s  avg={avg_all:.2f}s")
        if all_sims:
            print(f"  Similarity scores:    mean={sum(all_sims)/len(all_sims):.3f}  max={max(all_sims):.3f}")
        print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()
    asyncio.run(main(args.base_url))
