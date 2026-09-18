"""Measure retrieval quality, and sweep chunk settings to improve it.

This is the most important script in the project, and the one most RAG tutorials
leave out. Without it you are tuning blind: you change the chunk size, ask a
question, read an answer that "seems fine", and have no idea whether you made
the system better or worse.

Note what it deliberately does NOT measure: the quality of Claude's writing. It
evaluates RETRIEVAL only - did the right chunk come back at all? That separation
matters, because if the correct clause was never retrieved, no amount of prompt
tuning will fix the answer. Retrieval is the ceiling on everything downstream,
and it can be measured for free, with no API calls.

Usage:
    python scripts/evaluate.py                          # score the live index
    python scripts/evaluate.py --questions eval/q.json
    python scripts/evaluate.py --sweep                  # try several chunk sizes

The questions file is a JSON list:
    [
      {
        "question": "How much is the excess on a claim?",
        "expect_any": ["excess of 5,000", "compulsory excess"],
        "expect_pages": [1]
      }
    ]

`expect_any` holds substrings; a question counts as a hit when a retrieved chunk
contains at least one of them (case-insensitive). Writing these by hand from
your own policies is the work - and it is worth it, because it converts "the bot
feels wrong" into a number you can move.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.services.embedder import Embedder  # noqa: E402
from app.services.pipeline import IngestionPipeline  # noqa: E402
from app.services.vector_db import VectorStore  # noqa: E402
from scripts.ingest import document_id_for  # noqa: E402


@dataclass
class Outcome:
    question: str
    hit: bool
    best_rank: int | None  # 1-based rank of the first correct chunk
    top_score: float
    hit_score: float | None
    pages_found: list[int]


def load_questions(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path} must contain a non-empty JSON list")
    return data


def evaluate(
    store: VectorStore,
    embedder: Embedder,
    questions: list[dict],
    top_k: int,
) -> list[Outcome]:
    outcomes: list[Outcome] = []
    for entry in questions:
        question = entry["question"]
        expected = [text.lower() for text in entry.get("expect_any", [])]
        hits = store.search(embedder.embed_query(question), top_k=top_k)

        best_rank: int | None = None
        hit_score: float | None = None
        for rank, hit in enumerate(hits, start=1):
            haystack = hit.text.lower()
            if any(needle in haystack for needle in expected):
                best_rank, hit_score = rank, hit.score
                break

        outcomes.append(
            Outcome(
                question=question,
                hit=best_rank is not None,
                best_rank=best_rank,
                top_score=hits[0].score if hits else 0.0,
                hit_score=hit_score,
                pages_found=sorted({h.page_start for h in hits}),
            )
        )
    return outcomes


def summarise(outcomes: list[Outcome], top_k: int, label: str = "") -> dict:
    total = len(outcomes)
    hits = [o for o in outcomes if o.hit]
    # Mean Reciprocal Rank: 1.0 if the right chunk is always first, 0.5 if
    # always second, and so on. More informative than hit-rate alone because it
    # rewards putting the correct chunk at the top, where the model reads it.
    mrr = sum(1.0 / o.best_rank for o in hits) / total if total else 0.0
    misses = [o for o in outcomes if not o.hit]

    print(f"\n{'=' * 72}")
    print(f"{label or 'Retrieval quality'}  (top_k={top_k}, {total} questions)")
    print("=" * 72)
    print(f"  hit@{top_k}: {len(hits)}/{total} = {len(hits) / total:.0%}")
    print(f"  MRR:     {mrr:.3f}")

    if hits:
        scores = sorted(o.hit_score for o in hits)
        print(f"  correct-chunk similarity: min={scores[0]:.3f} median={scores[len(scores) // 2]:.3f} max={scores[-1]:.3f}")
        print(f"  -> set min_similarity comfortably BELOW {scores[0]:.3f} or you will refuse valid questions")

    if misses:
        print(f"\n  {len(misses)} miss(es):")
        for outcome in misses:
            print(f"    - {outcome.question}")
            print(f"      best retrieved score {outcome.top_score:.3f}, pages seen {outcome.pages_found}")

    return {"hit_rate": len(hits) / total if total else 0.0, "mrr": mrr}


def sweep(source: Path, questions: list[dict], top_k: int) -> None:
    """Re-index the corpus at several chunk settings and compare.

    Each configuration gets its own throwaway Chroma directory, because vectors
    built at one chunk size cannot be mixed with another's.
    """
    settings = get_settings()
    embedder = Embedder(model_name=settings.embedding_model)
    pdfs = sorted(source.rglob("*.pdf")) if source.is_dir() else [source]
    if not pdfs:
        print(f"error: no PDFs under {source}", file=sys.stderr)
        return

    configurations = [(400, 80), (600, 100), (800, 150), (1000, 150), (1400, 200)]
    results = []

    for chunk_size, overlap in configurations:
        workdir = Path(tempfile.mkdtemp(prefix="ragsweep-"))
        try:
            store = VectorStore(workdir, collection_name="sweep")
            pipeline = IngestionPipeline(store, embedder, chunk_size, overlap)
            for pdf in pdfs:
                try:
                    pipeline.ingest(pdf, document_id=document_id_for(pdf), filename=pdf.name)
                except Exception as exc:  # noqa: BLE001
                    print(f"  skipped {pdf.name}: {exc}", file=sys.stderr)
            outcomes = evaluate(store, embedder, questions, top_k)
            stats = summarise(outcomes, top_k, label=f"chunk_size={chunk_size} overlap={overlap}")
            results.append((chunk_size, overlap, store.count(), stats))
        finally:
            del store
            shutil.rmtree(workdir, ignore_errors=True)

    print(f"\n{'=' * 72}\nSWEEP SUMMARY\n{'=' * 72}")
    print(f"{'chunk':>7} {'overlap':>8} {'chunks':>7} {'hit rate':>9} {'MRR':>7}")
    for chunk_size, overlap, count, stats in results:
        print(f"{chunk_size:>7} {overlap:>8} {count:>7} {stats['hit_rate']:>8.0%} {stats['mrr']:>7.3f}")
    best = max(results, key=lambda row: (row[3]["mrr"], row[3]["hit_rate"]))
    print(f"\nBest by MRR: chunk_size={best[0]}, chunk_overlap={best[1]}")
    print("Set these in .env (CHUNK_SIZE / CHUNK_OVERLAP), then re-run scripts/ingest.py --reset")


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Measure retrieval quality.")
    parser.add_argument("--questions", type=Path, default=Path("eval/questions.json"))
    parser.add_argument("--top-k", type=int, default=settings.top_k)
    parser.add_argument("--sweep", action="store_true", help="Compare chunk configurations.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(settings.upload_dir),
        help="PDF folder to re-index during a sweep.",
    )
    args = parser.parse_args()

    if not args.questions.exists():
        print(f"error: {args.questions} not found. See the docstring for its format.", file=sys.stderr)
        return 1
    questions = load_questions(args.questions)

    if args.sweep:
        sweep(args.source, questions, args.top_k)
        return 0

    store = VectorStore(settings.chroma_dir, collection_name=settings.collection_name)
    if store.count() == 0:
        print("error: the index is empty - run scripts/ingest.py first.", file=sys.stderr)
        return 1

    outcomes = evaluate(store, Embedder(model_name=settings.embedding_model), questions, args.top_k)
    summarise(outcomes, args.top_k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
