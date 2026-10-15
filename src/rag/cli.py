import argparse
import json
import logging
import sys
from pathlib import Path

from pydantic import ValidationError

from rag.exceptions import RagError
from rag.factory import build_pipeline
from rag.models import Answer, IngestReport
from rag.pipeline import RagPipeline


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = _parser()
    args = parser.parse_args(argv)

    try:
        pipeline = build_pipeline()
    except ValidationError as exc:
        if any(error.get("loc") == ("groq_api_key",) for error in exc.errors()):
            print(
                "GROQ_API_KEY is missing or invalid. Copy .env.example to .env and set the key.",
                file=sys.stderr,
            )
        else:
            print(exc, file=sys.stderr)
        return 2
    except RagError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        if args.command == "ingest":
            return _ingest(pipeline, args.path)
        if args.command == "ask":
            return _ask(pipeline, args)
        if args.command == "stats":
            return _stats(pipeline)
    except RagError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag",
        description="Ask questions over PDFs. Embeddings and answers both go through Groq.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Index a PDF or every PDF under a folder")
    ingest.add_argument("path", nargs="?", default="data/pdfs", type=Path)

    ask = sub.add_parser("ask", help="Answer one question from the index")
    ask.add_argument("question")
    ask.add_argument("-k", type=int, default=None, help="How many chunks to retrieve")
    ask.add_argument("--source", default=None, help="Restrict retrieval to one indexed path")
    ask.add_argument("--min-relevance", type=float, default=None)
    ask.add_argument("--json", action="store_true", dest="as_json")

    sub.add_parser("stats", help="Show indexed files and chunk counts")
    return parser


def _ingest(pipeline: RagPipeline, path: Path) -> int:
    reports = pipeline.ingest_path(path)
    for report in reports:
        _print_report(report)
    return 0


def _ask(pipeline: RagPipeline, args: argparse.Namespace) -> int:
    answer = pipeline.ask(
        args.question,
        k=args.k,
        source=args.source,
        min_relevance=args.min_relevance,
    )
    if args.as_json:
        print(json.dumps(_answer_payload(answer), ensure_ascii=False, indent=2))
        return 0
    _print_answer(answer)
    return 0


def _stats(pipeline: RagPipeline) -> int:
    sources = pipeline.stats()
    if not sources:
        print("Index is empty.")
        return 0
    total = sum(sources.values())
    print(f"{total} chunks across {len(sources)} file(s)")
    for source, count in sources.items():
        print(f"{count:6d}  {source}")
    return 0


def _print_report(report: IngestReport) -> None:
    print(
        f"indexed {report.source}: "
        f"{report.chunks} chunks, {report.pages} pages, {report.empty_pages} empty"
    )


def _print_answer(answer: Answer) -> None:
    print(answer.text)
    if not answer.citations:
        return
    heading = (
        "Sources"
        if answer.answered_from_context
        else "Nearest excerpts below the relevance cutoff"
    )
    print(f"\n{heading}")
    for number, citation in enumerate(answer.citations, start=1):
        snippet = " ".join(citation.text.split())
        if len(snippet) > 180:
            snippet = snippet[:177] + "..."
        print(
            f"[{number}] {citation.source} p.{citation.page}  similarity={citation.score:.2f}"
        )
        print(f"    {snippet}")


def _answer_payload(answer: Answer) -> dict:
    return {
        "question": answer.question,
        "answer": answer.text,
        "answered_from_context": answer.answered_from_context,
        "citations": [
            {
                "source": citation.source,
                "page": citation.page,
                "score": round(citation.score, 4),
                "text": citation.text,
            }
            for citation in answer.citations
        ],
    }
