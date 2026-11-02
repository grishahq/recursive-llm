"""Analyze local files with source-checked citations and bounded execution.

python examples/analyze_documents.py report.pdf --query "What are the AI risks?" \
    --codex --model gpt-6.1-sol --output analysis.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from rlm import DocumentLimits, RLM, load_document
from rlm.codex import CodexCompletion


async def analyze(args: argparse.Namespace) -> None:
    """Load local pages, analyze them, and optionally persist a complete result."""
    documents = [
        load_document(path, document_id=f"document-{index}", limits=DocumentLimits())
        for index, path in enumerate(args.documents, start=1)
    ]
    transport = None
    if args.codex:
        transport = CodexCompletion(timeout=args.call_timeout, reasoning_effort=args.reasoning)
        await transport.check_login()
    engine = RLM(
        model=args.model,
        max_depth=args.max_depth,
        max_iterations=args.max_calls,
        max_total_calls=args.max_calls,
        max_elapsed_seconds=args.timeout,
        max_output_chars=12_000,
        completion_handler=transport,
    )
    result = await engine.acomplete_documents(args.query, documents)
    if args.output:
        result.write_jsonl(args.output)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


def main() -> None:
    """Run the explicit API-provider or ChatGPT-authenticated CLI transport."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("documents", type=Path, nargs="+")
    parser.add_argument("--query", required=True)
    parser.add_argument("--model", default="gpt-6.1-sol")
    parser.add_argument("--codex", action="store_true", help="Use the existing Codex ChatGPT login")
    parser.add_argument("--reasoning", choices=["low", "medium", "high"], default="low")
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-calls", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--call-timeout", type=float, default=180)
    parser.add_argument("--output", type=Path, help="Append content-bearing run records to JSONL")
    args = parser.parse_args()
    asyncio.run(analyze(args))


if __name__ == "__main__":
    main()
