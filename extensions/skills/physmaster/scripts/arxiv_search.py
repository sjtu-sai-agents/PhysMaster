#!/usr/bin/env python3
"""Standalone arXiv paper search tool.

Usage:
    python arxiv_search.py --query "quantum error correction" --top_k 5
    python arxiv_search.py --query "ti:black hole au:Hawking" --top_k 3
    python arxiv_search.py --query "2301.08727" --top_k 1
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from LANDAU.library import LibraryRetriever


def main():
    parser = argparse.ArgumentParser(description="Search arXiv for physics papers")
    parser.add_argument("--query", "-q", type=str, required=True, help="Search query or arXiv ID")
    parser.add_argument("--top_k", "-k", type=int, default=5, help="Number of results")
    args = parser.parse_args()

    lib = LibraryRetriever()
    results = lib.search(query=args.query, top_k=args.top_k)

    if not results:
        print("No results found.")
        return

    print(f"Found {len(results)} results for: {args.query}\n")
    print(lib.format_for_llm(results))


if __name__ == "__main__":
    main()
