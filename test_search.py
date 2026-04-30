#!/usr/bin/env python3
"""Test script for AI Search service with typo handling."""

from pathlib import Path
from ExpertServiceSearch import build_search_engine, print_results


def main():
    json_path = Path(__file__).parent / "ExpertServices.json"
    engine = build_search_engine(json_path)

    # Test cases with typos and variations
    test_queries = [
        "GP Consultation",
        "gynaecology consultaton",  # Typo: consultaton -> consultation
        "plumbing repair",
        "plumbing reapir",  # Typo: reapir -> repair
        "hello",  # Greeting
        "search for services",  # Help intent
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: '{query}'")
        print("=" * 60)

        print_results(query, engine, top_k=5)


if __name__ == "__main__":
    main()
