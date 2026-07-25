#!/usr/bin/env python
"""
Print Scores Script

A standalone script that imports score_interpreter and pretty-prints results.
Takes a JSON dict of task:score pairs as CLI argument.

Usage:
    python scripts/print_scores.py '{"easy": 0.82, "medium": 0.61, "hard": 0.15}'
    
This script does NOT import inference.py.
"""

from __future__ import annotations

import argparse
import json
import sys

from TITAN_env.evaluation.score_interpreter import interpret_score, format_result


def main() -> int:
    """Main entry point for the print_scores script."""
    parser = argparse.ArgumentParser(
        description="Pretty-print TITAN score interpretations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/print_scores.py '{"easy": 0.82, "medium": 0.61, "hard": 0.15}'
    python scripts/print_scores.py '{"easy_single_fault_recovery": 0.95}'
        """,
    )
    parser.add_argument(
        "scores_json",
        type=str,
        help="JSON string containing task:score pairs, e.g., '{\"easy\": 0.82}'",
    )
    parser.add_argument(
        "--format",
        choices=["table", "json", "text"],
        default="table",
        help="Output format (default: table)",
    )
    
    args = parser.parse_args()
    
    try:
        scores = json.loads(args.scores_json)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON input: {e}", file=sys.stderr)
        return 1
    
    if not isinstance(scores, dict):
        print("Error: Input must be a JSON object/dict", file=sys.stderr)
        return 1
    
    results = [interpret_score(task, score) for task, score in scores.items()]
    
    if args.format == "json":
        print(json.dumps(results, indent=2))
    elif args.format == "text":
        for result in results:
            print(format_result(result))
            print()
    else:  # table format
        print_table(results)
    
    return 0


def print_table(results: list) -> None:
    """Print results as a formatted table."""
    if not results:
        print("No scores to display.")
        return
    
    # Column widths
    task_width = max(len(r["task"]) for r in results)
    task_width = max(task_width, len("Task"))
    
    # Header
    print(f"{'Task':<{task_width}}  {'Score':>6}  {'Interpretation'}")
    print("-" * (task_width + 60))
    
    # Rows
    for result in results:
        task = result["task"]
        score = result["score"]
        interpretation = result["interpretation"]
        print(f"{task:<{task_width}}  {score:>6.2f}  {interpretation}")


if __name__ == "__main__":
    raise SystemExit(main())
