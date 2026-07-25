#!/usr/bin/env python
"""
Compare Baselines Script

Runs the noop_agent and prints a comparison table showing noop score vs LLM score.
Output format: task name, noop score, llm score, delta.

Usage:
    python scripts/compare_baselines.py '{"easy": 0.82, "medium": 0.61, "hard": 0.35}'

This script does NOT run inference.py directly.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import re
from typing import Dict


def run_noop_agent(seed: int = 42) -> Dict[str, float]:
    """
    Run the noop agent and parse scores from output.
    
    Args:
        seed: Random seed for reproducibility.
        
    Returns:
        Dictionary mapping task names to scores.
    """
    # Run noop_agent.py and capture output
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    noop_script = os.path.join(script_dir, "noop_agent.py")
    
    try:
        result = subprocess.run(
            [sys.executable, noop_script, "--seed", str(seed)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        output = result.stdout
    except subprocess.TimeoutExpired:
        print("Error: noop_agent.py timed out", file=sys.stderr)
        return {}
    except Exception as e:
        print(f"Error running noop_agent.py: {e}", file=sys.stderr)
        return {}
    
    # Parse scores from the summary section
    scores: Dict[str, float] = {}
    in_summary = False
    for line in output.splitlines():
        if "--- NOOP Baseline Summary ---" in line:
            in_summary = True
            continue
        if in_summary:
            # Parse lines like "easy: 0.1234"
            match = re.match(r"(\w+):\s*([\d.]+)", line.strip())
            if match:
                task = match.group(1)
                score = float(match.group(2))
                scores[task] = score
    
    return scores


def print_comparison_table(noop_scores: Dict[str, float], llm_scores: Dict[str, float]) -> None:
    """
    Print a formatted comparison table with improvement multipliers.
    
    Args:
        noop_scores: Dictionary of task -> noop score.
        llm_scores: Dictionary of task -> LLM score.
    """
    # Get all task names
    all_tasks = sorted(set(noop_scores.keys()) | set(llm_scores.keys()))
    
    if not all_tasks:
        print("\nNo tasks to compare.")
        return
    
    # Print header
    print("\n" + "=" * 72)
    print("TITAN PERFORMANCE COMPARISON")
    print("=" * 72)
    print(f"{'Task':<12} | {'Baseline':>10} | {'LLM':>10} | {'Delta':>10} | {'Improvement':>12}")
    print("-" * 12 + "-+-" + "-" * 10 + "-+-" + "-" * 10 + "-+-" + "-" * 10 + "-+-" + "-" * 12)
    
    # Print rows
    total_improvement = 0.0
    valid_comparisons = 0
    
    for task in all_tasks:
        noop = noop_scores.get(task, float('nan'))
        llm = llm_scores.get(task, float('nan'))
        
        # Check for NaN
        noop_valid = noop == noop
        llm_valid = llm == llm
        
        if noop_valid and llm_valid:
            delta = llm - noop
            # Calculate improvement multiplier (avoid division by zero)
            if noop > 0.01:
                improvement = llm / noop
                improvement_str = f"{improvement:.1f}x"
                if improvement > 1:
                    improvement_str = f"+{improvement:.1f}x"
                total_improvement += improvement
                valid_comparisons += 1
            else:
                improvement_str = "N/A"
            delta_str = f"{delta:+.2f}"
        else:
            delta_str = "N/A"
            improvement_str = "N/A"
        
        noop_str = f"{noop:.2f}" if noop_valid else "N/A"
        llm_str = f"{llm:.2f}" if llm_valid else "N/A"
        
        # Add interpretation
        if llm_valid:
            if llm >= 0.8:
                interp = " [SUCCESS]"
            elif llm >= 0.5:
                interp = " [PARTIAL]"
            else:
                interp = " [FAILED]"
        else:
            interp = ""
        
        print(f"{task:<12} | {noop_str:>10} | {llm_str:>10} | {delta_str:>10} | {improvement_str:>12}{interp}")
    
    print("=" * 72)
    
    # Print summary
    if valid_comparisons > 0:
        avg_improvement = total_improvement / valid_comparisons
        print(f"Average improvement over baseline: {avg_improvement:.1f}x")
    
    # Print interpretation guide
    print("\nScore Interpretation:")
    print("  > 0.8 = Successful recovery")
    print("  > 0.5 = Partial recovery")
    print("  <= 0.5 = Failed")
    print()


def main() -> int:
    """Main entry point for the compare_baselines script."""
    parser = argparse.ArgumentParser(
        description="Compare NOOP baseline scores against LLM scores.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/compare_baselines.py '{"easy": 0.82, "medium": 0.61, "hard": 0.35}'
        """,
    )
    parser.add_argument(
        "llm_scores_json",
        type=str,
        help="JSON string containing task:score pairs from LLM inference",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for noop agent (default: 42)",
    )
    parser.add_argument(
        "--skip-noop-run",
        action="store_true",
        help="Skip running noop_agent and use provided noop scores instead",
    )
    parser.add_argument(
        "--noop-scores",
        type=str,
        default=None,
        help="JSON string of precomputed noop scores (use with --skip-noop-run)",
    )
    
    args = parser.parse_args()
    
    # Parse LLM scores
    try:
        llm_scores = json.loads(args.llm_scores_json)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON input for LLM scores: {e}", file=sys.stderr)
        return 1
    
    if not isinstance(llm_scores, dict):
        print("Error: LLM scores must be a JSON object/dict", file=sys.stderr)
        return 1
    
    # Get noop scores
    if args.skip_noop_run and args.noop_scores:
        try:
            noop_scores = json.loads(args.noop_scores)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON input for noop scores: {e}", file=sys.stderr)
            return 1
    else:
        print("Running noop_agent.py...")
        noop_scores = run_noop_agent(seed=args.seed)
        if not noop_scores:
            print("Warning: Could not get noop scores, showing LLM scores only", file=sys.stderr)
    
    # Print comparison table
    print_comparison_table(noop_scores, llm_scores)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
