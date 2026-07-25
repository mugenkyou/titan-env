"""
Score Interpreter for TITAN evaluation results.

This module converts raw float scores into human-readable outcome labels.
It is a standalone utility that does not modify inference.py or any grader.
"""

from __future__ import annotations

from typing import Dict, Any


def interpret_score(task_name: str, score: float) -> Dict[str, Any]:
    """
    Convert a raw float score into a human-readable outcome interpretation.
    
    Args:
        task_name: The name of the task (e.g., 'easy_single_fault_recovery').
        score: The raw score value in range [0.0, 1.0].
        
    Returns:
        A dictionary with exactly three keys:
        - 'task': The task name
        - 'score': The numeric score
        - 'interpretation': A human-readable string describing the outcome
    """
    # Determine interpretation based on score thresholds
    if score >= 0.75:
        interpretation = "Successful recovery with minor inefficiency"
    elif score >= 0.40:
        interpretation = "Partial stabilization, system stress remains"
    else:
        interpretation = "Recovery failed, system remained degraded"
    
    return {
        "task": task_name,
        "score": score,
        "interpretation": interpretation,
    }


def interpret_scores(scores: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
    """
    Interpret multiple scores at once.
    
    Args:
        scores: A dictionary mapping task names to scores.
        
    Returns:
        A dictionary mapping task names to interpretation results.
    """
    return {task: interpret_score(task, score) for task, score in scores.items()}


def get_interpretation_category(score: float) -> str:
    """
    Get the category name for a given score.
    
    Args:
        score: The raw score value in range [0.0, 1.0].
        
    Returns:
        Category string: 'high', 'medium', or 'low'.
    """
    if score >= 0.75:
        return "high"
    elif score >= 0.40:
        return "medium"
    else:
        return "low"


def format_result(result: Dict[str, Any]) -> str:
    """
    Format an interpretation result as a human-readable string.
    
    Args:
        result: The result dictionary from interpret_score().
        
    Returns:
        A formatted string representation.
    """
    return f"Task: {result['task']}\nScore: {result['score']:.2f}\nOutcome: {result['interpretation']}"
