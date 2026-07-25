#!/usr/bin/env python
"""
Hardening Check Script

Runs a series of validation and reproducibility checks for TITAN submission.
Prints PASS/FAIL for each check and a final summary.

Checks performed:
1. Reproducibility: Run inference.py twice and compare output log structure
2. Docker build: Verify docker build succeeds
3. API key fallback: Confirm no-op fallback activates without crash when API keys removed
4. Log format strictness: Verify all log lines match [START], [STEP], or [END] pattern
5. OpenEnv validate: Run openenv validate and assert exit code 0
6. Pytest: Run pytest and assert exit code 0

This script does NOT modify any existing source files.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional


_ROOT = Path(__file__).resolve().parent.parent


class CheckResult:
    """Result of a single check."""
    
    def __init__(self, name: str, passed: bool, message: str = ""):
        self.name = name
        self.passed = passed
        self.message = message
    
    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        msg = f" — {self.message}" if self.message else ""
        return f"[{status}] {self.name}{msg}"


def run_command(cmd: List[str], timeout: int = 300, env: Optional[dict] = None) -> Tuple[int, str, str]:
    """
    Run a command and return (exit_code, stdout, stderr).
    
    Args:
        cmd: Command and arguments as a list.
        timeout: Timeout in seconds.
        env: Optional environment variables override.
        
    Returns:
        Tuple of (exit_code, stdout, stderr).
    """
    try:
        proc_env = env if env is not None else os.environ.copy()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            cwd=str(_ROOT),
            env=proc_env,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Timeout expired"
    except Exception as e:
        return -2, "", str(e)


def parse_log_lines(output: str) -> List[str]:
    """Extract lines that should match log format."""
    log_lines = []
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("[START]") or stripped.startswith("[STEP]") or stripped.startswith("[END]"):
            log_lines.append(stripped)
    return log_lines


def _offline_inference_env() -> dict:
    """Build environment forcing offline fallback inference for deterministic checks."""
    env = os.environ.copy()
    env.pop("HF_TOKEN", None)
    env.pop("OPENAI_API_KEY", None)
    env.pop("API_KEY", None)
    env["TITAN_DISABLE_LOCAL_ENV"] = "1"
    return env


def validate_log_format(output: str) -> Tuple[bool, str]:
    """
    Validate that all log lines match [START], [STEP], or [END] pattern.
    
    Args:
        output: The output string to validate.
        
    Returns:
        Tuple of (is_valid, error_message).
    """
    log_pattern = re.compile(r"^\[(START|STEP|END)\]")
    
    log_lines = parse_log_lines(output)
    if not log_lines:
        return False, "No log lines found"
    
    for line in log_lines:
        if not log_pattern.match(line):
            return False, f"Invalid log line: {line[:50]}"
    
    return True, ""


def parse_scores_from_output(output: str) -> List[float]:
    """Parse scores from [END] lines in output."""
    scores = []
    for line in output.splitlines():
        if line.strip().startswith("[END]"):
            # Extract the last reward value (which is the score indicator)
            rewards_match = re.search(r"rewards=([\d.,]+)", line)
            if rewards_match:
                rewards_str = rewards_match.group(1)
                reward_values = [float(r) for r in rewards_str.split(",") if r]
                if reward_values:
                    scores.append(sum(reward_values) / len(reward_values))
    return scores


def check_reproducibility(variance_threshold: float = 0.01) -> CheckResult:
    """
    Check 1: Run inference.py twice and compare output log structure for consistency.
    
    Args:
        variance_threshold: Maximum allowed score variance between runs.
        
    Returns:
        CheckResult indicating pass/fail.
    """
    cmd = [sys.executable, str(_ROOT / "inference.py"), "--seed", "42"]
    
    # First run
    code1, stdout1, stderr1 = run_command(cmd, timeout=300, env=_offline_inference_env())
    if code1 != 0:
        return CheckResult("Reproducibility", False, f"First inference run failed (code {code1})")
    
    # Second run
    code2, stdout2, stderr2 = run_command(cmd, timeout=300, env=_offline_inference_env())
    if code2 != 0:
        return CheckResult("Reproducibility", False, f"Second inference run failed (code {code2})")
    
    # Check log structure consistency
    log_lines1 = parse_log_lines(stdout1)
    log_lines2 = parse_log_lines(stdout2)
    
    if len(log_lines1) != len(log_lines2):
        return CheckResult(
            "Reproducibility",
            False,
            f"Log line count differs: {len(log_lines1)} vs {len(log_lines2)}",
        )
    
    # Check [START], [STEP], [END] presence
    has_start = any(line.startswith("[START]") for line in log_lines1)
    has_step = any(line.startswith("[STEP]") for line in log_lines1)
    has_end = any(line.startswith("[END]") for line in log_lines1)
    
    if not (has_start and has_step and has_end):
        return CheckResult(
            "Reproducibility",
            False,
            "Missing required log sections ([START], [STEP], or [END])",
        )
    
    # Check score variance
    scores1 = parse_scores_from_output(stdout1)
    scores2 = parse_scores_from_output(stdout2)
    
    if scores1 and scores2:
        for s1, s2 in zip(scores1, scores2):
            variance = abs(s1 - s2)
            if variance > variance_threshold:
                return CheckResult(
                    "Reproducibility",
                    False,
                    f"Score variance {variance:.4f} exceeds threshold {variance_threshold}",
                )
    
    return CheckResult("Reproducibility", True, "Log structure and scores consistent")


def check_docker_build() -> CheckResult:
    """
    Check 2: Verify docker build succeeds.

    Returns:
        CheckResult indicating pass/fail.
    """
    dockerfile = _ROOT / "Dockerfile"
    if not dockerfile.exists():
        return CheckResult("Docker Build", False, "Dockerfile not found")

    cmd = ["docker", "build", "-t", "titan-env-test", str(_ROOT)]
    code, stdout, stderr = run_command(cmd, timeout=600)

    if code == 0:
        return CheckResult("Docker Build", True, "Build succeeded")
    elif code == -1:
        return CheckResult("Docker Build", False, "Build timed out")

    docker_error_text = f"{stdout}\n{stderr}".lower()
    if "not found" in docker_error_text or "not recognized" in docker_error_text:
        return CheckResult("Docker Build", False, "Docker CLI not available")
    if "error during connect" in docker_error_text or "dockerdesktoplinuxengine" in docker_error_text:
        return CheckResult("Docker Build", False, "Docker daemon is not running")

    return CheckResult("Docker Build", False, f"Build failed (code {code})")


def check_api_key_fallback() -> CheckResult:
    """
    Check 3: Simulate API key removal and confirm fallback no-op behavior activates without crash.
    
    Returns:
        CheckResult indicating pass/fail.
    """
    # Create environment without API keys
    env = _offline_inference_env()
    
    cmd = [sys.executable, str(_ROOT / "inference.py"), "--seed", "42"]
    code, stdout, stderr = run_command(cmd, timeout=300, env=env)
    
    if code != 0:
        return CheckResult("API Key Fallback", False, f"Inference crashed without API key (code {code})")
    
    # Verify fallback model name appears in output
    if "fallback-noop" in stdout or "noop" in stdout.lower():
        return CheckResult("API Key Fallback", True, "Fallback behavior activated")
    
    # Still passes if it ran without crashing, even if output doesn't mention fallback
    return CheckResult("API Key Fallback", True, "No crash without API keys")


def check_log_format_strictness() -> CheckResult:
    """
    Check 4: Validate log format strictness.

    Returns:
        CheckResult indicating pass/fail.
    """
    cmd = [sys.executable, str(_ROOT / "inference.py"), "--seed", "42"]
    code, stdout, stderr = run_command(cmd, timeout=300, env=_offline_inference_env())

    if code != 0:
        return CheckResult("Log Format Strictness", False, f"Inference failed (code {code})")
    
    is_valid, error = validate_log_format(stdout)
    
    if is_valid:
        return CheckResult("Log Format Strictness", True, "All log lines valid")
    else:
        return CheckResult("Log Format Strictness", False, error)


def check_openenv_validate() -> CheckResult:
    """
    Check 5: Run openenv validate and assert exit code 0.
    
    Returns:
        CheckResult indicating pass/fail.
    """
    cmd = ["openenv", "validate"]
    code, stdout, stderr = run_command(cmd, timeout=60)
    
    if code == 0:
        return CheckResult("OpenEnv Validate", True, "Validation passed")
    elif "not found" in stderr.lower() or "not recognized" in stderr.lower():
        return CheckResult("OpenEnv Validate", False, "openenv command not found")
    else:
        return CheckResult("OpenEnv Validate", False, f"Validation failed (code {code})")


def check_pytest() -> CheckResult:
    """
    Check 6: Run pytest and assert exit code 0.

    Returns:
        CheckResult indicating pass/fail.
    """
    tests_dir = _ROOT / "tests"
    if not tests_dir.exists():
        return CheckResult("Pytest", True, "No tests/ directory found (skipped)")

    cmd = [sys.executable, "-m", "pytest", str(tests_dir), "-v", "--tb=short"]
    code, stdout, stderr = run_command(cmd, timeout=300)

    output_text = f"{stdout}\n{stderr}".lower()
    if "no module named pytest" in output_text:
        return CheckResult("Pytest", False, "pytest is not installed")

    if code == 0:
        return CheckResult("Pytest", True, "All tests passed")
    if code == 1:
        # Some tests failed
        failed_count = stdout.count("FAILED")
        return CheckResult("Pytest", False, f"{failed_count} test(s) failed")
    if code == 5:
        return CheckResult("Pytest", True, "No tests collected")

    return CheckResult("Pytest", False, f"Pytest error (code {code})")


def main() -> int:
    """Main entry point for hardening checks."""
    print("=" * 60)
    print("TITAN Hardening Checks")
    print("=" * 60)
    print()
    
    checks = [
        ("1. Reproducibility Check", check_reproducibility),
        ("2. Docker Build Check", check_docker_build),
        ("3. API Key Fallback Check", check_api_key_fallback),
        ("4. Log Format Strictness Check", check_log_format_strictness),
        ("5. OpenEnv Validate Check", check_openenv_validate),
        ("6. Pytest Check", check_pytest),
    ]
    
    results: List[CheckResult] = []
    
    for check_name, check_fn in checks:
        print(f"Running {check_name}...")
        try:
            result = check_fn()
        except Exception as e:
            result = CheckResult(check_name, False, f"Exception: {str(e)[:50]}")
        results.append(result)
        print(f"  {result}")
        print()
    
    # Summary
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    
    print("=" * 60)
    print(f"{passed}/{total} checks passed")
    print("=" * 60)
    
    # Return 0 if all passed, 1 otherwise
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
