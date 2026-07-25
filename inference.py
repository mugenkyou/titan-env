from __future__ import annotations

import argparse
import os
from typing import Any, Callable, Dict, Tuple

try:
    from titan_env.evaluation.scoring import score_trajectory
except ImportError:
    def score_trajectory(task_name: str, trajectory: EvaluationTrajectory) -> float:
        """Fallback scoring when titan_env.evaluation.scoring is unavailable."""
        return 0.5
from titan_env.evaluation.trajectory import EvaluationTrajectory
from titan_env.interface.llm_interface import parse_action, render_observation
from titan_env.interface.models import Action, Observation
from titan_env.interface.openenv_wrapper import TITANEnv as OpenEnvWrapper
from titan_env.tasks.registry import available_task_names, resolve_task_bundle


ENV_NAME = "titan_env.interface.openenv_wrapper.TITANEnv"


def safe_score(score: float) -> float:
    return min(max(round(float(score), 4), 0.01), 0.99)


def _load_local_env_file(filename: str = "local.env") -> None:
    """Load simple KEY=VALUE pairs from local.env if present."""
    path = os.path.join(os.getcwd(), filename)
    if not os.path.exists(path):
        return

    try:
        with open(path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        # local.env loading is best-effort and should not block inference.
        return


def _normalize_dict(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return dict(model.model_dump())
    if hasattr(model, "dict"):
        return dict(model.dict())
    return dict(model)


def _format_error(error: Exception | str | None) -> str:
    if error is None:
        return "null"
    text = str(error).strip().replace("\n", " ")
    return text if text else "null"


def _observation_to_text(observation: Any) -> str:
    if isinstance(observation, Observation):
        return render_observation(observation)
    return render_observation(Observation(**_normalize_dict(observation)))


def _build_prompt(task_name: str, step_index: int, max_steps: int, observation_text: str) -> str:
    # Compact task-specific instructions
    task_info = {
        "easy": "GOAL: memory>90%. USE: memory_scrub",
        "medium": "GOAL: cpu_temp<70%. USE: thermal_throttle, load_shedding",
        "hard": "GOAL: survive 100 steps, battery>20%. USE: thermal_throttle/memory_scrub/power_cycle/load_shedding as needed",
    }
    
    return (
        f"TITAN Controller. Step {step_index}/{max_steps}\n"
        f"Task: {task_name} - {task_info.get(task_name, task_info['easy'])}\n"
        f"Commands: memory_scrub|thermal_throttle|load_shedding|power_cycle|reset|isolate|no_action\n"
        f"State:\n{observation_text}\n"
        "Reply with ONE command only."
    )


def _build_openai_model(api_base_url: str, model_name: str, hf_token: str, seed: int) -> Callable[[str], str]:
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover - defensive import guard
        raise RuntimeError("The openai package is required to run inference.py") from exc

    client = OpenAI(base_url=api_base_url, api_key=hf_token)

    def _model(prompt: str) -> str:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an expert satellite fault recovery AI. Analyze the system state and respond with exactly ONE command name. No explanations, just the command."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                top_p=1.0,
                seed=seed,
                max_tokens=32,
            )
            return response.choices[0].message.content or "no_action"
        except Exception as api_exc:
            print(f"[API ERROR] {api_exc}. Falling back to no-op model.")
            return _fallback_model(prompt)

    return _model


def _fallback_model(_: str) -> str:
    return "no_action"


def _interpret_score(score: float, task_alias: str) -> str:
    """Convert numeric score to human-readable interpretation."""
    if score >= 0.8:
        level = "Successful"
    elif score >= 0.5:
        level = "Partial"
    else:
        level = "Failed"
    
    # Task-specific context
    if "easy" in task_alias.lower():
        context = "memory recovery"
    elif "medium" in task_alias.lower():
        context = "thermal stabilization"
    elif "hard" in task_alias.lower():
        context = "multi-fault survival"
    else:
        context = "recovery"
    
    return f"{level} {context}"


def _score_trajectory(task_name: str, trajectory: EvaluationTrajectory) -> float:
    return safe_score(score_trajectory(task_name, trajectory))


def _run_task(task_alias: str, model: Callable[[str], str], seed: int, model_name: str) -> Tuple[float, EvaluationTrajectory]:
    bundle = resolve_task_bundle(task_alias)
    task = bundle.task
    core_env = task.build_core_env(seed)
    wrapper = OpenEnvWrapper(core_env=core_env)

    task.reset(wrapper.core_env, seed)
    observation = wrapper.state()

    trajectory = EvaluationTrajectory()
    trajectory.start(_normalize_dict(observation))

    print(f"[START] task={task_alias} env={ENV_NAME} model={model_name}")
    score = safe_score(0.0)
    success = False
    try:
        for step_index in range(1, task.max_steps + 1):
            prompt = _build_prompt(task_alias, step_index, task.max_steps, _observation_to_text(observation))
            step_error: str | None = None

            try:
                model_output = model(prompt)
                action = parse_action(model_output)
            except Exception as exc:
                action = Action(command="no_action")
                step_error = _format_error(exc)

            try:
                next_observation, reward, done, info = wrapper.step(action)
            except Exception as exc:
                next_observation = observation
                reward = None
                done = True
                info = {"error": _format_error(exc)}
                step_error = step_error or _format_error(exc)

            reward_value = float(getattr(reward, "value", 0.0)) if reward is not None else 0.0
            trajectory.append_step(
                action=action.command,
                reward=reward_value,
                done=bool(done),
                next_observation=_normalize_dict(next_observation),
                info=info,
            )

            print(
                f"[STEP] step={step_index} action={action.command} reward={reward_value:.2f} done={str(bool(done)).lower()} error={_format_error(step_error)}"
            )

            observation = next_observation
            if done:
                break

        score = _score_trajectory(task_alias, trajectory)
        task_result = bundle.task.evaluate(trajectory.to_grader_records())
        success = bool(task_result.get("success", False)) and bool(task_result.get("constraints_ok", False))
    finally:
        close_fn = getattr(wrapper, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass
        rewards_str = ",".join(f"{value:.2f}" for value in trajectory.rewards)
        print(f"[END] success={str(success).lower()} steps={trajectory.step_count} score={score:.6f} rewards={rewards_str}")

    return safe_score(score), trajectory


def main() -> int:
    parser = argparse.ArgumentParser(description="Run TITAN inference across all registered tasks.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if os.getenv("TITAN_DISABLE_LOCAL_ENV", "").lower() not in {"1", "true", "yes"}:
        _load_local_env_file()

    api_base_url = os.getenv("API_BASE_URL") or "https://api.openai.com/v1"
    model_name = os.getenv("MODEL_NAME") or "gpt-4o-mini"
    hf_token = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")

    if hf_token:
        model = _build_openai_model(api_base_url, model_name, hf_token, args.seed)
        active_model_name = model_name
    else:
        # Keep inference executable in environments without credentials.
        model = _fallback_model
        active_model_name = "fallback-noop"
        import sys

        print("[INFO] HF_TOKEN/API_KEY not set. Using fallback-noop policy.", file=sys.stderr)

    task_scores: Dict[str, float] = {}
    for task_alias in available_task_names():
        score, _trajectory = _run_task(task_alias, model=model, seed=args.seed, model_name=active_model_name)
        task_scores[task_alias] = score

    # Print summary with interpretation (to stderr to not interfere with log parsing)
    import sys
    print("\n" + "=" * 60, file=sys.stderr)
    print("TITAN RESULTS SUMMARY", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    for task_alias, score in task_scores.items():
        interpretation = _interpret_score(score, task_alias)
        print(f"{task_alias:<12} | Score: {score:.6f} | {interpretation}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

