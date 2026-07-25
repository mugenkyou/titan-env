from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from titan_env.core.environment import FaultInjector, INTENSITY_PROFILES, TITANGymEnv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO/RecurrentPPO on TITAN.")
    parser.add_argument(
        "--algo",
        choices=("ppo", "recurrent_ppo"),
        default="recurrent_ppo",
        help="RL algorithm to train.",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        help="Override policy class (defaults: MlpPolicy or MlpLstmPolicy).",
    )
    parser.add_argument("--profile", choices=("low", "medium", "high", "storm"), default="medium")
    parser.add_argument(
        "--curriculum-profiles",
        nargs="+",
        choices=("low", "medium", "high", "storm"),
        default=None,
        help="If set, train sequentially across these radiation profiles.",
    )
    parser.add_argument("--timesteps", type=int, default=300_000, help="Single-profile timesteps.")
    parser.add_argument(
        "--phase-timesteps",
        type=int,
        default=200_000,
        help="Per-profile timesteps when curriculum is enabled.",
    )
    parser.add_argument("--reward-version", choices=("v1", "v2", "v3"), default="v3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument(
        "--save-path",
        type=str,
        default="models/lstm_ppo_v3_curriculum_seed42",
        help="Model output path. Stable-Baselines will append .zip if missing.",
    )
    return parser.parse_args()


def _make_vec_env(profile: str, reward_version: str, seed: int, max_steps: int):
    try:
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as exc:
        raise ImportError(
            "stable-baselines3 is required. Install with: pip install sb3-contrib"
        ) from exc

    def _factory():
        injector = FaultInjector(INTENSITY_PROFILES[profile], training_mode=True)
        return TITANGymEnv(
            fault_injector=injector,
            reward_version=reward_version,
            max_steps=max_steps,
            seed=seed,
            training_mode=True,
        )

    return DummyVecEnv([_factory])


def _build_model(algo: str, policy: str | None, env, seed: int):
    if algo == "recurrent_ppo":
        try:
            from sb3_contrib import RecurrentPPO
        except ImportError as exc:
            raise ImportError(
                "sb3-contrib is required for RecurrentPPO. Install with: pip install sb3-contrib"
            ) from exc
        return RecurrentPPO(policy or "MlpLstmPolicy", env, verbose=1, seed=seed)

    from stable_baselines3 import PPO

    return PPO(policy or "MlpPolicy", env, verbose=1, seed=seed)


def _train_curriculum(
    model,
    profiles: Sequence[str],
    reward_version: str,
    seed: int,
    max_steps: int,
    phase_timesteps: int,
) -> None:
    for idx, profile in enumerate(profiles, start=1):
        env = _make_vec_env(profile=profile, reward_version=reward_version, seed=seed, max_steps=max_steps)
        model.set_env(env)
        print(f"[TRAIN] phase={idx} profile={profile} timesteps={phase_timesteps}")
        model.learn(total_timesteps=phase_timesteps, reset_num_timesteps=False)
        env.close()


def main() -> int:
    args = _parse_args()

    initial_profile = (
        args.curriculum_profiles[0] if args.curriculum_profiles else args.profile
    )
    env = _make_vec_env(
        profile=initial_profile,
        reward_version=args.reward_version,
        seed=args.seed,
        max_steps=args.max_steps,
    )

    model = _build_model(algo=args.algo, policy=args.policy, env=env, seed=args.seed)

    if args.curriculum_profiles:
        _train_curriculum(
            model=model,
            profiles=args.curriculum_profiles,
            reward_version=args.reward_version,
            seed=args.seed,
            max_steps=args.max_steps,
            phase_timesteps=args.phase_timesteps,
        )
    else:
        print(f"[TRAIN] profile={args.profile} timesteps={args.timesteps}")
        model.learn(total_timesteps=args.timesteps)

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    env.close()
    print(f"[SAVED] {save_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
