"""Train a PPO agent on the training split of virtual patients.

    python -m tumor_rl.train --timesteps 2000000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from .cohort import default_split
from .env import TreatmentConfig, TreatmentEnv


def train(timesteps: int, n_envs: int, seed: int, out: Path, param_jitter: float) -> Path:
    train_patients, _ = default_split()
    config = TreatmentConfig(param_jitter=param_jitter)
    env = make_vec_env(TreatmentEnv, n_envs=n_envs, seed=seed,
                       env_kwargs={"patients": train_patients, "config": config})
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=2048,
        batch_size=4096,
        n_epochs=10,
        gamma=0.995,          # long horizon: a month now vs. years later both matter
        gae_lambda=0.95,
        learning_rate=3e-4,
        ent_coef=0.005,
        policy_kwargs={"net_arch": [64, 64]},
        seed=seed,
        device="cpu",
        verbose=1,
    )
    model.learn(total_timesteps=timesteps)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    return out.with_suffix(".zip")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jitter", type=float, default=0.1,
                        help="log-normal noise on patient parameters during training (robustness)")
    parser.add_argument("--out", type=Path, default=Path("models/ppo_tumor"))
    args = parser.parse_args()
    print(f"Saved model to {train(args.timesteps, args.n_envs, args.seed, args.out, args.jitter)}")


if __name__ == "__main__":
    main()
