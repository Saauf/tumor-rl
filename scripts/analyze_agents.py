"""Check that results hold across training seeds, and describe the rule the agents learned.

    python scripts/analyze_agents.py models/ppo_tumor.zip models/seeds/*.zip
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from tumor_rl.cohort import default_split, external_cohort
from tumor_rl.evaluate import evaluate, run_episode
from tumor_rl.policies import Continuous, RLAgent


def switch_points(agent: RLAgent, patients, seeds: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Measured PSA (relative to baseline) at visits where the agent stops / restarts treatment."""
    stops, starts = [], []
    for p in patients:
        for seed in range(seeds):
            _, hist = run_episode(agent, p, seed)
            psa, treat = np.array(hist["psa"]), np.array(hist["treatment"])
            for t in range(1, len(treat)):
                if treat[t - 1] == 1 and treat[t] == 0:
                    stops.append(psa[t])
                elif treat[t - 1] == 0 and treat[t] == 1:
                    starts.append(psa[t])
    return np.array(stops), np.array(starts)


def main(paths: list[str]) -> None:
    _, test = default_split()
    cohorts = {"held-out": test, "external": external_cohort()}
    print("| Model | Cohort | Progression-free 10 y (RL / continuous) | Patients worse than continuous "
          "| Time on treatment | Stop at PSA (median) | Restart at PSA (median) |")
    print("|---|---|---|---|---|---|---|")
    for path in paths:
        agent = RLAgent(path)
        for label, patients in cohorts.items():
            res = evaluate([agent, Continuous()], patients, seeds=20)
            rl, ct = res[agent.name], res[Continuous.name]
            worse = np.mean([a["ttp_months"] < b["ttp_months"] - 1
                             for a, b in zip(rl["per_patient"], ct["per_patient"])])
            stops, starts = switch_points(agent, patients)
            print(f"| {Path(path).stem} | {label} | {rl['progression_free_10y']:.0%} / "
                  f"{ct['progression_free_10y']:.0%} | {worse:.0%} | {rl['fraction_treated']:.0%} | "
                  f"{np.median(stops):.2f} | {np.median(starts):.2f} |")


if __name__ == "__main__":
    main(sys.argv[1:] or ["models/ppo_tumor.zip"])
