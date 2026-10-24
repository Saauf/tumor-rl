"""Grid-search the best simple threshold rule on the *training* patients.

Gives the RL agent a strong, interpretable competitor tuned on exactly the same
data and objective, and tests whether a two-number rule is all the agent learned.

    python -m tumor_rl.tune
"""

from __future__ import annotations

import json

import numpy as np

from .cohort import default_split
from .env import TreatmentConfig, TreatmentEnv
from .fit import RESULTS_DIR
from .policies import ThresholdRule

TUNED_PATH = RESULTS_DIR / "tuned_threshold_rule.json"


def mean_return(policy, patients, seeds: int, config: TreatmentConfig) -> float:
    total = []
    for p in patients:
        for seed in range(seeds):
            env = TreatmentEnv([p], config)
            obs, info = env.reset(seed=seed, options={"patient_index": 0})
            policy.reset(info)
            ret, done = 0.0, False
            while not done:
                obs, r, term, trunc, info = env.step(policy.act(obs, info))
                ret += r
                done = term or trunc
            total.append(ret)
    return float(np.mean(total))


def tune(seeds: int = 3) -> dict:
    train, _ = default_split()
    config = TreatmentConfig()
    results = []
    for off in np.arange(0.10, 0.65, 0.05):
        for on in np.arange(off + 0.05, 1.05, 0.05):
            rule = ThresholdRule(round(off, 2), round(on, 2))
            results.append((mean_return(rule, train, seeds, config), round(off, 2), round(on, 2)))
    best = max(results)
    return {"off_below": best[1], "on_above": best[2], "train_return": best[0],
            "grid": [{"return": r, "off_below": a, "on_above": b} for r, a, b in results]}


def tuned_rule() -> ThresholdRule | None:
    if not TUNED_PATH.exists():
        return None
    d = json.loads(TUNED_PATH.read_text())
    return ThresholdRule(d["off_below"], d["on_above"],
                         f"Threshold rule tuned on training patients ({d['off_below']:.2f} / {d['on_above']:.2f})")


def main() -> None:
    result = tune()
    TUNED_PATH.write_text(json.dumps(result, indent=1))
    print(f"Best rule on training patients: stop below {result['off_below']:.2f}, "
          f"restart above {result['on_above']:.2f} x baseline PSA (mean return {result['train_return']:.1f})")


if __name__ == "__main__":
    main()
