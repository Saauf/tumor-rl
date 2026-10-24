"""Compare treatment strategies on held-out and external virtual patients.

    python -m tumor_rl.evaluate --model models/ppo_tumor.zip
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cohort import default_split, external_cohort
from .env import TreatmentConfig, TreatmentEnv, VirtualPatient
from .fit import RESULTS_DIR
from .policies import Policy, RLAgent, clinical_policies
from .tune import tuned_rule


def run_episode(policy: Policy, patient: VirtualPatient, seed: int,
                config: TreatmentConfig | None = None) -> tuple[dict, dict]:
    env = TreatmentEnv([patient], config or TreatmentConfig())
    obs, info = env.reset(seed=seed, options={"patient_index": 0})
    policy.reset(info)
    done = False
    while not done:
        obs, _, terminated, truncated, info = env.step(policy.act(obs, info))
        done = terminated or truncated
    result = {
        "ttp_months": env.month,
        "progressed": env.progressed,
        "fraction_treated": env.months_treated / env.month,
    }
    return result, env.history


def evaluate(policies: list[Policy], patients: list[VirtualPatient], seeds: int = 20) -> dict:
    """Every policy sees the same patients with the same measurement-noise seeds."""
    out = {}
    for policy in policies:
        per_patient = []
        for p in patients:
            runs = [run_episode(policy, p, seed)[0] for seed in range(seeds)]
            per_patient.append({
                "patient": p.key,
                "outcome": p.outcome,
                "ttp_months": float(np.mean([r["ttp_months"] for r in runs])),
                "progressed": float(np.mean([r["progressed"] for r in runs])),
                "fraction_treated": float(np.mean([r["fraction_treated"] for r in runs])),
            })
        ttp = np.array([r["ttp_months"] for r in per_patient])
        out[policy.name] = {
            "median_ttp_months": float(np.median(ttp)),
            "mean_ttp_months": float(np.mean(ttp)),
            "progression_free_10y": float(np.mean([1 - r["progressed"] for r in per_patient])),
            "fraction_treated": float(np.mean([r["fraction_treated"] for r in per_patient])),
            "per_patient": per_patient,
        }
    return out


def paired_vs(results: dict, a: str, b: str) -> dict:
    """Per-patient comparison of mean TTP: how often a is better / worse than b (by > 1 month)."""
    ta = np.array([r["ttp_months"] for r in results[a]["per_patient"]])
    tb = np.array([r["ttp_months"] for r in results[b]["per_patient"]])
    diff = ta - tb
    return {"better": float(np.mean(diff > 1)), "worse": float(np.mean(diff < -1)),
            "median_gain_months": float(np.median(diff))}


def to_markdown(results: dict) -> str:
    lines = ["| Strategy | Progression-free at 10 y | Mean time to progression (months, max 120) "
             "| Time on treatment |",
             "|---|---|---|---|"]
    for name, r in results.items():
        lines.append(f"| {name} | {r['progression_free_10y']:.0%} | {r['mean_ttp_months']:.1f} | "
                     f"{r['fraction_treated']:.0%} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=Path("models/ppo_tumor.zip"))
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()

    policies = clinical_policies()
    rule = tuned_rule()
    if rule is not None:
        policies.append(rule)
    if args.model.exists():
        policies.append(RLAgent(args.model))
    _, test = default_split()
    report = {}
    for label, patients in [("held_out_bruchovsky", test), ("external_shaw", external_cohort())]:
        results = evaluate(policies, patients, args.seeds)
        report[label] = results
        print(f"\n## {label} ({len(patients)} patients)\n")
        print(to_markdown(results))
        if RLAgent.name in results:
            for other in [p.name for p in policies if p.name != RLAgent.name]:
                c = paired_vs(results, RLAgent.name, other)
                print(f"RL vs {other}: better for {c['better']:.0%}, worse for {c['worse']:.0%} of patients, "
                      f"median gain {c['median_gain_months']:+.1f} months")
    (RESULTS_DIR / "evaluation.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
