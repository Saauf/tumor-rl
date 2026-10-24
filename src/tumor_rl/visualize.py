"""Figures: model fits to real patients, validation, and treatment strategies.

    python -m tumor_rl.visualize
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .cohort import default_split
from .data import load_patients
from .evaluate import run_episode
from .fit import RESULTS_DIR, load_fits
from .model import relative_psa
from .policies import RLAgent, clinical_policies
from .tune import tuned_rule

COLORS = {"Continuous (standard)": "#4c72b0", "Adaptive (Zhang 2017)": "#dd8452",
          "Intermittent (trial protocol)": "#55a868", "RL agent (PPO)": "#c44e52"}
RULE_COLOR = "#8172b3"
OUTCOME_COLORS = {"no_relapse": "#4c72b0", "relapse": "#c44e52"}


def _shade_treatment(ax, days, on):
    for start, end, flag in zip(days[:-1], days[1:], on[:-1]):
        if flag:
            ax.axvspan(start, end, color="#999999", alpha=0.15, lw=0)


def plot_fit_gallery(path: Path, n: int = 12):
    fits = {f.key: f for f in load_fits(RESULTS_DIR / "fits_bruchovsky.json")}
    patients = [p for p in load_patients() if p.key in fits]
    # Show a spread: best, typical and poorest fits, both outcomes.
    patients.sort(key=lambda p: -fits[p.key].r2)
    pick = [patients[i] for i in np.linspace(0, len(patients) - 1, n).astype(int)]
    fig, axes = plt.subplots(3, 4, figsize=(15, 9), sharey=False)
    for ax, p in zip(axes.flat, pick):
        f = fits[p.key]
        sched = p.daily_schedule()
        pred = relative_psa(f.params, sched) * p.psa_baseline
        _shade_treatment(ax, p.day, p.on_treatment)
        ax.plot(np.arange(len(pred)), pred, color="#c44e52", lw=1.8, label="model")
        ax.plot(p.day, p.psa, "o", ms=3, color="black", label="measured PSA")
        ax.set_title(f"{p.key} ({p.outcome.replace('_', ' ')})\nR² = {f.r2:.2f}", fontsize=9)
        ax.tick_params(labelsize=8)
    axes[0, 0].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("days")
    for ax in axes[:, 0]:
        ax.set_ylabel("PSA (µg/L)")
    fig.suptitle("Model fitted to real patients from the Vancouver IAD trial (grey = on treatment)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_cost_turnover(path: Path):
    fits = [f for f in load_fits(RESULTS_DIR / "fits_bruchovsky.json") if f.good]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for outcome, color in OUTCOME_COLORS.items():
        sel = [f for f in fits if f.outcome == outcome]
        ax.scatter([f.params.turnover for f in sel], [f.params.cost for f in sel], color=color,
                   label=f"{outcome.replace('_', ' ')} (n={len(sel)})", alpha=0.8, edgecolor="white")
    ax.set_xlabel("turnover (death rate / proliferation rate)")
    ax.set_ylabel("cost of resistance")
    ax.set_title("Fitted patient parameters (cf. Strobl et al. 2021, Fig. 5C)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_forecast(path: Path, n: int = 4):
    holdout = {f.key: f for f in load_fits(RESULTS_DIR / "fits_bruchovsky_holdout0.5.json")}
    patients = [p for p in load_patients() if holdout[p.key].forecast_rmse is not None]
    patients.sort(key=lambda p: holdout[p.key].forecast_rmse)
    pick = [patients[i] for i in np.linspace(0, len(patients) - 1, n).astype(int)]
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.4))
    for ax, p in zip(axes, pick):
        f = holdout[p.key]
        pred = relative_psa(f.params, p.daily_schedule()) * p.psa_baseline
        _shade_treatment(ax, p.day, p.on_treatment)
        ax.axvline(f.fit_until_day, color="black", ls="--", lw=1)
        ax.plot(np.arange(len(pred)), pred, color="#c44e52", lw=1.8)
        ax.plot(p.day, p.psa, "o", ms=3, color="black")
        ax.set_title(f"{p.key}: forecast RMSE {f.forecast_rmse:.2f}", fontsize=9)
        ax.set_xlabel("days")
    axes[0].set_ylabel("PSA (µg/L)")
    fig.suptitle("Honest check: fit on data left of the dashed line, predict the right (best → worst)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_strategies(path: Path, policies, patients, seed: int = 0):
    fig, axes = plt.subplots(2, len(patients), figsize=(6 * len(patients), 6.5), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    axes = np.atleast_2d(axes).reshape(2, -1)
    for col, patient in enumerate(patients):
        ax, tax = axes[0, col], axes[1, col]
        for row, policy in enumerate(policies):
            _, hist = run_episode(policy, patient, seed)
            months = np.array(hist["day"]) / 30.44
            color = COLORS.get(policy.name, RULE_COLOR)
            ax.plot(months, hist["burden"], color=color, lw=2, label=policy.name)
            res = np.array(hist["resistant"]) / patient.params.n0
            ax.plot(months, res, color=color, lw=1, ls=":")
            treat = np.array(hist["treatment"])
            tax.fill_between(months[:-1] + 0.5, row, row + 0.8 * treat, step="mid", color=color)
        ax.axhline(1.2, color="black", lw=0.8, ls="--")
        ax.text(1.0, 1.22, "progression (120% of baseline)", fontsize=8, ha="right",
                transform=ax.get_yaxis_transform())
        ax.set_ylim(0, 1.35)
        ax.set_title(f"Virtual patient {patient.key} ({patient.outcome.replace('_', ' ')}, held out)", fontsize=10)
        ax.set_ylabel("tumour burden / baseline\n(dotted: resistant cells)")
        tax.set_yticks(np.arange(len(policies)) + 0.4, [p.name.split(" (")[0].replace("Threshold rule tuned on training patients", "Tuned rule") for p in policies], fontsize=8)
        tax.set_xlabel("months")
        tax.set_title("months on treatment", fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_summary(path: Path):
    report = json.loads((RESULTS_DIR / "evaluation.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    cohorts = list(report)
    names = list(report[cohorts[0]])
    width = 0.8 / len(cohorts)
    for i, cohort in enumerate(cohorts):
        x = np.arange(len(names)) + i * width
        pf = [report[cohort][n]["progression_free_10y"] * 100 for n in names]
        tr = [report[cohort][n]["fraction_treated"] * 100 for n in names]
        label = cohort.replace("_", " ")
        axes[0].bar(x, pf, width, label=label, alpha=0.85 if i == 0 else 0.5)
        axes[1].bar(x, tr, width, label=label, alpha=0.85 if i == 0 else 0.5)
    for ax, title in zip(axes, ["Progression-free at 10 years (%)", "Time on treatment (%)"]):
        ax.set_xticks(np.arange(len(names)) + width * (len(cohorts) - 1) / 2,
                      [n.replace(" (", "\n(") for n in names], fontsize=8)
        ax.set_title(title)
        ax.set_ylim(0, 105)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=Path("models/ppo_tumor.zip"))
    parser.add_argument("--out", type=Path, default=Path("assets"))
    parser.add_argument("--patients", nargs="*", default=None, help="held-out patient keys to plot")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    plot_fit_gallery(args.out / "fits.png")
    plot_cost_turnover(args.out / "cost_turnover.png")
    plot_forecast(args.out / "forecast_check.png")

    policies = clinical_policies()
    rule = tuned_rule()
    if rule is not None:
        policies.append(rule)
    if args.model.exists():
        policies.append(RLAgent(args.model))
    _, test = default_split()
    if args.patients:
        chosen = [p for p in test if p.key in args.patients]
    else:
        chosen = [next(p for p in test if p.outcome == "relapse"),
                  next(p for p in test if p.outcome == "no_relapse")]
    plot_strategies(args.out / "strategies.png", policies, chosen)
    if (RESULTS_DIR / "evaluation.json").exists():
        plot_summary(args.out / "summary.png")
    print(f"Figures written to {args.out}/")


if __name__ == "__main__":
    main()
