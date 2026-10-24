"""Fit the tumour model to each real patient ("digital twins").

Follows the protocol of Strobl et al. 2021: rS and dD are fixed; cost,
turnover, n0 and fR are fitted per patient by least squares on normalised PSA,
PSA(t)/PSA(0), with 25 random restarts. Fits with R^2 < 0.5 are flagged.

    python -m tumor_rl.fit                       # fit all 67 Bruchovsky patients
    python -m tumor_rl.fit --cohort shaw          # external cohort
    python -m tumor_rl.fit --holdout 0.5          # fit first half, forecast second half
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from .data import Patient, load_patients
from .model import TumorParams, relative_psa

# Bounds from Strobl et al. 2021. fR is fitted on a log10 scale for better conditioning.
LOWER = np.array([0.0, 0.0, 0.1, np.log10(1e-5)])
UPPER = np.array([1.0, 1.0, 1.0, np.log10(0.25)])
MIN_R2 = 0.5
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def _to_params(x: np.ndarray) -> TumorParams:
    return TumorParams(cost=float(x[0]), turnover=float(x[1]), n0=float(x[2]), f_r=float(10 ** x[3]))


@dataclass
class FitResult:
    key: str
    outcome: str
    params: TumorParams
    r2: float
    rmse: float
    aic: float
    n_obs: int
    psa_baseline: float
    fit_until_day: float
    forecast_rmse: float | None = None
    forecast_r2: float | None = None

    @property
    def good(self) -> bool:
        return self.r2 >= MIN_R2

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "params"}
        d["params"] = self.params.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FitResult":
        d = dict(d)
        d["params"] = TumorParams(**d["params"])
        return cls(**d)


def _observations(patient: Patient, until_day: float | None):
    mask = ~np.isnan(patient.psa)
    if until_day is not None:
        mask &= patient.day <= until_day
    return patient.day[mask].astype(int), patient.psa[mask] / patient.psa_baseline


def _r2_rmse(pred: np.ndarray, obs: np.ndarray) -> tuple[float, float]:
    sse = float(np.sum((pred - obs) ** 2))
    sst = float(np.sum((obs - obs.mean()) ** 2))
    return 1.0 - sse / sst if sst > 0 else float("nan"), float(np.sqrt(sse / len(obs)))


def fit_patient(patient: Patient, n_starts: int = 25, seed: int = 0,
                until_day: float | None = None) -> FitResult:
    """Fit one patient. ``until_day`` restricts fitting to early data (for forecasting tests)."""
    days, obs = _observations(patient, until_day)
    schedule = patient.daily_schedule()[: days[-1]]

    def residual(x):
        return relative_psa(_to_params(x), schedule)[days] - obs

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_starts):
        x0 = rng.uniform(LOWER, UPPER)
        sol = least_squares(residual, x0, bounds=(LOWER, UPPER), xtol=1e-8, ftol=1e-8)
        if best is None or sol.cost < best.cost:
            best = sol

    params = _to_params(best.x)
    pred = relative_psa(params, schedule)[days]
    r2, rmse = _r2_rmse(pred, obs)
    n, k = len(obs), 4
    aic = n * np.log(max(2 * best.cost, 1e-12) / n) + 2 * k
    result = FitResult(patient.key, patient.outcome, params, r2, rmse, float(aic), n,
                       patient.psa_baseline, float(days[-1]))

    if until_day is not None:  # score the forecast on data the fit never saw
        all_days, all_obs = _observations(patient, None)
        future = all_days > until_day
        if future.sum() >= 3:
            full_pred = relative_psa(params, patient.daily_schedule())[all_days[future]]
            result.forecast_r2, result.forecast_rmse = _r2_rmse(full_pred, all_obs[future])
    return result


def _job(args):
    patient, n_starts, holdout = args
    until = None
    if holdout:
        until = patient.day[~np.isnan(patient.psa)][-1] * holdout
    return fit_patient(patient, n_starts=n_starts, seed=patient.patient_id, until_day=until)


def fit_cohort(patients: list[Patient], n_starts: int = 25, holdout: float | None = None,
               processes: int | None = None) -> list[FitResult]:
    with mp.get_context("spawn").Pool(processes) as pool:
        return pool.map(_job, [(p, n_starts, holdout) for p in patients])


def save_fits(fits: list[FitResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([f.to_dict() for f in fits], indent=1))


def load_fits(path: Path) -> list[FitResult]:
    return [FitResult.from_dict(d) for d in json.loads(Path(path).read_text())]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohort", default="bruchovsky", choices=["bruchovsky", "shaw"])
    parser.add_argument("--starts", type=int, default=25)
    parser.add_argument("--holdout", type=float, default=None,
                        help="fit only this fraction of each patient's follow-up, forecast the rest")
    args = parser.parse_args()

    fits = fit_cohort(load_patients(args.cohort), args.starts, args.holdout)
    suffix = f"_holdout{args.holdout}" if args.holdout else ""
    out = RESULTS_DIR / f"fits_{args.cohort}{suffix}.json"
    save_fits(fits, out)

    good = [f for f in fits if f.good]
    print(f"{len(good)}/{len(fits)} patients fitted with R^2 >= {MIN_R2}")
    print(f"median R^2 {np.median([f.r2 for f in fits]):.2f}, median RMSE {np.median([f.rmse for f in fits]):.3f}")
    if args.holdout:
        fr = [f.forecast_r2 for f in fits if f.forecast_r2 is not None]
        fe = [f.forecast_rmse for f in fits if f.forecast_rmse is not None]
        print(f"forecast on unseen data: median R^2 {np.median(fr):.2f}, median RMSE {np.median(fe):.3f}")
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
