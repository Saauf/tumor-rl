"""Gymnasium environment: monthly treatment decisions for a virtual patient.

Each episode is one "digital twin": a real trial patient whose tumour model
was fitted to their PSA history. Every 28 days (one clinic visit) the agent
sees only what a clinician would see (noisy PSA and treatment history) and
decides whether to give androgen deprivation for the next 28 days.

Endpoint, following Strobl et al. 2021 / RECIST: progression occurs when
tumour burden exceeds 120% of its pre-treatment value. Follow-up is capped
at 10 years.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .model import TumorParams, advance

VISIT_DAYS = 28
HISTORY = 6


@dataclass(frozen=True)
class VirtualPatient:
    key: str
    params: TumorParams
    psa_baseline: float  # ug/L, PSA at the first visit
    outcome: str = "unknown"


@dataclass
class TreatmentConfig:
    horizon_months: int = 120          # 10 years of follow-up
    progression_threshold: float = 1.2 # burden relative to baseline
    psa_noise: float = 0.10            # log-normal sd of PSA measurement + biological noise
    treatment_burden: float = 0.1      # reward penalty per month on treatment (side effects)
    param_jitter: float = 0.0          # log-normal sd applied to fitted parameters each episode


class TreatmentEnv(gym.Env):
    """Observation (float32, length 2*HISTORY + 3):
        * log(PSA / baseline PSA) at the last HISTORY visits (newest first)
        * treatment given after each of those visits (1 = on)
        * log(nadir PSA / baseline PSA) so far
        * months elapsed / horizon
        * fraction of months spent on treatment so far

    Action: 0 = treatment holiday, 1 = androgen deprivation for the next 28 days.
    Reward: +1 per progression-free month, minus ``treatment_burden`` per month treated.
    """

    metadata = {"render_modes": []}

    def __init__(self, patients: list[VirtualPatient], config: TreatmentConfig | None = None, **overrides):
        super().__init__()
        if not patients:
            raise ValueError("Need at least one virtual patient")
        self.patients = patients
        self.cfg = config or TreatmentConfig()
        for key, value in overrides.items():
            if not hasattr(self.cfg, key):
                raise TypeError(f"Unknown config option: {key}")
            setattr(self.cfg, key, value)
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(2 * HISTORY + 3,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    # ----------------------------------------------------------------- gym API

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        idx = options.get("patient_index")
        if idx is None:
            idx = int(self.np_random.integers(len(self.patients)))
        self.patient = self.patients[idx]
        self.params = self._jitter(self.patient.params)
        self.state = self.params.initial_state
        self.month = 0
        self.months_treated = 0
        self.progressed = False

        first = self._measure()
        self.psa_hist = np.full(HISTORY, first)
        self.treat_hist = np.zeros(HISTORY)
        self.nadir = first
        self.history = {"day": [0], "burden": [1.0], "psa": [first], "sensitive": [self.state[0]],
                        "resistant": [self.state[1]], "treatment": []}
        return self._obs(), self._info()

    def step(self, action):
        action = int(action)
        self.state = advance(self.state, self.params, float(action), VISIT_DAYS)
        self.month += 1
        self.months_treated += action

        psa = self._measure()
        self.psa_hist = np.roll(self.psa_hist, 1)
        self.psa_hist[0] = psa
        self.treat_hist = np.roll(self.treat_hist, 1)
        self.treat_hist[0] = action
        self.nadir = min(self.nadir, psa)

        burden = self.burden
        h = self.history
        h["day"].append(self.month * VISIT_DAYS)
        h["burden"].append(burden)
        h["psa"].append(psa)
        h["sensitive"].append(self.state[0])
        h["resistant"].append(self.state[1])
        h["treatment"].append(action)

        self.progressed = burden > self.cfg.progression_threshold
        reward = 0.0 if self.progressed else 1.0 - self.cfg.treatment_burden * action
        truncated = self.month >= self.cfg.horizon_months and not self.progressed
        return self._obs(), reward, self.progressed, truncated, self._info()

    # ----------------------------------------------------------------- helpers

    @property
    def burden(self) -> float:
        """True tumour burden relative to baseline (hidden from the agent)."""
        return (self.state[0] + self.state[1]) / self.params.n0

    def _jitter(self, p: TumorParams) -> TumorParams:
        sd = self.cfg.param_jitter
        if sd <= 0:
            return p
        z = self.np_random.normal(0.0, sd, size=3)
        return replace(p, cost=float(np.clip(p.cost * np.exp(z[0]), 0, 1)),
                       turnover=float(np.clip(p.turnover * np.exp(z[1]), 0, 1)),
                       f_r=float(np.clip(p.f_r * np.exp(z[2]), 1e-6, 0.5)))

    def _measure(self) -> float:
        """Noisy relative PSA, as returned by the lab at a visit."""
        return float(self.burden * np.exp(self.np_random.normal(0.0, self.cfg.psa_noise)))

    def _obs(self) -> np.ndarray:
        return np.concatenate([
            np.log(np.maximum(self.psa_hist, 1e-3)),
            self.treat_hist,
            [np.log(max(self.nadir, 1e-3)),
             self.month / self.cfg.horizon_months,
             self.months_treated / max(self.month, 1)],
        ]).astype(np.float32)

    def _info(self) -> dict:
        return {
            "patient": self.patient.key,
            "month": self.month,
            "psa_relative": float(self.psa_hist[0]),
            "psa_ug_per_l": float(self.psa_hist[0] * self.patient.psa_baseline),
            "burden": self.burden,
            "resistant_fraction": self.state[1] / max(self.state[0] + self.state[1], 1e-12),
            "months_treated": self.months_treated,
        }
