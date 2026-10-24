"""Lotka-Volterra competition model of drug-sensitive and resistant tumour cells.

This is the model of Strobl et al., "Turnover modulates the need for a cost of
resistance in adaptive therapy", Cancer Research 81(4):1135-1147 (2021),
reimplemented independently from the published equations:

    dS/dt = rS * (1 - (S + R)/K) * (1 - dD * D(t)) * S - dT * S
    dR/dt = rR * (1 - (S + R)/K) * R                   - dT * R

* S, R    sensitive and resistant cell densities, scaled so K = 1
* rS      proliferation rate of sensitive cells, fixed at 0.027 / day
* rR      proliferation rate of resistant cells, rR = (1 - cost) * rS
* dT      turnover (death) rate, dT = turnover * rS, same for both populations
* dD      drug kill strength, fixed at 1.5; only dividing cells are killed
* D(t)    1 while on androgen deprivation, 0 during a treatment holiday

Serum PSA is taken to be proportional to tumour burden, so
PSA(t) / PSA(0) = N(t) / N(0) with N = S + R.

Integration uses classical RK4 with a 1-day step; treatment changes only at
whole days (clinic visits), and the fastest rate in the system is < 0.05/day,
so the step error is negligible (checked against scipy in the tests).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

R_S = 0.027
D_D = 1.5
K = 1.0


@dataclass(frozen=True)
class TumorParams:
    cost: float       # fractional growth-rate penalty of resistant cells, in [0, 1]
    turnover: float   # death rate as a fraction of rS, in [0, 1]
    n0: float         # initial tumour burden as a fraction of carrying capacity
    f_r: float        # initial fraction of resistant cells
    r_s: float = R_S
    d_d: float = D_D

    @property
    def r_r(self) -> float:
        return (1.0 - self.cost) * self.r_s

    @property
    def d_t(self) -> float:
        return self.turnover * self.r_s

    @property
    def initial_state(self) -> tuple[float, float]:
        return self.n0 * (1.0 - self.f_r), self.n0 * self.f_r

    def to_dict(self) -> dict:
        return asdict(self)


def _derivs(s: float, r: float, drug: float, r_s: float, r_r: float, d_d: float, d_t: float):
    crowding = 1.0 - (s + r) / K
    return (r_s * crowding * (1.0 - d_d * drug) * s - d_t * s,
            r_r * crowding * r - d_t * r)


def advance(state: tuple[float, float], params: TumorParams, drug: float, days: int,
            dt: float = 1.0) -> tuple[float, float]:
    """Integrate the model forward ``days`` days at a constant treatment level."""
    s, r = state
    r_s, r_r, d_d, d_t = params.r_s, params.r_r, params.d_d, params.d_t
    for _ in range(int(round(days / dt))):
        k1s, k1r = _derivs(s, r, drug, r_s, r_r, d_d, d_t)
        k2s, k2r = _derivs(s + 0.5 * dt * k1s, r + 0.5 * dt * k1r, drug, r_s, r_r, d_d, d_t)
        k3s, k3r = _derivs(s + 0.5 * dt * k2s, r + 0.5 * dt * k2r, drug, r_s, r_r, d_d, d_t)
        k4s, k4r = _derivs(s + dt * k3s, r + dt * k3r, drug, r_s, r_r, d_d, d_t)
        s += dt / 6.0 * (k1s + 2 * k2s + 2 * k3s + k4s)
        r += dt / 6.0 * (k1r + 2 * k2r + 2 * k3r + k4r)
    return max(s, 0.0), max(r, 0.0)


def simulate(params: TumorParams, schedule: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a daily treatment schedule. Returns S and R for days 0..len(schedule)."""
    s_out = np.empty(len(schedule) + 1)
    r_out = np.empty(len(schedule) + 1)
    state = params.initial_state
    s_out[0], r_out[0] = state
    for day, drug in enumerate(schedule):
        state = advance(state, params, float(drug), 1)
        s_out[day + 1], r_out[day + 1] = state
    return s_out, r_out


def relative_psa(params: TumorParams, schedule: np.ndarray) -> np.ndarray:
    """Predicted PSA(t) / PSA(0) for each day 0..len(schedule)."""
    s, r = simulate(params, schedule)
    return (s + r) / params.n0
