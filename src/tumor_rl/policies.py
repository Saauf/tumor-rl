"""Treatment strategies: clinical protocols and the RL agent.

All policies decide from what a clinician can observe (measured PSA and
treatment history), never from the hidden sensitive/resistant split.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

OFF, ON = 0, 1


class Policy:
    name = "policy"

    def reset(self, info: dict) -> None:
        pass

    def act(self, obs: np.ndarray, info: dict) -> int:
        raise NotImplementedError


class Continuous(Policy):
    """Standard of care: continuous androgen deprivation at full dose."""

    name = "Continuous (standard)"

    def act(self, obs, info):
        return ON


class ThresholdRule(Policy):
    """Stop treatment when PSA falls below ``off_below`` x baseline; restart at ``on_above`` x baseline."""

    def __init__(self, off_below: float, on_above: float, name: str | None = None):
        self.off_below, self.on_above = off_below, on_above
        self.name = name or f"Threshold rule ({off_below:.2f} / {on_above:.2f})"

    def reset(self, info):
        self.treating = True

    def act(self, obs, info):
        psa = info["psa_relative"]
        if self.treating and psa < self.off_below:
            self.treating = False
        elif not self.treating and psa >= self.on_above:
            self.treating = True
        return ON if self.treating else OFF


class AdaptiveZhang(ThresholdRule):
    """Adaptive therapy as in Zhang et al., Nat Commun 2017 (Moffitt abiraterone trial):
    treat until PSA falls below 50% of baseline, pause until it returns to baseline."""

    name = "Adaptive (Zhang 2017)"

    def __init__(self):
        super().__init__(0.5, 1.0, self.name)


class IntermittentBruchovsky(Policy):
    """The protocol actually used in the Vancouver trial (Bruchovsky et al. 2006):
    36-week lead-in on treatment, then stop when PSA < 4 ug/L and restart when PSA > 10 ug/L."""

    name = "Intermittent (trial protocol)"

    def __init__(self, lead_in_months: int = 9, off_below: float = 4.0, on_above: float = 10.0):
        self.lead_in, self.off_below, self.on_above = lead_in_months, off_below, on_above

    def reset(self, info):
        self.treating = True

    def act(self, obs, info):
        if info["month"] < self.lead_in:
            return ON
        psa = info["psa_ug_per_l"]
        if self.treating and psa < self.off_below:
            self.treating = False
        elif not self.treating and psa > self.on_above:
            self.treating = True
        return ON if self.treating else OFF


class RLAgent(Policy):
    """A trained Stable-Baselines3 PPO model, acting deterministically."""

    name = "RL agent (PPO)"

    def __init__(self, model_path: str | Path):
        from stable_baselines3 import PPO

        self.model = PPO.load(str(model_path), device="cpu")

    def act(self, obs, info):
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)


def clinical_policies() -> list[Policy]:
    return [Continuous(), AdaptiveZhang(), IntermittentBruchovsky()]
