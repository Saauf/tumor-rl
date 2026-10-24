"""Turn fitted patients into virtual patients and split them for training and testing."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .env import VirtualPatient
from .fit import RESULTS_DIR, load_fits


def virtual_patients(fits_path: Path, only_good: bool = True) -> list[VirtualPatient]:
    """Digital twins from a fits file, keeping only fits with R^2 >= 0.5 by default."""
    fits = load_fits(fits_path)
    return [VirtualPatient(f.key, f.params, f.psa_baseline, f.outcome)
            for f in fits if f.good or not only_good]


def train_test_split(patients: list[VirtualPatient], test_fraction: float = 0.3,
                     seed: int = 0) -> tuple[list[VirtualPatient], list[VirtualPatient]]:
    """Split patients, stratified by clinical outcome, so the agent is tested on people it never saw."""
    rng = np.random.default_rng(seed)
    train, test = [], []
    for outcome in sorted({p.outcome for p in patients}):
        group = [p for p in patients if p.outcome == outcome]
        order = rng.permutation(len(group))
        n_test = int(round(test_fraction * len(group)))
        test += [group[i] for i in order[:n_test]]
        train += [group[i] for i in order[n_test:]]
    return train, test


def default_split(seed: int = 0):
    return train_test_split(virtual_patients(RESULTS_DIR / "fits_bruchovsky.json"), seed=seed)


def external_cohort() -> list[VirtualPatient]:
    return virtual_patients(RESULTS_DIR / "fits_shaw.json")
