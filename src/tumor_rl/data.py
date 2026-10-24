"""Load real patient data from the intermittent androgen deprivation (IAD) trials.

Source: the dataset compiled by Tanaka et al. and hosted by Dr. Nicholas
Bruchovsky (http://www.nicholasbruchovsky.com/clinicalResearch.html). It holds
monthly PSA and testosterone measurements for:

* 72 patients from the Vancouver Phase II IAD trial (Bruchovsky et al.,
  Cancer 2006; J Urol 2007): locally advanced / biochemically recurrent
  prostate cancer treated with cyproterone acetate (CPA) + leuprolide (LEU),
  stopped when PSA < 4 ug/L and restarted when PSA > 10 ug/L.
* 18 patients from Shaw et al. (BJU Int 2007), a separate IAD cohort.

The data is not redistributed with this package. Run
``python -m tumor_rl.data --download`` to fetch it; the archive's SHA-256 is
checked so everyone works with the identical file.

Cleaning applied (all logged in ``Patient.notes``):
* rows are sorted by day (two Bruchovsky files list a few late rows out of order);
* rows sharing a day are merged, keeping the last treatment state and any PSA;
* PSA reported as 0 is set to the assay floor of 0.1 ug/L;
* files whose columns do not match the documented layout are skipped.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DATA_URL = "https://www.nicholasbruchovsky.com/dataTanaka.zip"
DATA_SHA256 = "845683954e99253ddf68e488a48223e89c19247a92aec27daa79130d8f46904f"
DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PSA_FLOOR = 0.1  # ug/L, lowest non-zero value reported in the dataset

COHORTS = {"bruchovsky": "Bruchovsky_et_al", "shaw": "Shaw_et_al"}


@dataclass
class Patient:
    patient_id: int
    cohort: str
    outcome: str                  # "no_relapse", "relapse", "metastasis"
    day: np.ndarray               # days since first visit
    psa: np.ndarray               # ug/L, NaN where not measured
    testosterone: np.ndarray      # nmol/L, NaN where not measured
    on_treatment: np.ndarray      # 1 = androgen deprivation given from this visit on, 0 = off
    cycle: np.ndarray
    notes: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.cohort}-{self.patient_id:03d}"

    @property
    def psa_baseline(self) -> float:
        return float(self.psa[~np.isnan(self.psa)][0])

    def daily_schedule(self) -> np.ndarray:
        """Treatment (0/1) for every day from day 0 to the last visit.

        The state recorded at a visit holds until the next visit, which is how
        the trial was run (decisions were made at monthly clinic visits).
        """
        schedule = np.zeros(int(self.day[-1]), dtype=float)
        for start, end, on in zip(self.day[:-1], self.day[1:], self.on_treatment[:-1]):
            schedule[int(start):int(end)] = on
        return schedule


def download(dest: Path = DEFAULT_DIR) -> Path:
    """Fetch and verify the dataset archive, then unpack it into ``dest``."""
    dest.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(DATA_URL, timeout=60) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != DATA_SHA256:
        raise RuntimeError(f"Checksum mismatch: got {digest}, expected {DATA_SHA256}")
    (dest / "dataTanaka.zip").write_bytes(payload)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(dest)
    return dest / "dataTanaka"


def _parse_outcomes(path: Path) -> dict[tuple[str, int], str]:
    """Read ``classificationsofpatients.txt`` into {(cohort, id): outcome}."""
    outcomes, cohort, label = {}, None, None
    for line in path.read_text().splitlines():
        text = line.strip()
        if text.startswith("Bruchovsky"):
            cohort = "bruchovsky"
        elif text.startswith("Shaw"):
            cohort = "shaw"
        elif text.startswith("Without relapse"):
            label = "no_relapse"
        elif text.startswith("With metastasis"):
            label = "metastasis"
        elif text.startswith("With relapse"):
            label = "relapse"
        elif label and text and text[0].isdigit():
            for token in text.replace(",", " ").split():
                outcomes[(cohort, int(token))] = label
    return outcomes


def _num(text: str) -> float:
    text = text.strip().strip('"')
    return float(text) if text else np.nan


def _load_file(path: Path, cohort: str, outcome: str) -> Patient | None:
    rows = [r for r in csv.reader(path.open()) if r]
    if not rows or any(len(r) != 10 for r in rows) or any(r[7] not in ("0", "1") for r in rows):
        return None
    notes = []
    days = [int(r[8]) for r in rows]
    if days != sorted(days):
        notes.append("rows re-sorted by day")
    rows = sorted(rows, key=lambda r: int(r[8]))  # stable: keeps file order within a day

    merged: dict[int, dict] = {}
    for r in rows:
        d = int(r[8])
        entry = merged.setdefault(d, {"psa": np.nan, "t": np.nan})
        if d in merged and "on" in entry:
            notes.append(f"merged duplicate visit on day {d}")
        psa, tst = _num(r[4]), _num(r[5])
        entry["psa"] = psa if not np.isnan(psa) else entry["psa"]
        entry["t"] = tst if not np.isnan(tst) else entry["t"]
        entry["on"] = int(r[7])
        entry["cycle"] = int(r[6])

    day0 = min(merged)
    keys = sorted(merged)
    psa = np.array([merged[k]["psa"] for k in keys])
    if np.any(psa <= 0):
        notes.append(f"{int(np.sum(psa <= 0))} PSA value(s) of 0 set to {PSA_FLOOR} ug/L")
        psa = np.where(psa <= 0, PSA_FLOOR, psa)
    if np.isnan(psa[0]):
        return None
    return Patient(
        patient_id=int(rows[0][0]),
        cohort=cohort,
        outcome=outcome,
        day=np.array([k - day0 for k in keys], dtype=float),
        psa=psa,
        testosterone=np.array([merged[k]["t"] for k in keys]),
        on_treatment=np.array([merged[k]["on"] for k in keys], dtype=float),
        cycle=np.array([merged[k]["cycle"] for k in keys]),
        notes=notes,
    )


def load_patients(cohort: str = "bruchovsky", root: Path = DEFAULT_DIR / "dataTanaka",
                  exclude_metastatic: bool = True) -> list[Patient]:
    """Load one cohort. Metastatic patients are excluded by default, as in Strobl et al. 2021."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"{root} not found. Run `python -m tumor_rl.data --download` first.")
    outcomes = _parse_outcomes(root / "classificationsofpatients.txt")
    patients = []
    for path in sorted((root / COHORTS[cohort]).glob("patient*.txt")):
        pid = int(path.stem.replace("patient", ""))
        outcome = outcomes.get((cohort, pid), "unknown")
        if exclude_metastatic and outcome == "metastasis":
            continue
        patient = _load_file(path, cohort, outcome)
        if patient is not None:
            patients.append(patient)
    return patients


def main() -> None:
    parser = argparse.ArgumentParser(description="Download or summarise the IAD trial data.")
    parser.add_argument("--download", action="store_true", help="fetch and verify the dataset")
    args = parser.parse_args()
    if args.download:
        print(f"Data unpacked to {download()}")
    for cohort in COHORTS:
        patients = load_patients(cohort, exclude_metastatic=False)
        n_psa = sum(int(np.sum(~np.isnan(p.psa))) for p in patients)
        print(f"{cohort}: {len(patients)} patients, {n_psa} PSA measurements")
        for p in patients:
            for note in p.notes:
                print(f"  {p.key}: {note}")


if __name__ == "__main__":
    main()
