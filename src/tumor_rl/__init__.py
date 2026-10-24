"""tumor-rl: treatment scheduling for prostate cancer on digital twins fitted to real trial patients."""

from .env import TreatmentConfig, TreatmentEnv, VirtualPatient
from .model import TumorParams, relative_psa, simulate

__all__ = ["TreatmentEnv", "TreatmentConfig", "VirtualPatient", "TumorParams", "simulate", "relative_psa"]
__version__ = "0.1.0"
