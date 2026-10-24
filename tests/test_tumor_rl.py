import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from scipy.integrate import solve_ivp

from tumor_rl.data import DEFAULT_DIR, Patient, load_patients
from tumor_rl.env import VISIT_DAYS, TreatmentEnv, VirtualPatient
from tumor_rl.fit import fit_patient
from tumor_rl.model import TumorParams, relative_psa, simulate
from tumor_rl.policies import AdaptiveZhang, Continuous, IntermittentBruchovsky

PARAMS = TumorParams(cost=0.4, turnover=0.2, n0=0.6, f_r=0.01)
HAS_DATA = (DEFAULT_DIR / "dataTanaka").exists()
needs_data = pytest.mark.skipif(not HAS_DATA, reason="run `python -m tumor_rl.data --download`")


def _patient(params: TumorParams, schedule: np.ndarray, every: int = 28) -> Patient:
    """A synthetic patient with monthly PSA measured from the model itself."""
    days = np.arange(0, len(schedule) + 1, every)
    psa = relative_psa(params, schedule)[days] * 12.0
    on = np.append(schedule[days[:-1]], 0)
    return Patient(0, "synthetic", "unknown", days.astype(float), psa, np.full(len(days), np.nan),
                   on, np.ones(len(days)))


# ------------------------------------------------------------------- model

def test_rk4_matches_scipy():
    schedule = np.array([1] * 300 + [0] * 200 + [1] * 300, dtype=float)
    s, r = simulate(PARAMS, schedule)

    def rhs(t, y):
        drug = schedule[min(int(t), len(schedule) - 1)]
        crowd = 1 - y[0] - y[1]
        return [PARAMS.r_s * crowd * (1 - PARAMS.d_d * drug) * y[0] - PARAMS.d_t * y[0],
                PARAMS.r_r * crowd * y[1] - PARAMS.d_t * y[1]]

    ref = solve_ivp(rhs, (0, len(schedule)), PARAMS.initial_state, t_eval=np.arange(len(schedule) + 1),
                    rtol=1e-10, atol=1e-12, max_step=0.5)
    np.testing.assert_allclose(s, ref.y[0], rtol=1e-4, atol=1e-8)
    np.testing.assert_allclose(r, ref.y[1], rtol=1e-4, atol=1e-8)


def test_treatment_kills_sensitive_and_selects_resistance():
    s_on, r_on = simulate(PARAMS, np.ones(365))
    s_off, r_off = simulate(PARAMS, np.zeros(365))
    assert s_on[-1] < PARAMS.initial_state[0] < s_off[-1]
    assert r_on[-1] / (s_on[-1] + r_on[-1]) > r_off[-1] / (s_off[-1] + r_off[-1])


def test_untreated_tumour_approaches_equilibrium():
    s, r = simulate(PARAMS, np.zeros(20000))
    assert s[-1] + r[-1] == pytest.approx(1 - PARAMS.turnover, rel=1e-3)


# --------------------------------------------------------------------- fit

def test_fit_recovers_known_parameters():
    schedule = np.array(([1] * 252 + [0] * 308) * 3, dtype=float)  # changes only at visits
    fit = fit_patient(_patient(PARAMS, schedule), n_starts=8)
    assert fit.r2 > 0.999
    np.testing.assert_allclose(relative_psa(fit.params, schedule), relative_psa(PARAMS, schedule), atol=0.01)


# --------------------------------------------------------------------- env

def _env(**kw):
    return TreatmentEnv([VirtualPatient("test", PARAMS, 12.0)], **kw)


def test_env_passes_gymnasium_checker():
    check_env(_env(), skip_render_check=True)


def test_progression_ends_episode():
    env = _env(psa_noise=0.0)
    env.reset(seed=0)
    done, reward = False, None
    while not done:
        _, reward, terminated, truncated, _ = env.step(0)  # never treat
        done = terminated or truncated
    assert terminated and reward == 0.0 and env.burden > 1.2


def test_treatment_burden_in_reward():
    env = _env(psa_noise=0.0, treatment_burden=0.25)
    env.reset(seed=0)
    assert env.step(1)[1] == pytest.approx(0.75)


def test_env_matches_model_with_no_noise():
    env = _env(psa_noise=0.0)
    env.reset(seed=0)
    for _ in range(5):
        env.step(1)
    expected = relative_psa(PARAMS, np.ones(5 * VISIT_DAYS))[-1]
    assert env.burden == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------- policies

def test_adaptive_zhang_switches_on_thresholds():
    pol = AdaptiveZhang()
    pol.reset({})
    assert pol.act(None, {"psa_relative": 0.8}) == 1
    assert pol.act(None, {"psa_relative": 0.4}) == 0
    assert pol.act(None, {"psa_relative": 0.9}) == 0
    assert pol.act(None, {"psa_relative": 1.0}) == 1


def test_trial_protocol_lead_in_and_thresholds():
    pol = IntermittentBruchovsky()
    pol.reset({})
    assert pol.act(None, {"month": 3, "psa_ug_per_l": 1.0}) == 1          # lead-in
    assert pol.act(None, {"month": 10, "psa_ug_per_l": 3.0}) == 0         # < 4 ug/L -> stop
    assert pol.act(None, {"month": 11, "psa_ug_per_l": 9.0}) == 0
    assert pol.act(None, {"month": 12, "psa_ug_per_l": 11.0}) == 1        # > 10 ug/L -> restart


def test_continuous_always_treats():
    assert Continuous().act(None, {}) == 1


# -------------------------------------------------------------------- data

@needs_data
def test_cohort_sizes_match_publications():
    assert len(load_patients(exclude_metastatic=False)) == 72
    assert len(load_patients()) == 67  # as in Strobl et al. 2021


@needs_data
def test_cleaning_is_applied_and_logged():
    patients = {p.patient_id: p for p in load_patients()}
    assert np.all(np.diff(patients[16].day) > 0)
    assert np.nanmin(patients[100].psa) > 0
    assert any("re-sorted" in n for n in patients[16].notes)
    assert len(patients[1].daily_schedule()) == int(patients[1].day[-1])
