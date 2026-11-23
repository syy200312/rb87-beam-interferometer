"""Evaluate local single-beam gyroscope readout using the frozen power model.

This is a conditional white detection-noise prediction, not a measured Allan
deviation or a time-dependent technical-noise simulation.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from rb87_beam.config import SimulationConfig
from rb87_beam.endpoint_statistics import endpoint_statistics
from rb87_beam.io import read_stateful_ensemble
from rb87_beam.joint_optimization import response_record
from rb87_beam.physics import K_EFF

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("outputs/gyro_sensitivity_estimate.json"),
                    help="Destination JSON file for this estimate")
args = parser.parse_args()
root = Path("beam_outputs/raman_power_calibration")
config = SimulationConfig.from_dict(json.loads((root / "config_power_driven.json").read_text(encoding="utf-8")))
study_path = Path("beam_outputs/endpoint_comparison/study.json")
if study_path.exists():
    study = json.loads(study_path.read_text(encoding="utf-8"))
    phase = study["strategies"]["joint"]["endpoint"]["operating_phase_rad"]
else:
    sample = json.loads(Path("samples/operating_phase.json").read_text(encoding="utf-8"))
    phase = sample["operating_phase_rad"]
source = Path("beam_outputs/endpoint_comparison/native_validation/baseline/nominal/prepared_0.h5")
ensemble = read_stateful_ensemble(source)
with np.load(source.with_name("response_0.npz")) as f:
    photons = f["scattered_photons"].copy()
config.pumping.enabled = False
config.detection.noise = False
config.raman.scan_points = 5
n = len(ensemble.r_m)

def evaluate(rotation):
    c = copy.deepcopy(config)
    c.raman.rotation_rad_s = (0., rotation, 0.)
    response = response_record(c, ensemble, photons)
    stat, _ = endpoint_statistics({"nominal": [response]}, n, ensemble.flux_per_s,
        c.detection, phase_rad=phase, bootstrap=32)
    return stat["scenarios"]["nominal"]

zero = evaluate(0.)
derivatives = []
for step in (1e-6, 5e-7):
    plus, minus = evaluate(step), evaluate(-step)
    slope = (plus["normalized_P2"] - minus["normalized_P2"]) / (2 * step)
    derivatives.append(dict(step_rad_s=step, dP2_dOmega_y=slope,
        P2_plus=plus["normalized_P2"], P2_minus=minus["normalized_P2"]))
derivative = derivatives[-1]["dP2_dOmega_y"]
relative_change = abs(derivative / derivatives[0]["dP2_dOmega_y"] - 1)
assert relative_change < 1e-5
sigma_gate = zero["noise_std_P2"] / abs(derivative)
# An explicit operating assumption: continuous, independent nonoverlapping
# 10-ms records, with the phase held at the frozen working point.
cycle = config.detection.gate_s
allan_coefficient = sigma_gate * np.sqrt(cycle)
reference_T = config.raman.zone_spacing_m / config.raman.reference_vz_m_s
report = dict(
    model="single forward beam, y rotation, fixed trained working phase, zero nominal rotation",
    input_config=str(root / "config_power_driven.json"), source_checkpoint=str(source),
    independent_MC_atoms=n, incident_flux_per_s=ensemble.flux_per_s,
    operating_phase_rad=phase, operating_phase_source="frozen endpoint joint training phase, not retuned on this sample",
    gate_s=config.detection.gate_s, assumed_independent_sample_period_s=cycle,
    reference_speed_m_s=config.raman.reference_vz_m_s, reference_T_s=reference_T,
    ideal_short_pulse_scale_factor_magnitude_s=2*K_EFF*config.raman.reference_vz_m_s*reference_T**2,
    local_effective_scale_factor_s=derivative / zero["normalized_slope_per_rad"],
    finite_difference=derivatives, derivative_relative_change=relative_change,
    expected_P2=zero["normalized_P2"], noise_std_P2=zero["noise_std_P2"],
    phase_noise_per_gate_rad=zero["phase_noise_rad"],
    detected_atoms_per_gate=zero["expected_detected_atoms"],
    dP2_dOmega_y=derivative, rotation_noise_per_gate_rad_s=sigma_gate,
    white_allan_coefficient_rad_per_sqrt_s=allan_coefficient,
    predicted_allan_deviation_at_1s_rad_s=allan_coefficient,
    one_sided_ASD_rad_s_per_sqrt_Hz=np.sqrt(2)*allan_coefficient,
    angle_random_walk_deg_per_sqrt_hour=allan_coefficient*180/np.pi*60,
    predicted_white_allan=[dict(tau_s=tau, sigma_rotation_rad_s=allan_coefficient/np.sqrt(tau)) for tau in (1,10,100)],
    conventions="sigma_Allan(tau)=N/sqrt(tau), N=sigma_gate*sqrt(Tc); strict one-sided PSD ASD=sqrt(2)*N; ARW_deg/sqrt(h)=N*180/pi*60.",
    noise_model="Poisson atom arrivals with state splitting, Poisson photons per atom, independent known background and electronic noise; normalized-ratio local error propagation.",
    scope=["Single-beam prediction with configured non-rotational acceleration held fixed; dual-beam common-mode rejection and closed-loop modulation are not included.",
           "The source flux 1e10/s is assumed normalization, not experimentally calibrated.",
           "No Raman-scattering decoherence, laser phase/intensity noise, vibration, magnetic noise or technical drift time series.",
           "Nonoverlapping independent gates and no dead time are assumptions; a multi-point fringe fit has a different measurement duration.",
           "The deterministic phase-fit residual is numerical error, not detector noise. A predicted white-noise Allan law is not a simulated or measured stability record.",
           "Finite-difference step stability does not establish Monte Carlo sample convergence; this calculation uses one 256-atom prepared realization."],
)
path = args.output
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
