"""Local phase readout noise for the two-channel fluorescence detector.

The count model matches ``physics.fluorescence_scan``: Poisson atom arrivals,
state-dependent Poisson photons per atom, independent background photons and
electronic noise. The reported sensitivity uses linear error propagation of
the normalized signal, not a Fisher-information calculation. Bootstrap
intervals describe finite source sampling, not additional laboratory noise.
"""

import hashlib
from dataclasses import asdict

import numpy as np

from .joint_statistics import _cluster_contributions, _integer, _ratio, _summary
from .physics import GAMMA


def _detector(config):
    value = dict(config) if isinstance(config, dict) else asdict(config)
    positive = ("saturation_F1", "saturation_F2", "interaction_length_m",
                "collection_fraction", "quantum_efficiency", "gate_s")
    for key in positive:
        if key not in value or not np.isfinite(value[key]) or value[key] <= 0:
            raise ValueError("Detector " + key + " must be positive and finite")
    for key in ("background_counts", "electronic_noise_std_counts"):
        if key not in value or not np.isfinite(value[key]) or value[key] < 0:
            raise ValueError("Detector " + key + " must be nonnegative and finite")
    if value["collection_fraction"] > 1 or value["quantum_efficiency"] > 1:
        raise ValueError("Detector collection fraction and quantum efficiency must not exceed one")
    return value


def _contributions(replicas, incident_samples, detector):
    # Reuse the common physical/shape/ID validation, including missing IDs.
    _cluster_contributions(replicas, incident_samples)
    result = np.zeros((incident_samples, 15))
    common = GAMMA / 2 * detector["interaction_length_m"] * detector["collection_fraction"] * detector["quantum_efficiency"]
    k1 = common * detector["saturation_F1"] / (1 + detector["saturation_F1"])
    k2 = common * detector["saturation_F2"] / (1 + detector["saturation_F2"])
    for record in replicas:
        ids = np.asarray(record["atom_ids"], dtype=int)
        weight = np.asarray(record["detection_weights"], dtype=float)
        speed = np.asarray(record["vz_m_s"], dtype=float)
        p2 = np.asarray(record["fringe_coefficients"], dtype=float)
        p1 = -p2.copy()
        p1[:, 0] += 1
        q1, q2 = k1 / speed, k2 / speed
        result[ids, 0] += weight
        result[ids, 1] += weight * q1
        result[ids, 2] += weight * q2
        result[ids, 3:6] += (weight * q1)[:, None] * p1
        result[ids, 6:9] += (weight * q2)[:, None] * p2
        # Var(total photons) = lambda * (q + q^2), not just lambda*q.
        result[ids, 9:12] += (weight * (q1 + q1 ** 2))[:, None] * p1
        result[ids, 12:15] += (weight * (q2 + q2 ** 2))[:, None] * p2
    return result / len(replicas)


def _estimates(counts, contribution, phase, atoms_per_sample, detector):
    sums = counts @ contribution
    harmonic = np.array([1., np.cos(phase), np.sin(phase)])
    derivative = np.array([0., -np.sin(phase), np.cos(phase)])
    gain1, gain2 = _ratio(sums[:, 1], sums[:, 0]), _ratio(sums[:, 2], sums[:, 0])
    mu1 = atoms_per_sample * (sums[:, 3:6] @ harmonic)
    mu2 = atoms_per_sample * (sums[:, 6:9] @ harmonic)
    # Significant negative intensity means an invalid probability, never clip it.
    tolerance = 1e-7 * np.maximum(atoms_per_sample * (sums[:, 1] + sums[:, 2]), 1.)
    if np.any(mu1 < -tolerance) or np.any(mu2 < -tolerance):
        raise ValueError("Fringe gives negative fluorescence intensity at the operating phase")
    floor = detector["background_counts"] + detector["electronic_noise_std_counts"] ** 2
    var1 = atoms_per_sample * (sums[:, 9:12] @ harmonic) + floor
    var2 = atoms_per_sample * (sums[:, 12:15] @ harmonic) + floor
    f1, f2 = _ratio(mu1, gain1), _ratio(mu2, gain2)
    total = f1 + f2
    y = _ratio(f2, total)
    df1 = _ratio(atoms_per_sample * (sums[:, 3:6] @ derivative), gain1)
    df2 = _ratio(atoms_per_sample * (sums[:, 6:9] @ derivative), gain2)
    slope = _ratio(df2 * total - f2 * (df1 + df2), total ** 2)
    variance = _ratio(y ** 2 * _ratio(var1, gain1 ** 2) +
                      (1 - y) ** 2 * _ratio(var2, gain2 ** 2), total ** 2)
    if np.any(variance < 0):
        raise ValueError("Negative normalized fluorescence variance; probabilities must be physical")
    noise = np.sqrt(variance)
    snr = np.zeros(len(sums))
    active = np.isfinite(noise) & (noise > 0) & np.isfinite(slope)
    snr[active] = np.abs(slope[active]) / noise[active]
    phase_noise = np.full(len(sums), np.inf)
    np.divide(1., snr, out=phase_noise, where=snr > 0)
    return dict(readout_snr_per_rad=snr, phase_noise_rad=phase_noise,
                F1_counts=mu1 + detector["background_counts"],
                F2_counts=mu2 + detector["background_counts"],
                F1_signal_counts=mu1, F2_signal_counts=mu2,
                F1_variance=var1, F2_variance=var2,
                normalized_P2=y, normalized_slope_per_rad=slope, noise_std_P2=noise,
                gain_F1_counts_per_atom=gain1, gain_F2_counts_per_atom=gain2,
                expected_detected_atoms=atoms_per_sample * sums[:, 0])


def endpoint_statistics(scenarios, incident_samples, incident_flux_per_s, detection_config,
                        phase_rad=None, bootstrap=1000, seed=83087, flux_scale=1.0):
    """Return point/95% summaries and shared source-cluster bootstrap draws.

    ``scenarios`` uses the response-record format of ``case_statistics``.
    When phase_rad is None, the nominal *point* signal chooses its quadrature;
    that one phase is then fixed for all scenarios and all bootstrap draws.
    Passing a phase freezes an operating phase obtained on training data.
    ``flux_scale`` uniformly thins all incident arrivals while preserving the
    phase-space distribution, photon yield per atom and detector backgrounds.
    It is a counterfactual equal-flux diagnostic, not a changed source sample.
    Zero slope/no signal gives SNR=0 and unbounded phase noise (null in JSON).
    """
    incident_samples = _integer(incident_samples, "incident_samples")
    bootstrap = _integer(bootstrap, "bootstrap")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not np.isfinite(incident_flux_per_s) or incident_flux_per_s <= 0:
        raise ValueError("incident_flux_per_s must be positive and finite")
    if not np.isfinite(flux_scale) or not 0 < flux_scale <= 1:
        raise ValueError("flux_scale must lie in (0,1]")
    if not isinstance(scenarios, dict) or not scenarios or any(not isinstance(k, str) or not k for k in scenarios):
        raise ValueError("scenarios must be a nonempty dictionary with nonempty string names")
    if phase_rad is not None and not np.isfinite(phase_rad):
        raise ValueError("phase_rad must be finite")
    if phase_rad is None and "nominal" not in scenarios:
        raise ValueError("Automatic quadrature requires a nominal scenario")
    detector = _detector(detection_config)
    contributions = {name: _contributions(replicas, incident_samples, detector)
                     for name, replicas in scenarios.items()}
    automatic_phase = phase_rad is None
    if automatic_phase:
        nominal = contributions["nominal"].sum(axis=0)
        # q1/q2 is constant across velocities, so the normalized signal is
        # harmonic and its quadrature follows the raw F2 harmonic exactly.
        phase_rad = float((np.arctan2(nominal[8], nominal[7]) + np.pi / 2) % (2 * np.pi))
    phase_rad = float(phase_rad)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(incident_samples, np.full(incident_samples, 1 / incident_samples), size=bootstrap)
    identity = dict(incident_samples=incident_samples, bootstrap=bootstrap, seed=int(seed),
                    counts_sha256=hashlib.sha256(counts.astype("<i8").tobytes()).hexdigest())
    point_counts = np.ones((1, incident_samples))
    atoms_per_sample = incident_flux_per_s * flux_scale * detector["gate_s"] / incident_samples
    points, scenario_draws, summaries = {}, {}, {}
    for name, contribution in contributions.items():
        points[name] = _estimates(point_counts, contribution, phase_rad, atoms_per_sample, detector)
        scenario_draws[name] = _estimates(counts, contribution, phase_rad, atoms_per_sample, detector)
        summaries[name] = _summary(points[name], scenario_draws[name])
        summaries[name]["recoil_replicas"] = len(scenarios[name])

    def robust(values):
        return dict(worst_readout_snr_per_rad=np.min(np.stack([v["readout_snr_per_rad"] for v in values.values()]), axis=0),
                    worst_phase_noise_rad=np.max(np.stack([v["phase_noise_rad"] for v in values.values()]), axis=0))

    robust_point, robust_draws = robust(points), robust(scenario_draws)
    summary = dict(scenarios=summaries, robust=_summary(robust_point, robust_draws),
        operating_phase_rad=phase_rad,
        phase_selection="nominal point quadrature, frozen across scenarios and bootstrap" if automatic_phase else "supplied fixed phase",
        incident_samples=incident_samples, incident_flux_per_s=float(incident_flux_per_s),
        flux_scale=float(flux_scale), detection_config=detector,
        bootstrap_identity=identity,
        score_definition="abs(d normalized_F2 / d phase) / std(normalized_F2), per detection gate",
        noise_model="Compound Poisson atoms/photons plus independent background and electronic noise; local linear ratio error propagation, not exact Fisher information",
        interval_scope="Source-ID Monte Carlo sampling, with recoil trajectories averaged within ID. The bootstrap is not detector-noise resampling or a continuous-domain guarantee. Undefined intervals remain null.")
    draws = dict(robust_draws, scenarios=scenario_draws, bootstrap_identity=identity)
    return summary, draws
