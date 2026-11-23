"""Finite spatial-pulse Rabi and linear-frequency-chirp experiments (SI units).

Each trajectory uses the flight clock measured from its MOT exit. No averaging
over asynchronous laboratory arrival times is implied by these scans.
"""

import copy

import numpy as np

from .physics import (K_EFF, effective_acceleration, fit_fringe, fluorescence_scan,
                      populations_from_unitary, preparation_weights,
                      prepare_ensemble, raman_zone, raman_peak_frequency)
from .states import resolved_state_handling


def _initial_state(ensemble):
    state = np.zeros((len(ensemble.r_m), 2), complex)
    state[:, 0] = 1
    return state


def _readout(states, ensemble, config, stream, noise=None):
    probability, population = populations_from_unitary(states, ensemble, config.raman)
    detector = copy.deepcopy(config.detection)
    if noise is not None:
        detector.noise = noise
    rng = np.random.default_rng(np.random.SeedSequence(config.seed, spawn_key=(stream,)))
    detection = fluorescence_scan(probability, ensemble, preparation_weights(ensemble, config.raman),
                                  detector, config.raman, rng)
    return probability, population, detection


def _result(prepared, states, experiment):
    config, ensemble = prepared["config"], prepared["ensemble"]
    probability, population, detection = _readout(states, ensemble, config, 30 if experiment == "rabi" else 31)
    return dict(prepared, experiment=experiment,
                state_handling=resolved_state_handling(ensemble, config.raman),
                per_atom_P2=probability, per_atom_ground_populations=population,
                detection=detection,
                max_state_norm_error=float(np.max(np.abs(np.sum(np.abs(states) ** 2, axis=-1) - 1))),
                clock_convention="Each trajectory uses elapsed flight time since its MOT exit; common reset reference. "
                                 "Asynchronous laboratory arrivals are not averaged; this is not a free-running "
                                 "continuous-chirp steady-state gate calculation.")


def run_rabi_scan(config=None, ensemble=None, progress=None, prepared=None):
    """Scan effective coupling at fixed first-zone width, then record one transit.

    The area axis is defined at the reference velocity and transverse beam center.
    Actual atoms retain their velocity-dependent area, Doppler shift and density.
    Scaling both Raman beam intensities together scales Omega and the configured
    differential light shift linearly; the spatial transit time stays fixed.
    """
    prepared = prepared if prepared is not None else prepare_ensemble(config, ensemble, progress)
    config, ensemble = prepared["config"], prepared["ensemble"]
    r, options = config.raman, config.rabi_scan
    width = r.waist_z_m if r.envelope == "gaussian" else r.rectangular_width_m
    extent = r.envelope_half_widths if r.envelope == "gaussian" else 0.5
    integrated_width = width * np.sqrt(np.pi / 2) if r.envelope == "gaussian" else width
    # The area axis remains a calibrated control scan even with an explicit peak.
    actual_peak = raman_peak_frequency(r)
    if actual_peak <= 0:
        raise ValueError("A reference-area Rabi scan requires nonzero Raman coupling")
    scale_per_pi = r.reference_vz_m_s / (2 * integrated_width * actual_peak)
    areas = np.linspace(options.area_min_pi, options.area_max_pi, options.points)
    initial = _initial_state(ensemble)
    states = []
    for i, area in enumerate(areas):
        states.append(raman_zone(initial, ensemble, r, 0, amplitude_scale=scale_per_pi * area))
        if progress is not None:
            progress(i + 1, len(areas), len(ensemble.r_m))
    result = _result(prepared, np.asarray(states), "rabi")
    history = raman_zone(initial, ensemble, r, 0, amplitude_scale=scale_per_pi * options.transit_area_pi,
                         return_history=True)
    _, _, transit_detection = _readout(history["states"], ensemble, config, 32, noise=False)
    result.update(scan_values=areas, reference_area_pi=areas,
                  peak_rabi_Hz=areas * r.reference_vz_m_s / (2 * integrated_width),
                  reference_transit_s=2 * extent * width / r.reference_vz_m_s,
                  transit=dict(reference_elapsed_s=history["reference_elapsed_s"],
                               expected_photon_P2=transit_detection["expected_photon_P2"],
                               expected_atom_P2=transit_detection["expected_atom_P2"],
                               reference_area_pi=float(options.transit_area_pi)))
    if r.coupling_model == "rb87_d2_power":
        result["beam1_power_mW"] = 1e3 * r.beam1_power_W * scale_per_pi * areas
        result["beam2_power_mW"] = 1e3 * r.beam2_power_W * scale_per_pi * areas
    result["max_state_norm_error"] = max(result["max_state_norm_error"],
        float(np.max(np.abs(np.sum(np.abs(history["states"]) ** 2, axis=-1) - 1))))
    return result


def run_chirp_scan(config=None, ensemble=None, progress=None, prepared=None):
    """Repropagate all three pulses for every absolute chirp alpha in Hz/s.

    Laser phase contains -pi*alpha*t**2, with instantaneous angular detuning
    shifted by -2*pi*alpha*t. The third-zone phase offset is held fixed.
    """
    prepared = prepared if prepared is not None else prepare_ensemble(config, ensemble, progress)
    config, ensemble = prepared["config"], prepared["ensemble"]
    r, options = config.raman, config.chirp_scan
    alpha = np.linspace(options.center_Hz_s - options.half_range_Hz_s,
                        options.center_Hz_s + options.half_range_Hz_s, options.points)
    reference_T = r.zone_spacing_m / r.reference_vz_m_s
    if 2 * np.pi * (alpha[1] - alpha[0]) * reference_T ** 2 >= np.pi:
        raise ValueError("Chirp scan undersamples the reference fringe; increase chirp_scan.points")
    velocity = np.array([[0.0, 0.0, r.reference_vz_m_s]])
    acceleration = effective_acceleration(velocity, r)[0]
    reference_chirp = float(K_EFF * np.dot(r.k_direction, acceleration) / (2 * np.pi))
    states = []
    for i, slope in enumerate(alpha):
        pulse_config = copy.deepcopy(r)
        pulse_config.chirp_rad_s2 = 2 * np.pi * slope
        state = _initial_state(ensemble)
        for zone in range(3):
            state = raman_zone(state, ensemble, pulse_config, zone)
        states.append(state)
        if progress is not None:
            progress(i + 1, len(alpha), len(ensemble.r_m))
    result = _result(prepared, np.asarray(states), "chirp")
    phase = -2 * np.pi * (alpha - reference_chirp) * reference_T ** 2
    expected = result["detection"]["expected_photon_P2"]
    denominator = float(expected.max() + expected.min())
    result.update(scan_values=alpha, chirp_Hz_s=alpha, chirp_rad_s2=2 * np.pi * alpha,
                  reference_T_s=reference_T, reference_chirp_Hz_s=reference_chirp,
                  phase_coordinate_rad=phase,
                  fit=fit_fringe(phase, result["detection"]["observed_P2"]),
                  expected_fit=fit_fringe(phase, expected),
                  sampled_expected_visibility=(float((expected.max() - expected.min()) / denominator)
                                               if denominator > 1e-12 else None))
    return result
