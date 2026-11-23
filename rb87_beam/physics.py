"""Reduced physical model. See README_Rb87束流.md for assumptions and signs."""

from dataclasses import dataclass, field

import numpy as np

from .config import SimulationConfig
from .states import (GROUND_LABELS, G_CLOCK, E_CLOCK, validate_density,
                     resolved_state_handling, raman_frame_density)

KB = 1.380649e-23
HBAR = 1.054571817e-34
MU_B = 9.2740100783e-24
MASS = 86.909180531 * 1.66053906660e-27
WAVELENGTH = 780.241209686e-9
GAMMA = 2 * np.pi * 6.065e6  # spontaneous population decay, rad/s convention
K = 2 * np.pi / WAVELENGTH
K_EFF = 2 * K
RECOIL_V = HBAR * K / MASS
HYPERFINE_HZ = 6.83468261090429e9


@dataclass
class Ensemble:
    r_m: np.ndarray  # (N, 3), common reference plane (source or pumping exit)
    v_m_s: np.ndarray
    flux_per_s: float
    source_metadata: dict = field(default_factory=lambda: {"mode": "provided_ensemble"})
    ground_density: np.ndarray = None  # (N,8,8), fixed laboratory-z F,m basis, after free decay
    exit_populations: np.ndarray = None  # (N,24), PyLCP local-axis populations at the exit plane
    exit_time_s: np.ndarray = None
    quantization_axes: np.ndarray = None  # local axes corresponding to exit_populations
    population_trace: np.ndarray = None  # t, number active, mean 24 local populations
    reference_plane_z_m: float = 0.0
    reference_time_s: np.ndarray = None  # flight time since MOT exit, NOT source residence time

    def validate(self):
        if self.r_m.ndim != 2 or self.r_m.shape[1] != 3 or self.r_m.shape != self.v_m_s.shape:
            raise ValueError("Ensemble positions and velocities must both have shape (N,3)")
        if len(self.r_m) == 0:
            raise ValueError("No atoms left in ensemble; inspect source/aperture parameters")
        if not np.all(np.isfinite(self.r_m)) or not np.all(np.isfinite(self.v_m_s)):
            raise ValueError("Non-finite ensemble coordinates")
        if (not np.isfinite(self.reference_plane_z_m) or self.reference_plane_z_m < 0
                or np.any(self.v_m_s[:, 2] <= 0)
                or not np.allclose(self.r_m[:, 2], self.reference_plane_z_m, atol=1e-9)):
            raise ValueError("Atoms must travel along +z from their common reference plane")
        if self.reference_time_s is not None:
            if (np.shape(self.reference_time_s) != (len(self.r_m),)
                    or not np.all(np.isfinite(self.reference_time_s)) or np.any(self.reference_time_s < 0)):
                raise ValueError("reference_time_s must contain N finite nonnegative flight times")
        elif self.reference_plane_z_m != 0:
            raise ValueError("A downstream reference plane requires reference_time_s")
        if not np.isfinite(self.flux_per_s) or self.flux_per_s <= 0:
            raise ValueError("Ensemble flux must be positive")
        if self.ground_density is not None:
            validate_density(self.ground_density, len(self.r_m))
        if self.exit_populations is not None:
            p = np.asarray(self.exit_populations)
            if p.shape != (len(self.r_m), 24) or not np.all(np.isfinite(p)) or np.min(p) < -1e-9 or not np.allclose(p.sum(axis=1), 1, atol=1e-8):
                raise ValueError("exit_populations must have shape (N,24) and normalized nonnegative rows")
        if self.exit_time_s is not None:
            if np.shape(self.exit_time_s) != (len(self.r_m),) or not np.all(np.isfinite(self.exit_time_s)) or np.any(self.exit_time_s < 0):
                raise ValueError("exit_time_s must contain N nonnegative finite times")
        if self.quantization_axes is not None:
            if np.shape(self.quantization_axes) != (len(self.r_m), 3) or not np.allclose(np.linalg.norm(self.quantization_axes, axis=1), 1):
                raise ValueError("quantization_axes must contain N unit vectors")


def disk_positions(rng, n, radius):
    azimuth = rng.uniform(0, 2 * np.pi, n)
    radial = radius * np.sqrt(rng.random(n))
    return np.column_stack((radial * np.cos(azimuth), radial * np.sin(azimuth), np.zeros(n)))


def calibrated_source(config, rng):
    """Sample a flux-weighted EXIT distribution; this does not simulate cooling."""
    n = config.n_atoms
    r = np.zeros((n, 3)) if config.calibrated_point_source else disk_positions(rng, n, config.exit_radius_m)
    sigma = np.sqrt(KB * config.calibrated_transverse_temperature_K / MASS)
    v = rng.normal(0, sigma, (n, 3))
    vz_sigma = config.calibrated_fwhm_vz_m_s / np.sqrt(8 * np.log(2))
    v[:, 2] = rng.normal(config.calibrated_mean_vz_m_s, vz_sigma, n)
    # Rejection sampling, never clipping a Gaussian into a delta spike at zero.
    for _ in range(100):
        bad = v[:, 2] <= 0
        if not np.any(bad):
            break
        v[bad, 2] = rng.normal(config.calibrated_mean_vz_m_s, vz_sigma, np.count_nonzero(bad))
    if np.any(v[:, 2] <= 0):
        raise ValueError("Could not sample forward velocities")
    return Ensemble(r, v, config.calibrated_flux_per_s,
                    {"mode": "calibrated", "sampled": n, "extracted": n,
                     "reference": config.calibration_reference,
                     "absolute_flux_is_calibrated": config.calibrated_flux_is_measured})


def mot_scattering_rates(r, v, source):
    """Six independent beams with shared saturation; effective cycling transition."""
    axes = np.array([0, 0, 1, 1, 2, 2])
    signs = np.array([1, -1, 1, -1, 1, -1])
    radius2 = np.sum(r[:, :2] ** 2, axis=1)
    # Long transverse beams: retain transverse profile, truncate at chamber ends.
    transverse = source.cooling_saturation * np.exp(
        -2 * r[:, [1, 1, 0, 0]] ** 2 / source.cooling_waist_m ** 2)
    axial_profile = np.exp(-2 * radius2 / source.cooling_waist_m ** 2)
    forward = source.axial_forward_saturation * axial_profile
    backward = source.axial_backward_saturation * axial_profile
    backward = backward * (radius2 >= source.exit_radius_m ** 2)
    saturation = np.column_stack((transverse, forward, backward))
    detuning = np.full_like(saturation, source.cooling_detuning_gamma * GAMMA)
    detuning[:, 4:] = source.axial_detuning_gamma * GAMMA
    detuning -= K * v[:, axes] * signs
    zeeman = (source.effective_magnetic_moment_muB * MU_B
              * source.magnetic_gradient_T_m / HBAR)
    detuning[:, :4] -= zeeman * r[:, [0, 0, 1, 1]] * signs[:4]
    rates = np.zeros_like(saturation)
    shared_saturation = 1 + saturation.sum(axis=1)[:, None]
    for subset, offsets, powers in (
            (slice(0, 4), source.cooling_spectrum_gamma, source.cooling_spectrum_weights),
            (slice(4, 6), source.axial_spectrum_gamma, source.axial_spectrum_weights)):
        for offset, power in zip(offsets, powers):
            rates[:, subset] += (GAMMA / 2 * power * saturation[:, subset]
                                 / (shared_saturation + 4 * (detuning[:, subset] / GAMMA + offset) ** 2))
    return rates


def mot_source(config, rng):
    """Langevin 2D+MOT, conditioned on a captured reservoir, with first-exit events."""
    n = config.n_atoms
    r = disk_positions(rng, n, config.initial_radius_m)
    v = rng.normal(0, np.sqrt(KB * config.initial_temperature_K / MASS), (n, 3))
    v[:, 2] = rng.normal(config.initial_vz_m_s, config.initial_sigma_vz_m_s, n)
    active = np.ones(n, dtype=bool)
    extracted_r, extracted_v = [], []
    trace = []
    time = 0.0
    step = 0
    while time < config.max_time_s and np.any(active):
        ids = np.flatnonzero(active)
        rr, vv = r[ids], v[ids]
        rates = mot_scattering_rates(rr, vv, config)
        dt = min(config.dt_s, config.max_time_s - time)
        rate_axes = rates.reshape(-1, 3, 2)
        acceleration = RECOIL_V * (rate_axes[:, :, 0] - rate_axes[:, :, 1])
        acceleration[:, 1] -= 9.80665
        diffusion = RECOIL_V ** 2 * (rate_axes.sum(axis=2) + rates.sum(axis=1)[:, None] / 3)
        new_v = vv + acceleration * dt + rng.normal(size=vv.shape) * np.sqrt(diffusion * dt)
        new_r = rr + 0.5 * (vv + new_v) * dt
        # A first-plane crossing is interpolated; do not count later returns again.
        crossed = (rr[:, 2] < config.mot_length_m) & (new_r[:, 2] >= config.mot_length_m)
        if np.any(crossed):
            f = ((config.mot_length_m - rr[crossed, 2])
                 / (new_r[crossed, 2] - rr[crossed, 2]))[:, None]
            hit_r = rr[crossed] + f * (new_r[crossed] - rr[crossed])
            hit_v = vv[crossed] + f * (new_v[crossed] - vv[crossed])
            accepted = np.sum(hit_r[:, :2] ** 2, axis=1) < config.exit_radius_m ** 2
            accepted &= hit_v[:, 2] > 0
            hit_r[:, 2] = 0.0
            extracted_r.extend(hit_r[accepted])
            extracted_v.extend(hit_v[accepted])
        lost = (np.sum(new_r[:, :2] ** 2, axis=1) >= config.chamber_radius_m ** 2)
        lost |= new_r[:, 2] < 0
        active[ids[crossed | lost]] = False
        r[ids], v[ids] = new_r, new_v
        time += dt
        step += 1
        if step % 100 == 0 or not np.any(active):
            remaining_v = v[active]
            transverse_T = (MASS * np.mean(np.var(remaining_v[:, :2], axis=0)) / KB
                            if len(remaining_v) >= 20 else np.nan)
            mean_vz = float(np.mean(remaining_v[:, 2])) if len(remaining_v) else np.nan
            trace.append([time, int(active.sum()), len(extracted_r),
                          transverse_T, mean_vz])
    if not extracted_r:
        raise ValueError("No MOT atoms extracted; adjust initial reservoir, pushing force or integration time")
    count = len(extracted_r)
    ensemble = Ensemble(np.asarray(extracted_r), np.asarray(extracted_v),
                        config.incident_flux_per_s * count / n,
                        {"mode": "mot", "sampled": n, "extracted": count,
                         "timed_out": int(active.sum()), "extraction_fraction": count / n,
                         "absolute_flux_is_calibrated": False,
                         "flux_model": "assumed incident captured-reservoir flux times extraction fraction"})
    return ensemble, np.asarray(trace)


def effective_acceleration(v, config):
    """First order in rotation: a_eff = a - 2 Omega x v, transverse components."""
    acceleration = np.asarray(config.acceleration_m_s2) + 2 * np.cross(v, config.rotation_rad_s)
    acceleration[:, 2] = 0.0  # frozen longitudinal speed, as stated in model scope
    return acceleration


def trajectory(ensemble, times_s, config):
    origin = ensemble.reference_time_s if ensemble.reference_time_s is not None else 0.0
    t = (np.asarray(times_s) - origin)[:, None]
    a = effective_acceleration(ensemble.v_m_s, config)
    return ensemble.r_m + ensemble.v_m_s * t + 0.5 * a * t ** 2, ensemble.v_m_s + a * t


def flight_time(ensemble, z_m):
    origin = ensemble.reference_time_s if ensemble.reference_time_s is not None else 0.0
    return origin + (z_m - ensemble.reference_plane_z_m) / ensemble.v_m_s[:, 2]


def reference_y(z, config):
    ref_v = np.array([[0.0, 0.0, config.reference_vz_m_s]])
    return 0.5 * effective_acceleration(ref_v, config)[0, 1] * (z / config.reference_vz_m_s) ** 2


def pulse_step(state, omega, phase, phase_rate, dt, light_shift=0.0):
    """Unitary midpoint update in interaction picture; exact for constant Omega, phase rate.

    H_I/hbar has coupling H_eg=Omega*exp(i*phase)/2 and differential
    diagonal shift light_shift. The laser-frame transform makes Omega=0 an
    identity when light_shift=0: Doppler/free phases are not counted twice.
    """
    detuning = phase_rate + light_shift
    generalized = np.hypot(omega, detuning)
    half_angle = generalized * dt / 2
    sinc = np.sinc(half_angle / np.pi) * dt / 2
    cosine = np.cos(half_angle)
    gg = np.exp(-0.5j * phase_rate * dt) * (cosine + 1j * detuning * sinc)
    ee = np.exp(0.5j * phase_rate * dt) * (cosine - 1j * detuning * sinc)
    ge = -1j * omega * sinc * np.exp(-1j * phase)
    eg = -1j * omega * sinc * np.exp(1j * phase)
    return np.column_stack((gg * state[:, 0] + ge * state[:, 1],
                            eg * state[:, 0] + ee * state[:, 1]))


def raman_optical_frequency_offset(config):
    """Laser-frequency difference minus hyperfine splitting, in Hz.

    two_photon_detuning_Hz in this simulator is relative to the recoil- and
    reference-Doppler-shifted clock transition, not the bare hyperfine line.
    Keep the power calibration consistent with the exported optical carrier.
    """
    return (config.two_photon_detuning_Hz + HBAR * K_EFF ** 2 / (4 * np.pi * MASS)
            + K_EFF * config.k_direction[2] * config.reference_vz_m_s / (2 * np.pi))


def raman_power_calibration(config, power_scale=1.0):
    """Physical per-color power mapping, evaluated only in the opt-in model."""
    if config.coupling_model != "rb87_d2_power":
        return None
    from .raman_calibration import calibrate_powers
    return calibrate_powers(config.beam1_power_W * power_scale, config.beam2_power_W * power_scale,
        config.waist_y_m, config.waist_z_m,
        detuning_Hz=config.single_photon_detuning_Hz,
        reference_Fprime=config.detuning_reference_Fprime,
        polarization1=config.polarization1, polarization2=config.polarization2,
        two_photon_detuning_Hz=raman_optical_frequency_offset(config))


def raman_peak_frequency(config):
    calibration = raman_power_calibration(config)
    if calibration is not None:
        return calibration["peak_rabi_Hz"]
    if config.peak_rabi_Hz > 0:
        return config.peak_rabi_Hz
    width = (config.waist_z_m * np.sqrt(np.pi / 2) if config.envelope == "gaussian"
             else config.rectangular_width_m)
    return config.reference_vz_m_s / (4 * width)


def raman_zone(state, ensemble, config, zone_index, scanned_phase=0.0,
               amplitude_scale=1.0, return_history=False):
    if not np.isfinite(amplitude_scale) or amplitude_scale < 0:
        raise ValueError("Raman amplitude_scale must be finite and nonnegative")
    z = config.first_zone_z_m + zone_index * config.zone_spacing_m
    centers = flight_time(ensemble, z)
    gaussian = config.envelope == "gaussian"
    extent = config.envelope_half_widths if gaussian else 0.5
    width = config.waist_z_m if gaussian else config.rectangular_width_m
    du = 2 * extent / config.integration_steps
    dt = du * width / ensemble.v_m_s[:, 2]
    area = (np.pi / 2, np.pi, np.pi / 2)[zone_index]
    # Integral exp(-2(v*t/w)^2) dt = w/v * sqrt(pi/2).
    integrated_width = width * np.sqrt(np.pi / 2) if gaussian else width
    peak = area * config.reference_vz_m_s / integrated_width * amplitude_scale
    if config.peak_rabi_Hz > 0:
        peak = 2 * np.pi * config.peak_rabi_Hz * (1, 2, 1)[zone_index] * amplitude_scale
    offset = config.phase_offsets_rad[zone_index] + scanned_phase
    peak_light_shift = config.differential_light_shift_Hz * amplitude_scale
    calibration = raman_power_calibration(config, (1, 2, 1)[zone_index] * amplitude_scale)
    if calibration is not None:
        # Both color powers, Omega_eff and the light shift scale together.
        # This also fixes the pi-zone light-shift scaling in the power model.
        factor = (1, 2, 1)[zone_index]
        peak = 2 * np.pi * calibration["peak_rabi_Hz"]
        peak_light_shift = (calibration["differential_light_shift_Hz"]
                            + factor * config.differential_light_shift_Hz * amplitude_scale)
        offset += calibration["coupling_phase_rad"]
    k = K_EFF * np.asarray(config.k_direction)
    history = [state.copy()] if return_history else None
    for j in range(config.integration_steps):
        u = -extent + (j + 0.5) * du
        t = centers + u * width / ensemble.v_m_s[:, 2]
        r, v = trajectory(ensemble, t, config)
        envelope = np.exp(-2 * ((u ** 2 if gaussian else 0) + ((r[:, 1] - reference_y(z, config)) / config.waist_y_m) ** 2))
        laser_offset = 2 * np.pi * config.two_photon_detuning_Hz + k[2] * config.reference_vz_m_s
        phase = r @ k - laser_offset * t - 0.5 * config.chirp_rad_s2 * t ** 2 + offset
        phase_rate = v @ k - laser_offset - config.chirp_rad_s2 * t
        light_shift = 2 * np.pi * peak_light_shift * envelope
        state = pulse_step(state, peak * envelope, phase, phase_rate, dt, light_shift)
        if return_history:
            history.append(state.copy())
    if return_history:
        return {"state": state, "states": np.asarray(history),
                "reference_elapsed_s": np.linspace(0, 2 * extent * width / config.reference_vz_m_s,
                                                    config.integration_steps + 1)}
    return state


def preparation_weights(ensemble, config):
    mode = resolved_state_handling(ensemble, config)
    if mode == "inherit":
        weights = np.ones(len(ensemble.r_m))
    elif mode == "project_F1_m0":
        weights = np.real(raman_frame_density(ensemble, config)[:, G_CLOCK, G_CLOCK]) * config.preparation_efficiency
    else:
        weights = np.full(len(ensemble.r_m), config.preparation_efficiency if config.remove_unprepared else 1.0)
    if config.selection_sigma_vx_m_s > 0:
        k = np.asarray(config.k_direction)
        resonant_vx = ((2 * np.pi * config.two_photon_detuning_Hz / K_EFF
                       - k[2] * (ensemble.v_m_s[:, 2] - config.reference_vz_m_s)) / k[0])
        weights *= np.exp(-0.5 * ((ensemble.v_m_s[:, 0] - resonant_vx) / config.selection_sigma_vx_m_s) ** 2)
    return weights


def fit_fringe(phase_rad, probability):
    """Fit A+B*cos(phi)+D*sin(phi) without unstable nonlinear frequency fitting."""
    x = np.asarray(phase_rad)
    y = np.asarray(probability)
    valid = np.isfinite(y)
    design = np.column_stack((np.ones(len(x)), np.cos(x), np.sin(x)))
    if valid.sum() < 4 or np.linalg.matrix_rank(design[valid]) < 3:
        raise ValueError("At least four valid, independent phase points required")
    coefficients = np.linalg.lstsq(design[valid], y[valid], rcond=None)[0]
    baseline, cosine, sine = coefficients
    amplitude = np.hypot(cosine, sine)
    residual = y[valid] - design[valid] @ coefficients
    rms = float(np.sqrt(np.mean(residual ** 2)))
    phase = float(np.arctan2(-sine, cosine)) if amplitude > 1e-12 else None
    return {"offset": float(baseline), "amplitude": float(amplitude),
            "contrast": float(amplitude / baseline) if baseline > 1e-12 else None,
            "phase_rad": phase, "residual_rms": rms,
            "phase_std_rad": float(np.sqrt(2 / valid.sum()) * rms / amplitude) if amplitude > 1e-12 else None,
            "coefficients": coefficients.tolist()}


def fluorescence_scan(probabilities, ensemble, weights, config, raman, rng):
    """Stationary continuous flux, compound Poisson atoms/photons, ideal two-channel readout."""
    times = flight_time(ensemble, config.z_m)
    r, _ = trajectory(ensemble, times, raman)
    radial2 = r[:, 0] ** 2 + (r[:, 1] - reference_y(config.z_m, raman)) ** 2
    accepted = radial2 < config.aperture_radius_m ** 2
    weights = weights * accepted
    if weights.sum() <= 0:
        raise ValueError("No prepared atoms reach the detector")
    expected_atoms = ensemble.flux_per_s / len(weights) * config.gate_s * weights
    efficiency = config.collection_fraction * config.quantum_efficiency
    transit = config.interaction_length_m / ensemble.v_m_s[:, 2]
    q1 = GAMMA / 2 * config.saturation_F1 / (1 + config.saturation_F1) * transit * efficiency
    q2 = GAMMA / 2 * config.saturation_F2 / (1 + config.saturation_F2) * transit * efficiency
    gain1 = np.average(q1, weights=weights)
    gain2 = np.average(q2, weights=weights)
    raw1, raw2, observed, ideal_photon, ideal_atom = [], [], [], [], []
    for probability in probabilities:
        mean1 = np.sum(expected_atoms * (1 - probability) * q1)
        mean2 = np.sum(expected_atoms * probability * q2)
        if config.noise:
            n1 = rng.poisson(expected_atoms * (1 - probability))
            n2 = rng.poisson(expected_atoms * probability)
            signal1 = rng.poisson(np.sum(n1 * q1) + config.background_counts)
            signal2 = rng.poisson(np.sum(n2 * q2) + config.background_counts)
            signal1 += rng.normal(0, config.electronic_noise_std_counts)
            signal2 += rng.normal(0, config.electronic_noise_std_counts)
        else:
            signal1 = mean1 + config.background_counts
            signal2 = mean2 + config.background_counts
        f1 = (signal1 - config.background_counts) / gain1
        f2 = (signal2 - config.background_counts) / gain2
        raw1.append(signal1)
        raw2.append(signal2)
        # Do not clip background-subtracted signals: clipping biases weak fringes.
        observed.append(f2 / (f1 + f2) if f1 + f2 > 0 else np.nan)
        ideal_photon.append((mean2 / gain2) / (mean1 / gain1 + mean2 / gain2))
        ideal_atom.append(np.average(probability, weights=weights))
    return {"F1_counts": np.asarray(raw1), "F2_counts": np.asarray(raw2),
            "observed_P2": np.asarray(observed), "expected_photon_P2": np.asarray(ideal_photon),
            "expected_atom_P2": np.asarray(ideal_atom), "accepted_mask": accepted,
            "detected_flux_per_s": float(expected_atoms.sum() / config.gate_s),
            "expected_atoms_per_gate": float(expected_atoms.sum()),
            "gain_F1_counts_per_atom": float(gain1), "gain_F2_counts_per_atom": float(gain2),
            "detection_weights": weights}


def prepare_ensemble(config=None, ensemble=None, progress=None):
    """Generate/load and prepare once, so several scans share the same atoms."""
    config = config or SimulationConfig()
    config.validate()
    if config.raman.coupling_model == "rb87_d2_power":
        # Check the strongest physical zone before doing an expensive MOT run.
        raman_power_calibration(config.raman, 2.)
    # Separate streams: source changes do not change the detector's random stream.
    source_seed = np.random.SeedSequence(config.seed).spawn(2)[0]
    source_rng = np.random.default_rng(source_seed)
    trace = np.empty((0, 5))
    if ensemble is None:
        if config.source.mode == "mot":
            ensemble, trace = mot_source(config.source, source_rng)
        elif config.source.mode == "pylcp":
            from .pylcp_mot import MOT2DPlus
            ensemble, trace = MOT2DPlus(config.source, config.pylcp).simulate(source_rng, progress)
        else:
            ensemble = calibrated_source(config.source, source_rng)
    ensemble.validate()
    source_ensemble, pumping_result = ensemble, None
    if config.pumping.enabled:
        from .pumping import OpticalPumping
        pumping_rng = np.random.default_rng(np.random.SeedSequence(config.seed, spawn_key=(2,)))
        pumping_result = OpticalPumping(config.pumping, config.raman).prepare(ensemble, pumping_rng, progress)
        ensemble = pumping_result["ensemble"]
    span = (config.raman.envelope_half_widths * config.raman.waist_z_m
            if config.raman.envelope == "gaussian" else config.raman.rectangular_width_m / 2)
    if ensemble.reference_plane_z_m >= config.raman.first_zone_z_m - span:
        raise ValueError("Ensemble reference plane must precede the first Raman envelope")
    resolved_state_handling(ensemble, config.raman)
    flight_times = flight_time(ensemble, config.detection.z_m)
    if np.max(flight_times) * np.linalg.norm(config.raman.rotation_rad_s) > 0.01:
        raise ValueError("First-order rotation approximation requires |Omega|*flight_time <= 0.01")
    result = {"config": config, "ensemble": ensemble, "source_trace": trace,
              "source_ensemble": source_ensemble, "pumping": pumping_result}
    calibration = raman_power_calibration(config.raman)
    if calibration is not None:
        result["raman_power_calibration"] = calibration
        result["raman_spin_transport"] = dict(
            model=config.raman.spin_transport, input_axis=[0., 0., 1.],
            raman_axis=list(config.raman.quantization_axis),
            population_preserving=True, reset_to_m0=False,
            note="Ideal nonzero-field adiabatic transport; losses and extra Zeeman phases omitted. "
                 "Input checkpoint remains in its upstream laboratory-z basis; Raman output m labels refer to the local field.")
    return result


def populations_from_unitary(states, ensemble, raman):
    """Apply a clock-pair SU(2) to the inherited 8-state mixture/coherence.

    states has shape (scan, atom, 2) and is the first column of each unitary.
    """
    states = np.asarray(states)
    coherent_probabilities = np.abs(states[:, :, 1]) ** 2
    probabilities = coherent_probabilities
    output_populations = None
    mode = resolved_state_handling(ensemble, raman)
    if mode == "inherit":
        rho = raman_frame_density(ensemble, raman)
        initial_pop = np.real(np.diagonal(rho, axis1=1, axis2=2)).copy()
        clock_e = (coherent_probabilities * initial_pop[:, G_CLOCK]
                   + (1 - coherent_probabilities) * initial_pop[:, E_CLOCK]
                   + 2 * np.real(states[:, :, 1] * rho[None, :, G_CLOCK, E_CLOCK] * states[:, :, 0]))
        pair_population = initial_pop[:, G_CLOCK] + initial_pop[:, E_CLOCK]
        if np.min(clock_e) < -1e-8 or np.max(clock_e - pair_population) > 1e-8:
            raise RuntimeError("Raman density propagation violated positivity")
        clock_e = np.clip(clock_e, 0, pair_population)
        output_populations = np.broadcast_to(initial_pop, states.shape[:2] + (8,)).copy()
        output_populations[:, :, G_CLOCK] = pair_population - clock_e
        output_populations[:, :, E_CLOCK] = clock_e
        probabilities = output_populations[:, :, GROUND_LABELS[:, 0] == 2].sum(axis=2)
    elif mode == "ideal_preparation" and not raman.remove_unprepared:
        eta = raman.preparation_efficiency
        probabilities = eta * probabilities + (1 - eta) * raman.spectator_F2_fraction
    elif mode == "project_F1_m0":
        output_populations = np.zeros(states.shape[:2] + (8,))
        output_populations[:, :, G_CLOCK] = 1 - coherent_probabilities
        output_populations[:, :, E_CLOCK] = coherent_probabilities
    return probabilities, output_populations


def run_simulation(config=None, ensemble=None, progress=None, prepared=None):
    prepared = prepared if prepared is not None else prepare_ensemble(config, ensemble, progress)
    config, ensemble = prepared["config"], prepared["ensemble"]
    detector_rng = np.random.default_rng(np.random.SeedSequence(config.seed).spawn(2)[1])
    state_mode = resolved_state_handling(ensemble, config.raman)
    state = np.zeros((len(ensemble.r_m), 2), dtype=complex)
    state[:, 0] = 1
    for zone in (0, 1):
        state = raman_zone(state, ensemble, config.raman, zone)
    phases = np.linspace(0, 2 * np.pi, config.raman.scan_points, endpoint=False)
    final_states = []
    norm_error = 0.0
    for phase in phases:
        final = raman_zone(state, ensemble, config.raman, 2, phase)
        final_states.append(final)
        norm_error = max(norm_error, float(np.max(np.abs(np.sum(np.abs(final) ** 2, axis=1) - 1))))
    final_states = np.asarray(final_states)
    coherent_probabilities = np.abs(final_states[:, :, 1]) ** 2
    probabilities, output_populations = populations_from_unitary(final_states, ensemble, config.raman)
    weights = preparation_weights(ensemble, config.raman)
    detection = fluorescence_scan(probabilities, ensemble, weights, config.detection, config.raman, detector_rng)
    return dict(prepared, **{
            "phase_rad": phases, "per_atom_P2": probabilities,
            "per_atom_coherent_P2": coherent_probabilities,
            "per_atom_ground_populations": output_populations,
            "state_handling": state_mode,
            "detection": detection, "fit": fit_fringe(phases, detection["observed_P2"]),
            "expected_fit": fit_fringe(phases, detection["expected_photon_P2"]),
            "max_state_norm_error": norm_error})
