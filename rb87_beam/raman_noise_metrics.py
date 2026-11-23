"""Two-channel fluorescence statistics for independent noisy three-pulse records.

All atoms in one record see the same technical realization. Averaging fringe
coefficients over records corresponds to averaging measurements at fixed phase.
The spread of a single sweep is evaluated with a fresh gate at each phase.
"""

from dataclasses import dataclass

import numpy as np

from .physics import fit_fringe, fluorescence_scan, preparation_weights
from .states import G_CLOCK, raman_frame_density, resolved_state_handling


@dataclass
class NoiseDetector:
    atom_counts: np.ndarray
    q1: np.ndarray
    q2: np.ndarray
    gain1: float
    gain2: float
    gate_s: float
    background: float
    electronic_std: float
    accepted_flux_per_s: float
    target_flux_per_s: float

    def coefficients(self, h):
        """Per-record normalized fluorescence coefficients, weighted by photon yield."""
        # q1/gain1 and q2/gain2 have identical 1/vz dependence. Therefore the
        # mean gain-corrected denominator is independent of state and phase.
        weights = self.atom_counts * self.q2 / self.gain2
        return np.einsum("...ca,a->...c", h, weights / weights.sum())

    def readout(self, h, phase_rad):
        """Mean-signal normalized P2 for each record at one phase, before counting noise."""
        return self.coefficients(h) @ np.array([1., np.cos(phase_rad), np.sin(phase_rad)])

    def moments(self, h, phase_rad):
        """Conditional P2 and delta-method variance for independent compound-Poisson channels."""
        harmonic = np.array([1., np.cos(phase_rad), np.sin(phase_rad)])
        p = np.einsum("...ca,c->...a", h, harmonic)
        y = self.readout(h, phase_rad)
        floor = self.background + self.electronic_std ** 2
        # E[N]=lambda, Var[sum of N Poisson(q) photon yields]=lambda*(q+q^2).
        var1 = (1 - p) @ (self.atom_counts * self.q1 * (1 + self.q1)) + floor
        var2 = p @ (self.atom_counts * self.q2 * (1 + self.q2)) + floor
        scale = self.atom_counts.sum()
        var = (y ** 2 * var1 / self.gain1 ** 2 + (1 - y) ** 2 * var2 / self.gain2 ** 2) / scale ** 2
        return y, var


def make_detector(ensemble, config):
    """Freeze aperture, preparation, velocity-dependent photon yields, and physical fluxes."""
    d, r = config.detection, config.raman
    # Use the production readout's geometry and gain calibration. Empty phase
    # input requests those quantities without drawing a measurement.
    info = fluorescence_scan((), ensemble, preparation_weights(ensemble, r), d, r, None)
    weights = info["detection_weights"]
    atom_counts = ensemble.flux_per_s * d.gate_s / len(weights) * weights
    mode = resolved_state_handling(ensemble, r)
    if mode == "inherit":
        target = raman_frame_density(ensemble, r)[:, G_CLOCK, G_CLOCK].real
    elif mode == "ideal_preparation" and not r.remove_unprepared:
        target = r.preparation_efficiency
    else:
        # The state-selection factor was already included in preparation_weights.
        target = 1.
    gain1 = info["gain_F1_counts_per_atom"]
    gain2 = info["gain_F2_counts_per_atom"]
    inverse_v = 1 / ensemble.v_m_s[:, 2]
    relative_yield = inverse_v / np.average(inverse_v, weights=weights)
    return NoiseDetector(atom_counts, relative_yield * gain1, relative_yield * gain2,
                         gain1, gain2, d.gate_s, d.background_counts, d.electronic_noise_std_counts,
                         info["detected_flux_per_s"], float(np.sum(atom_counts * target) / d.gate_s))


def _contrast(coeff):
    return np.hypot(coeff[..., 1], coeff[..., 2]) / coeff[..., 0]


def _jackknife_se(delete_one):
    d = np.asarray(delete_one)
    if not np.all(np.isfinite(d)):
        return None
    # Center relative to one sample to give exactly zero for identical records.
    delta = d - d[0]
    return float(np.sqrt((len(d) - 1) * np.mean((delta - np.mean(delta)) ** 2)))


def _scan_draws(h, detector, phases, scans, rng):
    """Resample a joint whole-ensemble technical realization for each gate, then draw counts."""
    design = np.column_stack((np.ones(len(phases)), np.cos(phases), np.sin(phases)))
    scan_data = np.full((scans, len(phases)), np.nan)
    chunk = max(1, min(32, 600_000 // (len(phases) * h.shape[-1])))
    for start in range(0, scans, chunk):
        count = min(chunk, scans - start)
        ids = rng.integers(0, len(h), (count, len(phases)))
        p = np.einsum("rpca,pc->rpa", h[ids], design)
        p = np.clip(p, 0, 1)  # only roundoff after full harmonic bound validation

        def channel(probability, q, gain):
            n_atoms = rng.poisson(detector.atom_counts * probability)
            photons = rng.poisson(np.einsum("rpa,a->rp", n_atoms, q) + detector.background)
            net = (photons - detector.background
                   + rng.normal(0., detector.electronic_std, (count, len(phases))))
            return net / gain

        f1 = channel(1 - p, detector.q1, detector.gain1)
        f2 = channel(p, detector.q2, detector.gain2)
        np.divide(f2, f1 + f2, where=(f1 + f2 > 0), out=scan_data[start:start + count])
    return scan_data


def _optional_list(values):
    return [float(x) if np.isfinite(x) else None for x in values]


def summarize_noise(h, detector, operating_phase_rad, rotation_slope, *, phases=None,
                    scans=256, seed=87, cycle_s=None):
    """Return scalar metrics and fringe arrays conditional on this fixed atom sample.

    h has shape (independent records, offset/cos/sin, trajectories). A vector
    rotation_slope is the per-record paired derivative at the same fixed phase.
    Monte Carlo standard errors delete whole independently propagated records.
    The one-second equivalent assumes iid gates separated by cycle_s; the
    exact repeated-sweep count draws also support visibly non-Gaussian ratios.
    """
    h = np.asarray(h, dtype=float)
    if (h.ndim != 3 or h.shape[1:] != (3, len(detector.atom_counts)) or len(h) < 2
            or not np.isfinite(h).all()):
        raise ValueError("Expected >=2 finite records with offset/cos/sin for each trajectory")
    amplitude = np.hypot(h[:, 1], h[:, 2])
    if np.any(h[:, 0] - amplitude < -1e-8) or np.any(h[:, 0] + amplitude > 1 + 1e-8):
        raise ValueError("Harmonics give per-atom probabilities outside [0,1]")
    if not isinstance(scans, (int, np.integer)) or isinstance(scans, bool) or scans < 1:
        raise ValueError("scans must be a positive integer")
    cycle = detector.gate_s if cycle_s is None else float(cycle_s)
    if not np.isfinite(cycle) or cycle < detector.gate_s or not np.isfinite(operating_phase_rad):
        raise ValueError("Require finite operating phase and independent cycle >= gate duration")
    phases = np.linspace(0, 2 * np.pi, 41, endpoint=False) if phases is None else np.asarray(phases, dtype=float)
    if phases.ndim != 1 or len(phases) < 4 or not np.isfinite(phases).all():
        raise ValueError("Provide at least four finite scan phases")
    design = np.column_stack((np.ones(len(phases)), np.cos(phases), np.sin(phases)))
    if np.linalg.matrix_rank(design) < 3:
        raise ValueError("Scan phases must resolve offset, sine and cosine")
    n = len(h)
    slopes = np.broadcast_to(np.asarray(rotation_slope, dtype=float), (n,))
    if not np.isfinite(slopes).all():
        raise ValueError("Rotation slope must be finite")
    coeff = detector.coefficients(h)
    center = coeff.mean(axis=0)
    contrast = float(_contrast(center)) if center[0] > 1e-12 else None
    loo_coeff = center + (center - coeff) / (n - 1)
    contrast_mc = _jackknife_se(_contrast(loo_coeff)) if np.min(loo_coeff[:, 0]) > 1e-12 else None

    mu, conditional_var = detector.moments(h, operating_phase_rad)
    # Use the law of total variance, conditioning on one technical realization.
    deviations = mu - mu[0]
    deviations -= deviations.mean()
    ss = float(np.sum(deviations ** 2))
    vtech = ss / (n - 1)
    vdet = float(np.mean(conditional_var))
    slope = float(np.mean(slopes))
    sigma_gate = np.sqrt(vdet + vtech) / abs(slope) if slope != 0 else None
    allan = float(sigma_gate * np.sqrt(cycle)) if sigma_gate is not None else None
    allan_mc = None
    loo_slope = slope + (slope - slopes) / (n - 1)
    if n > 2 and np.all(loo_slope != 0):
        # Unbiased sample variance after deleting record i:
        # (SS - n/(n-1)*(x_i - mean)^2)/(n-2).
        loo_vtech = np.maximum(0., ss - n / (n - 1) * deviations ** 2) / (n - 2)
        loo_vdet = vdet + (vdet - conditional_var) / (n - 1)
        allan_mc = _jackknife_se(np.sqrt((loo_vtech + loo_vdet) * cycle) / np.abs(loo_slope))

    observed = _scan_draws(h, detector, phases, scans, np.random.default_rng(seed))
    finite = np.isfinite(observed).sum(axis=0)
    mean_observed = np.divide(np.nansum(observed, axis=0), finite,
                              where=finite > 0, out=np.full(len(phases), np.nan))

    def fitted_contrast(values):
        try:
            return fit_fringe(phases, values)["contrast"]
        except ValueError:  # fewer than four usable phases or rank deficiency
            return None

    draws = [c for c in (fitted_contrast(line) for line in observed) if c is not None]
    quantiles = np.quantile(draws, [0.025, 0.5, 0.975]).tolist() if draws else [None] * 3
    metrics = dict(
        records=n, scans=int(scans), successful_fits=len(draws),
        complete_scans=int(np.all(np.isfinite(observed), axis=1).sum()),
        valid_gate_fraction=float(np.mean(np.isfinite(observed))),
        accepted_flux_per_s=float(detector.accepted_flux_per_s), target_flux_per_s=detector.target_flux_per_s,
        expected_atoms_per_gate=float(detector.atom_counts.sum()), gate_s=detector.gate_s, cycle_s=cycle,
        contrast=contrast, contrast_mc_se=contrast_mc, mean_observed_contrast=fitted_contrast(mean_observed),
        single_scan_contrast_q025=quantiles[0], single_scan_contrast_q50=quantiles[1],
        single_scan_contrast_q975=quantiles[2], operating_phase_rad=float(operating_phase_rad),
        mean_operating_P2=float(np.mean(mu)), dP2_dOmega=slope,
        detection_variance_P2=vdet, technical_variance_P2=vtech,
        sigma_rotation_gate_rad_s=None if sigma_gate is None else float(sigma_gate),
        allan_1s_rad_s=allan, allan_1s_mc_se_rad_s=allan_mc,
        arw_deg_per_sqrt_h=None if allan is None else float(allan * 180 / np.pi * 60))
    fringe = dict(phase_rad=phases.tolist(), mean_expected_P2=(design @ center).tolist(),
                  mean_observed_P2=_optional_list(mean_observed), example_observed_P2=_optional_list(observed[0]))
    return metrics, fringe
