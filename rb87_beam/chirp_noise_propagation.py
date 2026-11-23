"""Finite-width three-pulse propagation at one absolute two-photon chirp rate.

The SU(2) columns are sampled with the production midpoint propagator for
each chirp, zone and interpolation node. Interpolation is only over the
quasistatic clock frequency and common power within that optical zone.
"""

import copy
from math import factorial

import numpy as np

from .physics import K_EFF, raman_power_calibration, raman_peak_frequency
from .raman_noise import CLOCK_ZEEMAN_HZ_PER_G2, make_kernel


def _interpolation_axis(values, derivative_bound):
    """Partition the data and choose Chebyshev-Lobatto nodes in each interval.

    For a linear Hamiltonian parameter x, ||d^n U/dx^n|| <= L^n with
    L = integral ||d(H/hbar)/dx|| dt. The monic nodal polynomial at n
    Lobatto points on [-1,1] has norm <= 2^(2-n). The tile widths and
    orders below bound one-dimensional column interpolation error by 1e-9.
    """
    low, high = values.min(), values.max()
    if low == high or derivative_bound == 0.:
        return [(np.arange(len(values)), np.array([low]), np.ones((len(values), 1)))]
    intervals = max(1, int(np.ceil((high-low) * derivative_bound / 0.5)))
    edges = np.linspace(low, high, intervals + 1)
    tile = np.searchsorted(edges[1:-1], values, side="right")
    result = []
    for index in np.unique(tile):
        ids = np.flatnonzero(tile == index)
        lower, upper = edges[index:index+2]
        radius = (upper-lower)/2
        order = 2
        while (radius*derivative_bound)**order * 2.**(2-order) / factorial(order) > 1e-9:
            order += 1
        nodes = np.cos(np.pi*np.arange(order)/(order-1))
        coordinate = (values[ids]-lower)/radius - 1
        weights = np.ones((len(ids), order))
        for k in range(order):
            for j in range(order):
                if j != k:
                    weights[:, k] *= (coordinate-nodes[j])/(nodes[k]-nodes[j])
        physical_nodes = lower + (nodes+1)*radius
        physical_nodes[0], physical_nodes[-1] = upper, lower
        result.append((ids, physical_nodes, weights))
    return result


def _interaction_bounds(kernel, zone):
    """Bounds on SU(2) generator derivatives for power and clock shift (Hz)."""
    raman = kernel.raman
    speed = np.min(kernel.ensemble.v_m_s[:, 2])
    if raman.envelope == "gaussian":
        extent = raman.envelope_half_widths
        span = 2 * extent * raman.waist_z_m
        u = -extent + (np.arange(raman.integration_steps)+0.5) * 2*extent/raman.integration_steps
        integrated_time = span/speed * np.mean(np.exp(-2*u*u))
    else:
        span = raman.rectangular_width_m
        integrated_time = span/speed
    factor = (1., 2., 1.)[zone]
    peak = factor * raman_peak_frequency(raman)
    shift = raman.differential_light_shift_Hz
    calibration = raman_power_calibration(raman, factor)
    if calibration is not None:
        peak = calibration["peak_rabi_Hz"]
        shift = calibration["differential_light_shift_Hz"] + factor * shift
    # The generator has eigenvalues +/-sqrt(Omega^2 + delta^2)/2.
    return np.pi * np.hypot(peak, shift) * integrated_time, np.pi * span/speed


def _zone_columns(kernel, zone, power, clock):
    power_bound, field_bound = _interaction_bounds(kernel, zone)
    p_tiles = _interpolation_axis(power, power_bound)
    f_tiles = _interpolation_axis(clock, field_bound)
    n_atoms = len(kernel.pg)
    columns = np.empty((len(power), 2, n_atoms), complex)
    for p_ids, p_nodes, p_weights in p_tiles:
        for f_ids, f_nodes, f_weights in f_tiles:
            ids, ip, jf = np.intersect1d(p_ids, f_ids, assume_unique=True, return_indices=True)
            if not len(ids):
                continue
            samples = np.stack([kernel._column(zone, p, f).ravel() for p in p_nodes for f in f_nodes])
            weights = (p_weights[ip, :, None] * f_weights[jf, None, :]).reshape(len(ids), -1)
            columns[ids] = (weights @ samples).reshape(len(ids), 2, n_atoms)
    columns /= np.sqrt(np.sum(np.abs(columns)**2, axis=1, keepdims=True))
    return columns[:, 0], columns[:, 1]


def chirp_probabilities(ensemble, raman, alpha_Hz_s, draws, bias_G, *, check_indices=()):
    """Return nominal P2, noisy P2[record, atom] and direct-check maximum error.

    The configured phase offsets stay fixed. Record-wise phase and mirror motion
    enter the coupling at each pulse; the quadratic Zeeman term acts over both
    pulses and free flight. check_indices selects rows for direct reintegration.
    """
    if not np.isfinite(alpha_Hz_s) or not np.isfinite(bias_G):
        raise ValueError("Chirp rate and magnetic bias must be finite")
    phase, mirror, field, power = [
        np.asarray(draws[key], dtype=float) for key in ("phase", "vibration", "magnetic", "intensity")]
    if (phase.ndim != 2 or phase.shape[1] != 3 or len(phase) == 0
            or any(v.shape != phase.shape or not np.isfinite(v).all() for v in (phase, mirror, field, power))
            or np.any(power < 0)):
        raise ValueError("Each disturbance needs finite (records, 3) values and nonnegative power")
    checked = list(check_indices)
    if any(not isinstance(i, (int, np.integer)) or i < 0 or i >= len(phase) for i in checked):
        raise ValueError("Invalid direct-check record index")
    config = copy.copy(raman)
    config.chirp_rad_s2 = 2*np.pi*alpha_Hz_s
    kernel = make_kernel(ensemble, config)
    zero = np.zeros(3)
    baseline = kernel.harmonics(zero, zero, zero, np.ones(3), bias_G, exact=True)
    nominal = baseline[0] + baseline[1]
    optical_phase = phase - K_EFF * mirror
    clock = CLOCK_ZEEMAN_HZ_PER_G2 * field * (2*bias_G + field)
    pg, pe, coherence = kernel.pg, kernel.pe, kernel.coherence
    for zone in range(3):
        a, b = _zone_columns(kernel, zone, power[:, zone], clock[:, zone])
        b *= np.exp(1j * optical_phase[:, zone, None])
        new_pe = abs(b)**2 * pg + abs(a)**2 * pe + 2*(b * a * coherence).real
        if zone < 2:
            coherence = a*b.conj()*(pg-pe) + a**2*coherence - b.conj()**2*coherence.conj()
            dark = np.pi*(clock[:, zone] + clock[:, zone+1])[:, None] * kernel.dark_time_s
            coherence *= np.exp(1j*dark)
        pg, pe = pg+pe-new_pe, new_pe
    noisy = kernel.spectator_e + pe

    max_error = 0.
    for i in checked:
        exact = kernel.harmonics(phase[i], mirror[i], field[i], power[i], bias_G, exact=True)
        max_error = max(max_error, float(np.max(abs(noisy[i] - (exact[0]+exact[1])))))
    return nominal, noisy, max_error
