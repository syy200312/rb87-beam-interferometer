"""Rb87 D2 conversion between optical powers and effective Raman coupling.

Powers are those of two individual frequency/direction components *at the
atoms*. Beam 1 has the higher optical frequency, propagates along +x and
addresses F=1; beam 2 propagates along -x and addresses F=2. Both polarization
labels refer to the SAME +x quantization axis, so equal sigma labels have
opposite helicity relative to their respective propagation directions.

This calibration assumes adiabatic transport of the prepared F,m populations
from the pumping z axis to the Raman x axis, without crossing zero field.
Only zero-field hyperfine energies enter the denominators. A weak guide field
defines the axis; Zeeman corrections, nonadiabatic spin rotation, extra laser
sidebands, and Raman scattering trajectories are not simulated here.

Steck, https://steck.us/alkalidata/rubidium87numbers.pdf, Eqs.34-39 and Table 7:
the CG-normalized reduced dipole is dS=3.58424e-29 C m, NOT the standard 3j
reduced dipole (sqrt(2)*dS). PyLCP's normalized D2 cycling amplitude is one,
so its dipole tensor is multiplied by d_cycling=dS/sqrt(2).
The shared PyLCP Hamiltonian supplies a consistent set of hyperfine energies.

Coherent intermediate-state summation and Omega_1*Omega_2/(2*Delta):
https://arxiv.org/pdf/1605.05230, Sec.III A (its endpoint states differ from
the m=0 states here; its numerical CG factors are NOT imported).
Clock-state polarization paths: Yunfan Wu (2021), Appendix A,
https://web.stanford.edu/group/kasevich/cgi-bin/wordpress/wp-content/uploads/2022/04/Y-WU_2021_PhD_Thesis.pdf

Convention: H_eff off-diagonal = hbar*Omega_eff/2, positive detuning is blue,
and I=epsilon_0*c*|E|^2/2. Computation uses rad/s; returned frequencies are Hz.
All light shifts and total spontaneous-rate diagnostics include both optical
frequencies on ALL eight ground states, including the off-resonant wrong legs.
The spontaneous rates are diagnostics, not a dissipative Raman propagator.
"""

import copy
from functools import lru_cache

import numpy as np
from scipy.constants import c as LIGHT_SPEED, epsilon_0

from .physics import GAMMA, HBAR
from .states import GROUND_LABELS, EXCITED_LABELS, G_CLOCK, E_CLOCK


STECK_D2_CG_REDUCED_CM = 3.58424e-29
D2_CYCLING_DIPOLE_CM = STECK_D2_CG_REDUCED_CM / np.sqrt(2.)
MIN_DETUNING_LINEWIDTHS = 20.
MAX_SINGLE_PHOTON_RABI_RATIO = .1
_POLARIZATIONS = {
    "sigma_minus": (1., 0., 0.),
    "sigma_plus": (0., 0., 1.),
    # Equal q amplitudes: one fixed transverse linear polarization. A common
    # overall phase is immaterial. Two equal linear beams cancel at mF=0.
    "linear": (1 / np.sqrt(2.), 0., 1 / np.sqrt(2.)),
}


@lru_cache(maxsize=1)
def _d2_cached():
    # Lazy import avoids a dependency cycle when physics imports calibration.
    from .pylcp_atom import d2_atom
    _, lowering, _, ground, excited = d2_atom()
    if np.shape(lowering) != (3, 8, 16):
        raise ValueError("Unexpected PyLCP dipole shape; expected q,ground,excited")
    absorption = np.empty((3, 16, 8), complex)
    for index, q in enumerate((-1, 0, 1)):
        absorption[index] = (-1.) ** q * np.conj(lowering[2-index]).T * D2_CYCLING_DIPOLE_CM
    ground_Hz = np.asarray(ground, float) * GAMMA / (2 * np.pi)
    excited_Hz = np.asarray(excited, float) * GAMMA / (2 * np.pi)
    for value in (absorption, ground_Hz, excited_Hz):
        value.setflags(write=False)
    return absorption, ground_Hz, excited_Hz


def d2_dipoles_SI():
    """Return absorption dq[3,16,8], ground_Hz[8], excited_Hz[16].

    q order is -1,0,+1; basis order is states.GROUND_LABELS/EXCITED_LABELS.
    Arrays are independent writable copies so callers cannot corrupt caches.
    d_eg^q=(-1)^q*conj(d_ge^-q) retains the phase needed for Raman interference.
    """
    return tuple(value.copy() for value in _d2_cached())


def _number(value, name, positive=False, nonnegative=False):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or np.iscomplexobj(value):
        raise ValueError(name + " must be a finite real scalar")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(name + " must be a finite real scalar")
    if not np.isfinite(number) or (positive and number <= 0) or (nonnegative and number < 0):
        limit = "positive" if positive else "nonnegative" if nonnegative else "finite"
        raise ValueError(name + " must be " + limit + " and finite")
    return number


def _settings(waist_y_m, waist_z_m, detuning_Hz, reference_Fprime,
              polarization1, polarization2, two_photon_detuning_Hz):
    wy = _number(waist_y_m, "waist_y_m", positive=True)
    wz = _number(waist_z_m, "waist_z_m", positive=True)
    delta = _number(detuning_Hz, "detuning_Hz")
    two = _number(two_photon_detuning_Hz, "two_photon_detuning_Hz")
    if isinstance(reference_Fprime, (bool, np.bool_)) or not isinstance(reference_Fprime, (int, np.integer)) or reference_Fprime not in (0, 1, 2, 3):
        raise ValueError("reference_Fprime must be an integer from 0 to 3")
    for polarization in (polarization1, polarization2):
        if not isinstance(polarization, str) or polarization not in _POLARIZATIONS:
            raise ValueError("Polarization must be sigma_plus, sigma_minus or linear about the common +x axis")
    return wy, wz, delta, int(reference_Fprime), polarization1, polarization2, two


def _geometry(settings):
    wy, wz, delta, reference, pol1, pol2, two = settings
    dipole, ground, excited = _d2_cached()
    reference_energy = excited[np.flatnonzero(EXCITED_LABELS[:, 0] == reference)[0]]
    hyperfine = ground[E_CLOCK] - ground[G_CLOCK]
    # Optical frequencies relative to the D2 fine-structure centroid. This
    # reference cancels when subtracting each actual optical transition.
    high = reference_energy - ground[G_CLOCK] + delta
    low = high - hyperfine - two
    transitions = excited[:, None] - ground[None, :]
    detunings = 2 * np.pi * (np.array([high, low])[:, None, None] - transitions[None, :, :])
    if not np.all(np.isfinite(detunings)):
        raise ValueError("Optical detunings are outside the representable range")
    matrix = np.stack([np.einsum("q,qeg->eg", _POLARIZATIONS[pol], dipole)
                       for pol in (pol1, pol2)])
    area = np.pi * wy * wz / 2
    if not np.isfinite(area) or area <= 0:
        raise ValueError("Gaussian beam area must be positive and representable")
    return matrix, detunings, area, float(hyperfine), (float(high), float(low))


def _calculate(powers, settings, enforce=True):
    matrix, detunings, area, hyperfine, optical = _geometry(settings)
    powers = np.asarray(powers, float)
    intensities = powers / area
    if not np.all(np.isfinite(intensities)):
        raise ValueError("Peak intensities are outside the representable range")
    fields = np.sqrt(2 * intensities / (epsilon_0 * LIGHT_SPEED))
    omega = matrix * fields[:, None, None] / HBAR
    active = (np.abs(matrix) > 0) & (powers[:, None, None] > 0)
    possible = np.abs(matrix) > 0
    selected = active if np.any(active) else possible
    minimum = float(np.min(np.abs(detunings[selected])) / (2 * np.pi))
    if enforce and np.any(active) and minimum < MIN_DETUNING_LINEWIDTHS * GAMMA / (2 * np.pi):
        raise ValueError("Far-detuned Raman model requires every illuminated allowed transition to have |Delta| >= 20 Gamma (121.3 MHz)")
    inverse = np.divide(1., detunings, out=np.zeros_like(detunings), where=detunings != 0)
    ratios = np.abs(omega * inverse)
    ratio = float(np.max(ratios))
    if enforce and (not np.isfinite(ratio) or ratio > MAX_SINGLE_PHOTON_RABI_RATIO):
        raise ValueError("Far-detuned Raman model requires max single-photon |Omega/Delta| <= 0.1")
    squared = np.abs(omega) ** 2
    shifts = np.sum(squared * inverse / 4, axis=(0, 1))
    scattering = GAMMA * np.sum(squared * inverse ** 2 / 4, axis=(0, 1))
    paths = (omega[0, :, G_CLOCK] * np.conj(omega[1, :, E_CLOCK]) / 4 *
             (inverse[0, :, G_CLOCK] + inverse[1, :, E_CLOCK]))
    coupling = np.sum(paths)
    # Avoid assigning an arbitrary pi phase to roundoff left by cancellation.
    scale = np.sum(np.abs(paths))
    if abs(coupling) <= 64 * np.finfo(float).eps * scale:
        coupling = 0j
    all_values = np.r_[shifts, scattering, intensities, coupling.real, coupling.imag]
    if not np.all(np.isfinite(all_values)):
        raise ValueError("Calibration produced nonfinite results; check powers, waists and detunings")
    contributions = []
    for f in range(4):
        value = np.sum(paths[EXCITED_LABELS[:, 0] == f]) / (2 * np.pi)
        contributions.append(dict(Fprime=f, real_Hz=float(value.real), imag_Hz=float(value.imag),
                                  magnitude_Hz=float(abs(value))))
    return dict(power1_W=float(powers[0]), power2_W=float(powers[1]),
        peak_rabi_Hz=float(abs(coupling) / (2 * np.pi)),
        coupling_phase_rad=float(np.angle(coupling)) if coupling != 0 else 0.,
        coupling_real_Hz=float(coupling.real / (2 * np.pi)),
        coupling_imag_Hz=float(coupling.imag / (2 * np.pi)),
        differential_light_shift_Hz=float((shifts[E_CLOCK] - shifts[G_CLOCK]) / (2 * np.pi)),
        ground_light_shifts_Hz=(shifts / (2 * np.pi)).tolist(),
        ground_scattering_rates_per_s=scattering.tolist(),
        path_contributions=contributions, peak_intensities_W_m2=intensities.tolist(),
        min_abs_detuning_Hz=minimum, max_single_photon_rabi_to_detuning_ratio=ratio,
        hyperfine_ground_splitting_Hz=hyperfine,
        optical_frequency_offsets_Hz=list(optical),
        detuning_Hz=settings[2], reference_Fprime=settings[3],
        polarization1=settings[4], polarization2=settings[5],
        two_photon_detuning_Hz=settings[6], quantization_axis="+x",
        state_transport_assumption="adiabatic z-to-x guide-field rotation, preserving F,m populations",
        approximation="zero-field D2 hyperfine energies; far-detuned coherent Raman elimination; scattering rates are diagnostics only")


@lru_cache(maxsize=4096)
def _calibrate_cached(power1_W, power2_W, settings):
    return _calculate((power1_W, power2_W), settings)


def calibrate_powers(power1_W, power2_W, waist_y_m, waist_z_m, detuning_Hz=1.073e9,
                     reference_Fprime=1, polarization1="sigma_plus", polarization2="sigma_plus",
                     two_photon_detuning_Hz=0.):
    """Convert two Gaussian-beam powers into peak Raman/AC-shift/rate data.

    ``detuning_Hz`` is high-frequency beam 1 minus the F=1 -> reference_Fprime
    zero-field line; beam 2 is lower by ground_hyperfine + two_photon_detuning.
    Waists are 1/e^2 *intensity* radii. All returned data are JSON serializable.
    Every nonzero-power allowed optical leg must be at least 20 linewidths
    detuned, with |single-photon Omega/Delta| <= .1. Zero powers are permitted.
    """
    p1 = _number(power1_W, "power1_W", nonnegative=True)
    p2 = _number(power2_W, "power2_W", nonnegative=True)
    settings = _settings(waist_y_m, waist_z_m, detuning_Hz, reference_Fprime,
                         polarization1, polarization2, two_photon_detuning_Hz)
    # Callers may annotate their result without altering later cached results.
    return copy.deepcopy(_calibrate_cached(p1, p2, settings))


def solve_powers(target_rabi_Hz, waist_y_m, waist_z_m, detuning_Hz=1.073e9,
                 reference_Fprime=1, polarization1="sigma_plus", polarization2="sigma_plus",
                 two_photon_detuning_Hz=0., power_ratio="cancel_light_shift"):
    """Find powers for a nonnegative target Omega_eff/(2*pi).

    A positive numeric power_ratio specifies P_high/P_low. The default solves
    the ratio that cancels the differential AC shift of the two m=0 states;
    this does not cancel their common-mode light shift or spontaneous rates.
    A nonzero target with coherently cancelling polarizations is rejected.
    """
    target = _number(target_rabi_Hz, "target_rabi_Hz", nonnegative=True)
    settings = _settings(waist_y_m, waist_z_m, detuning_Hz, reference_Fprime,
                         polarization1, polarization2, two_photon_detuning_Hz)
    # Work with analytic low-power coefficients, then validate the final
    # physical powers. No arbitrary unit-power field should fail elimination.
    probe = 1e-12
    one = _calculate((probe, 0.), settings, enforce=False)
    two = _calculate((0., probe), settings, enforce=False)
    cancel_shift = isinstance(power_ratio, str) and power_ratio == "cancel_light_shift"
    if cancel_shift:
        a, b = one["differential_light_shift_Hz"], two["differential_light_shift_Hz"]
        if a == 0 or b == 0 or np.sign(a) == np.sign(b):
            raise ValueError("No unique positive power ratio cancels the differential light shift for this configuration")
        ratio = -b / a
    else:
        ratio = _number(power_ratio, "power_ratio", positive=True)
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError("Light-shift cancellation did not yield a positive finite power ratio")
    unit = _calculate((probe, probe), settings, enforce=False)
    coefficient = unit["peak_rabi_Hz"] / probe
    if target > 0 and coefficient <= 0:
        raise ValueError("These polarization paths cancel the clock-state Raman coupling; no finite powers produce a nonzero target")
    p2 = target / (coefficient * np.sqrt(ratio)) if target > 0 else 0.
    p1 = ratio * p2
    result = calibrate_powers(p1, p2, *settings)
    result.update(power_ratio=float(ratio), target_rabi_Hz=target,
        power_ratio_rule="cancel_light_shift" if cancel_shift else "supplied high/low power ratio")
    return result
