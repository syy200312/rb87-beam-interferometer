"""Three finite Raman pulses with spatially piecewise-constant technical errors.

Inputs are common to the atoms in a single record. The caller supplies their
spatial and temporal statistics. An extra phase on pulse 3 remains a free scan
coordinate, so each record yields the entire per-atom sinusoidal response.
"""

import copy
from functools import lru_cache

import numpy as np

from .physics import K_EFF, raman_power_calibration, raman_zone
from .raman_calibration import MAX_SINGLE_PHOTON_RABI_RATIO
from .states import E_CLOCK, G_CLOCK, GROUND_LABELS, raman_frame_density, resolved_state_handling


# Steck, Rubidium 87 D Line Data, clock transition Zeeman shift, https://steck.us/alkalidata/rubidium87numbers.pdf
CLOCK_ZEEMAN_HZ_PER_G2 = 575.15


def _cubic_nodes(coordinate, lower_limit=None):
    """Local four-point Lagrange stencil in units of the interpolation spacing."""
    nearest = round(coordinate)
    if abs(coordinate - nearest) < 1e-12:
        return [(nearest, 1.0)]
    left = int(np.floor(coordinate)) - 1
    if lower_limit is not None:
        left = max(lower_limit, left)
    x = coordinate - left
    return [(left, -(x - 1) * (x - 2) * (x - 3) / 6),
            (left + 1, x * (x - 2) * (x - 3) / 2),
            (left + 2, -x * (x - 1) * (x - 3) / 2),
            (left + 3, x * (x - 1) * (x - 2) / 6)]


def _three_values(value, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(name + " must be a finite three-zone vector")
    return array


class _RamanNoiseKernel:
    def __init__(self, ensemble, raman):
        self.ensemble = ensemble
        self.raman = raman
        n = len(ensemble.r_m)
        self.initial_column = np.zeros((n, 2), complex)
        self.initial_column[:, 0] = 1.
        mode = resolved_state_handling(ensemble, raman)
        self.pg = np.ones(n)
        self.pe = np.zeros(n)
        self.coherence = np.zeros(n, complex)
        self.spectator_e = np.zeros(n)
        if mode == "inherit":
            rho = raman_frame_density(ensemble, raman)
            self.pg = rho[:, G_CLOCK, G_CLOCK].real.copy()
            self.pe = rho[:, E_CLOCK, E_CLOCK].real.copy()
            self.coherence = rho[:, G_CLOCK, E_CLOCK].copy()
            spectator = (GROUND_LABELS[:, 0] == 2) & (np.arange(8) != E_CLOCK)
            self.spectator_e = np.diagonal(rho, axis1=1, axis2=2).real[:, spectator].sum(axis=1)
        elif mode == "ideal_preparation" and not raman.remove_unprepared:
            self.pg[:] = raman.preparation_efficiency
            self.spectator_e[:] = (1 - raman.preparation_efficiency) * raman.spectator_F2_fraction

        span = (raman.waist_z_m * raman.envelope_half_widths if raman.envelope == "gaussian"
                else raman.rectangular_width_m / 2)
        self.dark_time_s = (raman.zone_spacing_m - 2 * span) / ensemble.v_m_s[:, 2]
        if np.any(self.dark_time_s <= 0):
            raise ValueError("Raman envelopes overlap or longitudinal speed is nonpositive")
        # Magnetic coordinates are scaled for the slowest trajectory. The node
        # spacing gives a maximal pulse phase increment of 0.02 rad.
        max_duration = 2 * span / np.min(ensemble.v_m_s[:, 2])
        self.field_step_Hz = 0.02 / (2 * np.pi * max_duration)
        self.power_step = 0.01
        # Preserve the physical domain of the underlying power calibration even
        # near the edge of the interpolation mesh.
        self.valid_power = [np.inf] * 3
        for zone, factor in enumerate((1., 2., 1.)):
            calibration = raman_power_calibration(raman, factor)
            if calibration and calibration["max_single_photon_rabi_to_detuning_ratio"] > 0:
                ratio = calibration["max_single_photon_rabi_to_detuning_ratio"]
                self.valid_power[zone] = (MAX_SINGLE_PHOTON_RABI_RATIO / ratio) ** 2
        self.grid_column = lru_cache(maxsize=512)(self._calculate_node)

    def _column(self, zone, power, clock_Hz):
        return raman_zone(self.initial_column, self.ensemble, self.raman, zone,
                          amplitude_scale=power, clock_shift_Hz=clock_Hz).T

    def _calculate_node(self, zone, power_node, field_node):
        return self._column(zone, power_node * self.power_step, field_node * self.field_step_Hz)

    def _interpolated(self, zone, power, clock_Hz):
        p_nodes = _cubic_nodes(power / self.power_step, lower_limit=0)
        f_nodes = _cubic_nodes(clock_Hz / self.field_step_Hz)
        if p_nodes[-1][0] * self.power_step > self.valid_power[zone]:
            return self._column(zone, power, clock_Hz)
        if len(p_nodes) == len(f_nodes) == 1:
            return self.grid_column(zone, p_nodes[0][0], f_nodes[0][0])
        column = np.zeros((2, len(self.pg)), complex)
        for p, wp in p_nodes:
            for f, wf in f_nodes:
                column += (wp * wf) * self.grid_column(zone, p, f)
        # An SU(2) matrix is fixed by its unit-length first column (a, b).
        return column / np.sqrt(np.sum(np.abs(column) ** 2, axis=0))

    def harmonics(self, phase_rad, displacement_m, bfield_G, power_factors, bias_G, *, exact=False):
        """Return offset, cosine and sine coefficients (3, atoms) of P(F=2).

        Each length-three input follows the atoms from zone 1 to 3. A positive
        reference-optic displacement along k_eff subtracts k_eff * displacement
        from laser phase. Both Raman colors scale by the same power factor.
        bfield_G is the deviation from the compensated longitudinal bias bias_G.
        Each B value covers its pulse and the adjacent half of each dark gap.
        Clock coherences at pulse 1 start from the supplied preparation matrix.
        exact=True directly integrates every requested pulse for cross-checks.
        """
        phase = _three_values(phase_rad, "phase_rad") - K_EFF * _three_values(displacement_m, "displacement_m")
        field = _three_values(bfield_G, "bfield_G")
        power = _three_values(power_factors, "power_factors")
        if np.any(power < 0) or np.ndim(bias_G) != 0 or not np.isfinite(bias_G):
            raise ValueError("Powers must be nonnegative and bias_G a finite scalar")
        clock = CLOCK_ZEEMAN_HZ_PER_G2 * field * (2 * bias_G + field)
        pg, pe, coherence = self.pg, self.pe, self.coherence
        propagator = self._column if exact else self._interpolated
        for zone in range(3):
            a, bare_b = propagator(zone, power[zone], clock[zone])
            b = bare_b * np.exp(1j * phase[zone])
            e_offset = np.abs(b)**2 * pg + np.abs(a)**2 * pe
            interference = 2 * b * a * coherence
            if zone == 2:
                return np.array([self.spectator_e + e_offset, interference.real, -interference.imag])
            new_coherence = a * b.conj() * (pg - pe) + a**2 * coherence - b.conj()**2 * coherence.conj()
            new_pe = e_offset + interference.real
            pg, pe = pg + pe - new_pe, new_pe
            # Real level shift gives rho_ge -> rho_ge * exp(+i integral delta_eg dt).
            dark_phase = np.pi * (clock[zone] + clock[zone + 1]) * self.dark_time_s
            coherence = new_coherence * np.exp(1j * dark_phase)


def make_kernel(ensemble, raman, rotation_delta_y=0.0):
    """Precompute a reusable three-pulse kernel at a specified rotation offset."""
    if not np.isfinite(rotation_delta_y):
        raise ValueError("rotation_delta_y must be finite")
    config = copy.deepcopy(raman)
    rotation = np.asarray(config.rotation_rad_s, dtype=float).copy()
    rotation[1] += rotation_delta_y
    config.rotation_rad_s = tuple(rotation)
    return _RamanNoiseKernel(ensemble, config)
