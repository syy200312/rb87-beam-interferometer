"""Regression and analytic checks for noise inside the three Raman pulses."""

import unittest
from dataclasses import replace

import numpy as np
from scipy.linalg import expm

from rb87_beam.config import SimulationConfig
from rb87_beam.physics import Ensemble, GAMMA, K_EFF, raman_zone, run_simulation
from rb87_beam.raman_noise import make_kernel
from rb87_beam.raman_noise_metrics import make_detector, summarize_noise
from rb87_beam.raman_noise_study import NoiseSettings, sample_disturbances
from rb87_beam.states import E_CLOCK, G_CLOCK


ZERO = np.zeros(3)
UNITY = np.ones(3)


def example(count=1):
    c = SimulationConfig()
    c.pumping.enabled = False
    c.detection.noise = False
    c.raman.envelope = "rectangular"
    c.raman.acceleration_m_s2 = (0., 0., 0.)
    c.raman.selection_sigma_vx_m_s = 0.
    c.raman.preparation_efficiency = 1.
    c.raman.integration_steps = 16
    c.raman.scan_points = 11
    v = np.zeros((count, 3))
    v[:, 2] = c.raman.reference_vz_m_s
    return c, Ensemble(np.zeros_like(v), v, 1e6)


def fringe_at(coefficients, phases):
    phase = np.atleast_1d(phases)
    return np.column_stack((np.ones(len(phase)), np.cos(phase), np.sin(phase))) @ coefficients


class RamanNoiseChecks(unittest.TestCase):
    def test_zero_noise_matches_original_for_each_atom_including_coherences(self):
        c, e = example(3)
        e.v_m_s[:, 2] = (10.4, 11., 13.)
        e.v_m_s[:, 0] = (.008, -.005, .003)
        e.r_m[:, 1] = (.001, .0006, -.0006)
        c.raman.envelope = "gaussian"
        c.raman.two_photon_detuning_Hz = 1700.
        c.raman.chirp_rad_s2 = 3.6e5
        c.raman.differential_light_shift_Hz = -350.
        c.raman.rotation_rad_s = (0., 4e-5, 0.)
        rho = np.zeros((3, 8, 8), complex)
        rho[:, G_CLOCK, G_CLOCK] = .45
        rho[:, E_CLOCK, E_CLOCK] = .35
        rho[:, 7, 7] = .20
        rho[:, G_CLOCK, E_CLOCK] = .08 + .11j
        rho[:, E_CLOCK, G_CLOCK] = rho[:, G_CLOCK, E_CLOCK].conj()
        e.ground_density = rho
        expected = run_simulation(c, e)
        h = make_kernel(e, c.raman).harmonics(ZERO, ZERO, ZERO, UNITY, .25)
        np.testing.assert_allclose(fringe_at(h, expected["phase_rad"]), expected["per_atom_P2"],
                                   rtol=0, atol=3e-11)

    def test_retroreflector_displacement_sign_and_common_phase_rejection(self):
        c, e = example()
        kernel = make_kernel(e, c.raman)
        phi = np.array([.1, -.13, .02])
        x = np.array([2.3, -3.7, .15]) * 1e-9
        scan = np.array([.2, 1.4, 3.1])
        h = kernel.harmonics(phi, x, ZERO, UNITY, 0.)
        phase_mz = np.dot([1., -2., 1.], phi - K_EFF * x)
        np.testing.assert_allclose(fringe_at(h, scan)[:, 0],
                                   .5 * (1 - np.cos(scan + phase_mz)), atol=1e-11)
        common = kernel.harmonics(phi + .9, x + 43e-9, ZERO, UNITY, 0.)
        np.testing.assert_allclose(common, h, atol=1e-11)

    def test_phase_jitter_reduces_average_fringe_even_when_single_shots_have_unit_contrast(self):
        c, e = example()
        kernel = make_kernel(e, c.raman)
        offset = .23
        h = np.stack([kernel.harmonics([0., sign * offset, 0.], ZERO, ZERO, UNITY, 0.)
                      for sign in (-1., 1.)])[:, :, 0]
        np.testing.assert_allclose(np.hypot(h[:, 1], h[:, 2]) / h[:, 0], 1., atol=1e-12)
        avg = h.mean(axis=0)
        self.assertAlmostEqual(np.hypot(*avg[1:]) / avg[0], np.cos(2 * offset), places=11)

    def test_two_color_common_power_scales_rabi_and_ac_stark_shift(self):
        c, e = example(2)
        c.raman.envelope = "gaussian"
        c.raman.coupling_model = "rb87_d2_power"
        c.raman.spin_transport = "adiabatic_from_z"
        c.raman.waist_z_m = 18.e-6
        c.raman.beam1_power_W = 2.14e-4
        c.raman.beam2_power_W = 1.2043e-4
        c.raman.differential_light_shift_Hz = 4600.
        e.v_m_s[1, 2] = 12.5
        factors = np.array([.89, 1.12, .96])
        h = make_kernel(e, c.raman).harmonics(ZERO, ZERO, ZERO, factors, .25, exact=True)
        for phi in (.41, 1.72, 2.96):
            s = np.zeros((2, 2), complex)
            s[:, 0] = 1.
            for j, factor in enumerate(factors):
                s = raman_zone(s, e, c.raman, j, scanned_phase=phi if j == 2 else 0.,
                               amplitude_scale=factor)
            np.testing.assert_allclose(fringe_at(h, phi)[0], np.abs(s[:, 1]) ** 2, atol=2e-11)

    def test_fast_noise_interpolation_against_direct_pulse_integration(self):
        c, e = example(2)
        e.v_m_s[1] = [.004, 0., 9.]
        c.raman.envelope = "gaussian"
        c.raman.differential_light_shift_Hz = -1600.
        c.raman.two_photon_detuning_Hz = 1700.
        kernel = make_kernel(e, c.raman, rotation_delta_y=1e-5)
        args = ([.04, .07, -.012], [1.2e-9, 1.8e-9, -1.1e-9],
                [.0094, -.0143, .0079], [1.0167, .9892, 1.0786], .1)
        direct = kernel.harmonics(*args, exact=True)
        np.testing.assert_allclose(kernel.harmonics(*args), direct, rtol=0, atol=4e-8)

    def test_noise_sampler_covariance_positive_power_and_shared_seeds(self):
        s = NoiseSettings(phase_rms_rad=.05, phase_zone_correlation=-1.,
                          mirror_displacement_rms_m=3e-9, mirror_zone_correlation=1.,
                          bias_field_G=.1, field_rms_G=.008, field_zone_correlation=.55,
                          fractional_power_rms=.08, log_power_zone_correlation=.3,
                          gate_s=.01, record_period_s=.02, operating_phase_rad=.6,
                          rotation_step_rad_s=1e-6, records=30000, scans=4, phase_points=11, seed=5587)
        draws = sample_disturbances(s)
        np.testing.assert_array_equal(draws["phase"][:, 0], -draws["phase"][:, 1])
        np.testing.assert_array_equal(draws["vibration"][:, 0], draws["vibration"][:, 2])
        self.assertAlmostEqual(np.corrcoef(draws["magnetic"].T)[0, 2], .55**2, delta=.025)
        self.assertGreater(draws["intensity"].min(), 0.)
        self.assertAlmostEqual(draws["intensity"].mean(), 1., delta=.002)
        self.assertAlmostEqual(draws["intensity"].std(), .08, delta=.002)
        other = sample_disturbances(replace(s, field_rms_G=0., fractional_power_rms=0.))
        np.testing.assert_array_equal(draws["phase"], other["phase"])
        np.testing.assert_array_equal(other["magnetic"], 0.)
        np.testing.assert_array_equal(other["intensity"], 1.)

    def test_zeeman_in_pulses_and_dark_gaps_matches_matrix_exponential(self):
        c, e = example()
        bias = .35
        b = np.array([.12, -.027, -.093])
        pulse_phases = np.array([.13, -.06, 0.])
        h = make_kernel(e, c.raman).harmonics(pulse_phases, ZERO, b, UNITY, bias, exact=True)
        # Constant rectangular drives and spatially piecewise constant B permit
        # an independent exact matrix exponential for each pulse and dark half-gap.
        delta = 2 * np.pi * 575.15 * (2 * bias * b + b ** 2)
        duration = c.raman.rectangular_width_m / e.v_m_s[0, 2]
        gap = c.raman.zone_spacing_m / e.v_m_s[0, 2] - duration
        for scan_phase in (.41, 1.72, 2.96):
            state = np.array([1., 0.], complex)
            for j, area in enumerate((np.pi / 2, np.pi, np.pi / 2)):
                phi = pulse_phases[j] + (scan_phase if j == 2 else 0.)
                coupling = area / duration * np.exp(1j * phi)
                ham = .5 * np.array([[-delta[j], coupling.conjugate()], [coupling, delta[j]]])
                state = expm(-1j * ham * duration) @ state
                if j < 2:
                    integrated_shift = .5 * gap * (delta[j] + delta[j+1])
                    state *= np.exp(.5j * np.array([1., -1.]) * integrated_shift)
            self.assertAlmostEqual(fringe_at(h, scan_phase)[0, 0], abs(state[1]) ** 2, places=11)

    def test_readout_conditional_variance_common_flux_and_one_second_scaling(self):
        c, e = example(2)
        e.r_m[1, 0] = 2 * c.detection.aperture_radius_m
        rho = np.zeros((2, 8, 8), complex)
        rho[:, G_CLOCK, G_CLOCK] = .6
        rho[:, 0, 0] = .4
        e.ground_density = rho
        det = make_detector(e, c)
        accepted_flux = e.flux_per_s / 2
        self.assertAlmostEqual(det.accepted_flux_per_s, accepted_flux)
        self.assertAlmostEqual(det.target_flux_per_s, .6 * accepted_flux)
        # For the single admitted speed, the gain-normalised detector is a
        # two-channel ratio with known independent compound-Poisson variances.
        records = 32
        fixed = np.zeros((records, 3, 2))
        fixed[:, 0, :] = .3
        fixed[:, 1, :] = .1
        phase = np.pi / 2
        y, variance = det.moments(fixed, phase)
        atoms = accepted_flux * c.detection.gate_s
        d = c.detection
        sat = np.array([d.saturation_F1, d.saturation_F2])
        q = (GAMMA / 2 * d.interaction_length_m / e.v_m_s[0, 2] *
             d.collection_fraction * d.quantum_efficiency * sat / (1 + sat))
        var_counts = atoms * np.array([.7, .3]) * (q + q ** 2)
        var_counts += d.background_counts + d.electronic_noise_std_counts ** 2
        analytic_var = np.sum((np.array([.3, .7]) / q) ** 2 * var_counts) / atoms ** 2
        np.testing.assert_allclose(y, .3, atol=2e-15)
        np.testing.assert_allclose(variance, analytic_var, rtol=1e-12)
        fluctuating = fixed.copy()
        fluctuating[::2, 0, :] += .02
        fluctuating[1::2, 0, :] -= .02
        slope = -15.
        for h in (fixed, fluctuating):
            metrics, _ = summarize_noise(h, det, phase, slope, scans=64, seed=7)
            mean_p, conditional_var = det.moments(h, phase)
            total_var = conditional_var.mean() + mean_p.var(ddof=1)
            self.assertAlmostEqual(metrics["accepted_flux_per_s"], accepted_flux)
            self.assertAlmostEqual(metrics["target_flux_per_s"], .6 * accepted_flux)
            self.assertAlmostEqual(metrics["technical_variance_P2"], mean_p.var(ddof=1))
            self.assertAlmostEqual(metrics["detection_variance_P2"], conditional_var.mean())
            self.assertAlmostEqual(metrics["allan_1s_rad_s"], np.sqrt(total_var*d.gate_s) / abs(slope))


if __name__ == "__main__":
    unittest.main()
