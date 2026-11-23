"""Independent angular-momentum and SI-power checks for Raman calibration."""

import unittest

import numpy as np
from scipy.constants import c, epsilon_0, h, hbar
from sympy import Rational
from sympy.physics.wigner import wigner_3j, wigner_6j

from rb87_beam.physics import GAMMA
from rb87_beam.raman_calibration import calibrate_powers, d2_dipoles_SI, solve_powers


STECK_D = 3.58424e-29
WY, WZ = .006, 18e-6
GROUND = [(f, m) for f in (1, 2) for m in range(-f, f + 1)]
EXCITED = [(f, m) for f in (0, 1, 2, 3) for m in range(-f, f + 1)]
G1, G2 = GROUND.index((1, 0)), GROUND.index((2, 0))
EXCITED_HZ = np.array([dict(enumerate((-302073750., -229851750., -72911250., 193740750.)))[f]
                       for f, _ in EXCITED])
HF_HZ = 6834682610.90429


def independent_dipoles():
    """Condon--Shortley 3-j/6-j reduction, independent of PyLCP matrices.

    Start with lower-state-bra matrix elements in the standard Wigner-Eckart
    convention, then use d_q^dagger=(-1)^q d_-q to form absorption elements.
    """
    j, jp, spin = Rational(1, 2), Rational(3, 2), Rational(3, 2)
    result = np.zeros((3, 16, 8))
    for iq, q in enumerate((-1, 0, 1)):
        for ie, (fp, mp) in enumerate(EXCITED):
            for ig, (f, m) in enumerate(GROUND):
                if mp != m + q:
                    continue
                phase = (-1.) ** int(f - m + j + spin + fp + 1 + q)
                angular = float(wigner_3j(f, 1, fp, -m, -q, mp)
                                * wigner_6j(j, f, spin, fp, jp, 1))
                result[iq, ie, ig] = (phase * np.sqrt((2*f+1)*(2*fp+1))
                                      * angular * np.sqrt(2.) * STECK_D)
    return result


def analytic_sigma_plus(power1, power2, detuning=1.073e9, two_photon=0.):
    """Direct Hz formulas including both laser frequencies on both states."""
    d = independent_dipoles()[2]
    reference = EXCITED_HZ[EXCITED.index((1, 0))]
    delta1 = detuning + reference - EXCITED_HZ
    delta2 = delta1 - two_photon
    field1 = np.sqrt(4 * power1 / (np.pi * WY * WZ * epsilon_0 * c))
    field2 = np.sqrt(4 * power2 / (np.pi * WY * WZ * epsilon_0 * c))
    one1, one2 = field1 * d / h, field2 * d / h
    rabi = abs(np.sum(one1[:, G1] * one2[:, G2]
                      * (1 / delta1 + 1 / delta2)) / 4)
    shift1 = np.sum(one1[:, G1] ** 2 / delta1
                    + one2[:, G1] ** 2 / (delta2 - HF_HZ)) / 4
    shift2 = np.sum(one1[:, G2] ** 2 / (delta1 + HF_HZ)
                    + one2[:, G2] ** 2 / delta2) / 4
    scattering1 = GAMMA * np.sum(one1[:, G1] ** 2 / delta1 ** 2
                                 + one2[:, G1] ** 2 / (delta2 - HF_HZ) ** 2) / 4
    scattering2 = GAMMA * np.sum(one1[:, G2] ** 2 / (delta1 + HF_HZ) ** 2
                                 + one2[:, G2] ** 2 / delta2 ** 2) / 4
    return rabi, np.array([shift1, shift2]), np.array([scattering1, scattering2])


class RamanCalibrationTests(unittest.TestCase):
    def test_dipoles_match_independent_wigner_reduction_and_decay_sum_rule(self):
        actual, ground, excited = d2_dipoles_SI()
        expected = independent_dipoles()
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=1e-43)
        cycling = abs(actual[2, EXCITED.index((3, 3)), GROUND.index((2, 2))])
        self.assertAlmostEqual(cycling / (STECK_D / np.sqrt(2.)), 1., places=13)
        np.testing.assert_allclose(np.sum(abs(actual) ** 2, axis=(0, 2)),
                                   STECK_D ** 2 / 2, rtol=2e-13, atol=0.)
        np.testing.assert_allclose(excited, EXCITED_HZ, rtol=0., atol=1e-6)
        self.assertAlmostEqual((ground[G2] - ground[G1]) / HF_HZ, 1., places=13)

    def test_cycling_matrix_element_reproduces_steck_linewidth_and_saturation(self):
        d = STECK_D / np.sqrt(2.)
        omega = 2 * np.pi * c / 780.241209686e-9
        gamma_from_d = omega ** 3 * d ** 2 / (3 * np.pi * epsilon_0 * hbar * c ** 3)
        intensity_sat = c * epsilon_0 * hbar ** 2 * gamma_from_d ** 2 / (4 * d ** 2)
        self.assertAlmostEqual(gamma_from_d / (2 * np.pi * 1e6), 6.06654, delta=.00002)
        self.assertAlmostEqual(intensity_sat, 16.6933, delta=.0002)
        # The repository's rounded 6.065 MHz linewidth is not an independent
        # normalization convention; its small numerical difference is explicit.
        self.assertAlmostEqual(gamma_from_d / GAMMA, 1., delta=.0003)

    def test_sigma_path_sum_and_both_frequency_stark_shifts_match_si_formulas(self):
        for two_photon in (0., -25e3):
            with self.subTest(two_photon=two_photon):
                actual = calibrate_powers(.21e-3, .12e-3, WY, WZ,
                                          two_photon_detuning_Hz=two_photon)
                rabi, shifts, scattering = analytic_sigma_plus(.21e-3, .12e-3, two_photon=two_photon)
                # The repository rounds hbar to ten significant digits; the
                # reference above uses the exact SI Planck constant instead.
                self.assertAlmostEqual(actual["peak_rabi_Hz"] / rabi, 1., delta=2e-9)
                np.testing.assert_allclose(np.asarray(actual["ground_light_shifts_Hz"])[[G1, G2]], shifts, rtol=2e-9)
                np.testing.assert_allclose(np.asarray(actual["ground_scattering_rates_per_s"])[[G1, G2]], scattering, rtol=2e-9)
                self.assertAlmostEqual(actual["differential_light_shift_Hz"], shifts[1] - shifts[0], delta=5e-4)

    def test_parallel_linear_and_pi_paths_cancel_coherently(self):
        actual = calibrate_powers(.2e-3, .2e-3, WY, WZ,
                                 polarization1="linear", polarization2="linear")
        self.assertLess(actual["peak_rabi_Hz"], 1e-8)
        self.assertGreater(max(actual["ground_scattering_rates_per_s"]), 0.)
        dipoles = independent_dipoles()
        np.testing.assert_array_equal(dipoles[1, :, G1] * dipoles[1, :, G2], 0.)
        # A pi electric field is longitudinal in the assumed B||k geometry.
        with self.assertRaises(ValueError):
            calibrate_powers(.2e-3, .2e-3, WY, WZ, polarization1="pi", polarization2="pi")

    def test_cancellation_ratio_and_power_solution_include_wrong_legs(self):
        target = 182848.54518399003
        result = solve_powers(target, WY, WZ, two_photon_detuning_Hz=-25e3)
        self.assertAlmostEqual(result["power_ratio"], 1.776909969888562, delta=1e-12)
        self.assertAlmostEqual(result["power1_W"] / .214000012430338e-3, 1., delta=2e-9)
        self.assertAlmostEqual(result["power2_W"] / .120433795778499e-3, 1., delta=2e-9)
        self.assertAlmostEqual(result["peak_rabi_Hz"], target, delta=1e-7)
        self.assertAlmostEqual(result["differential_light_shift_Hz"], 0., delta=1e-7)
        self.assertGreater(result["ground_light_shifts_Hz"][G1], 2e5)
        equal = solve_powers(target, WY, WZ, two_photon_detuning_Hz=-25e3, power_ratio=1.)
        self.assertEqual(equal["power1_W"], equal["power2_W"])
        self.assertAlmostEqual(equal["differential_light_shift_Hz"], 116180.21853563747, delta=1e-6)
        # F'=3 is absent from the clock Raman product but remains in Stark
        # shifts/scattering. Keeping only common Raman paths changes the ratio.
        dipoles = independent_dipoles()[2]
        delta = 1.073e9 + EXCITED_HZ[EXCITED.index((1, 0))] - EXCITED_HZ
        naive_ratio = np.sum(dipoles[:, G2] ** 2 / (delta + 25e3)) / np.sum(dipoles[:, G1] ** 2 / delta)
        self.assertGreater(abs(result["power_ratio"] - naive_ratio), .1)
        self.assertEqual([row["Fprime"] for row in result["path_contributions"]], [0, 1, 2, 3])
        self.assertEqual(result["path_contributions"][0]["magnitude_Hz"], 0.)
        self.assertEqual(result["path_contributions"][3]["magnitude_Hz"], 0.)
        self.assertGreater(result["path_contributions"][1]["magnitude_Hz"], 0.)
        self.assertGreater(result["path_contributions"][2]["magnitude_Hz"], 0.)

    def test_no_finite_power_recovers_a_coherently_forbidden_clock_transition(self):
        with self.assertRaisesRegex(ValueError, "cancel"):
            solve_powers(100e3, WY, WZ, polarization1="linear", polarization2="linear", power_ratio=1.)
        forbidden = calibrate_powers(.2e-3, .2e-3, WY, WZ,
                                    polarization1="sigma_plus", polarization2="sigma_minus")
        self.assertEqual(forbidden["peak_rabi_Hz"], 0.)

    def test_power_product_area_and_middle_pulse_scaling(self):
        base = calibrate_powers(.21e-3, .12e-3, WY, WZ)
        one_scaled = calibrate_powers(.84e-3, .12e-3, WY, WZ)
        middle = calibrate_powers(.42e-3, .24e-3, WY, WZ)
        wider = calibrate_powers(.21e-3, .12e-3, WY, 2 * WZ)
        self.assertAlmostEqual(one_scaled["peak_rabi_Hz"] / base["peak_rabi_Hz"], 2.)
        self.assertAlmostEqual(wider["peak_rabi_Hz"] / base["peak_rabi_Hz"], .5)
        for key in ("peak_rabi_Hz", "differential_light_shift_Hz",
                    "ground_light_shifts_Hz", "ground_scattering_rates_per_s"):
            np.testing.assert_allclose(middle[key], 2 * np.asarray(base[key]), rtol=1e-12, atol=1e-9)

    def test_near_resonance_and_nonperturbative_coupling_are_rejected(self):
        with self.assertRaises(ValueError):
            calibrate_powers(.2e-3, .2e-3, WY, WZ, detuning_Hz=0.)
        # The intended pair is far away, but low-frequency light is resonant
        # with the wrong F=1 optical leg. It must also invalidate elimination.
        with self.assertRaises(ValueError):
            calibrate_powers(.2e-3, .2e-3, WY, WZ, detuning_Hz=HF_HZ)
        with self.assertRaises(ValueError):
            calibrate_powers(1., 1., WY, WZ)

    def test_reference_line_changes_leave_the_same_physical_frequencies_invariant(self):
        first = calibrate_powers(.21e-3, .12e-3, WY, WZ,
                                  reference_Fprime=1, detuning_Hz=1.073e9)
        second = calibrate_powers(.21e-3, .12e-3, WY, WZ,
                                  reference_Fprime=2, detuning_Hz=916059500.)
        for key in ("peak_rabi_Hz", "differential_light_shift_Hz", "optical_frequency_offsets_Hz",
                    "ground_light_shifts_Hz", "ground_scattering_rates_per_s"):
            np.testing.assert_allclose(second[key], first[key], rtol=1e-13, atol=1e-7)

    def test_one_beam_has_stark_and_scattering_but_no_raman_transfer(self):
        one = calibrate_powers(.2e-3, 0., WY, WZ)
        self.assertEqual(one["peak_rabi_Hz"], 0.)
        self.assertGreater(one["ground_scattering_rates_per_s"][G1], 0.)
        self.assertGreater(abs(one["ground_light_shifts_Hz"][G1]), 0.)
        dark = calibrate_powers(0., 0., WY, WZ, detuning_Hz=0.)
        self.assertEqual(dark["peak_rabi_Hz"], 0.)
        np.testing.assert_array_equal(dark["ground_light_shifts_Hz"], 0.)
        np.testing.assert_array_equal(dark["ground_scattering_rates_per_s"], 0.)


if __name__ == "__main__":
    unittest.main()
