"""Physical readout-noise and source-cluster checks for end-signal tuning."""

import copy
import json
import unittest

import numpy as np

from rb87_beam.config import DetectionConfig, RamanConfig
from rb87_beam.endpoint_statistics import endpoint_statistics
from rb87_beam.joint_statistics import case_statistics
from rb87_beam.physics import GAMMA, Ensemble, fluorescence_scan


def record(coefficients, speed=None, ids=None, weight=None):
    coefficients = np.asarray(coefficients, float)
    n = len(coefficients)
    return dict(atom_ids=np.arange(n) if ids is None else np.asarray(ids),
        fringe_coefficients=coefficients,
        vz_m_s=np.full(n, 10.) if speed is None else np.asarray(speed, float),
        detection_weights=np.ones(n) if weight is None else np.asarray(weight, float),
        target_population=np.full(n, .98), scattered_photons=np.full(n, 3.6))


def detector():
    c = DetectionConfig()
    c.saturation_F1 = c.saturation_F2 = 1.
    c.background_counts = c.electronic_noise_std_counts = 0.
    return c


def stats(scenarios, n, config=None, **kwargs):
    return endpoint_statistics(scenarios, n, 1e6, config or detector(), bootstrap=160, seed=83087, **kwargs)


class EndpointStatisticsTests(unittest.TestCase):
    def test_uniform_beam_matches_closed_form_compound_photon_noise(self):
        c = detector()
        s, _ = stats({"nominal": [record([[.5, .2, 0]] * 4)]}, 4, c)
        nominal = s["scenarios"]["nominal"]
        arrivals = 1e6 * c.gate_s
        q = GAMMA / 4 * c.interaction_length_m / 10 * c.collection_fraction * c.quantum_efficiency
        # At balanced output the normalized variance is (1+1/q)/(4*N).
        variance = (1 + 1 / q) / (4 * arrivals)
        self.assertAlmostEqual(nominal["normalized_P2"], .5)
        self.assertAlmostEqual(nominal["normalized_slope_per_rad"], -.2)
        self.assertAlmostEqual(nominal["noise_std_P2"] ** 2, variance)
        self.assertAlmostEqual(nominal["readout_snr_per_rad"], .2 / np.sqrt(variance))
        self.assertAlmostEqual(nominal["phase_noise_rad"], np.sqrt(variance) / .2)
        self.assertAlmostEqual(nominal["F2_counts"], arrivals * q / 2)
        self.assertAlmostEqual(nominal["F2_variance"], arrivals * (q + q ** 2) / 2)
        self.assertGreater(nominal["F2_variance"], nominal["F2_counts"])
        json.dumps(s, allow_nan=False)

    def test_count_moments_match_independent_compound_poisson_sampling(self):
        c = detector()
        c.gate_s = 1e-4
        c.background_counts, c.electronic_noise_std_counts = 7., 2.
        s, _ = stats({"nominal": [record([[.3, .1, 0]])]}, 1, c)
        nominal = s["scenarios"]["nominal"]
        rng = np.random.default_rng(881)
        q = GAMMA / 4 * c.interaction_length_m / 10 * c.collection_fraction * c.quantum_efficiency
        atoms = rng.poisson(1e6 * c.gate_s * .3, 150000)
        counts = rng.poisson(atoms * q + 7.) + rng.normal(0, 2., len(atoms))
        self.assertAlmostEqual(counts.mean() / nominal["F2_counts"], 1., delta=.003)
        self.assertAlmostEqual(counts.var() / nominal["F2_variance"], 1., delta=.012)

    def test_asymmetric_channels_and_lost_launches_match_actual_detector(self):
        c = DetectionConfig(noise=False)
        raman = RamanConfig(acceleration_m_s2=(0., 0., 0.))
        source = record([[.5, .2, .1], [.3, .1, -.05]], speed=[10., 20.],
                        ids=[0, 2], weight=[.2, .8])
        phase, step = .4, 1e-5
        phases = np.array([phase - step, phase, phase + step])
        design = np.column_stack((np.ones(3), np.cos(phases), np.sin(phases)))
        probabilities = design @ source["fringe_coefficients"].T
        # Two of four incident IDs reach the output: preserve that flux loss.
        ensemble = Ensemble(np.zeros((2, 3)), np.array([[0., 0., 10.], [0., 0., 20.]]), .5e6)
        actual = fluorescence_scan(probabilities, ensemble, source["detection_weights"],
                                   c, raman, np.random.default_rng(41))
        summary, _ = stats({"nominal": [source]}, 4, c, phase_rad=phase)
        s = summary["scenarios"]["nominal"]
        self.assertAlmostEqual(s["F1_counts"], actual["F1_counts"][1])
        self.assertAlmostEqual(s["F2_counts"], actual["F2_counts"][1])
        self.assertAlmostEqual(s["normalized_P2"], actual["expected_photon_P2"][1])
        derivative = (actual["expected_photon_P2"][2] - actual["expected_photon_P2"][0]) / (2 * step)
        self.assertAlmostEqual(s["normalized_slope_per_rad"], derivative, delta=2e-11)
        self.assertAlmostEqual(s["expected_detected_atoms"], actual["expected_atoms_per_gate"])

    def test_nominal_quadrature_is_frozen_when_scenario_phase_changes(self):
        nominal = record([[.5, 0., .2]] * 4)
        drift = record([[.5, .2, 0.]] * 4)
        s, draws = stats({"nominal": [nominal], "drift": [drift]}, 4)
        self.assertAlmostEqual(s["operating_phase_rad"], np.pi)
        self.assertAlmostEqual(s["scenarios"]["nominal"]["normalized_slope_per_rad"], -.2)
        self.assertLess(s["scenarios"]["drift"]["readout_snr_per_rad"], 1e-12)
        np.testing.assert_array_equal(draws["worst_readout_snr_per_rad"],
            np.minimum(draws["scenarios"]["nominal"]["readout_snr_per_rad"], draws["scenarios"]["drift"]["readout_snr_per_rad"]))

    def test_source_bootstrap_matches_common_statistics_identity(self):
        scenarios = {"nominal": [record([[.2, .1, 0], [.4, .2, 0], [.6, .1, 0], [.7, .1, 0]], speed=[10., 12., 13., 20.])]}
        s, d = stats(scenarios, 4)
        joint, _ = case_statistics(scenarios, 4, 1e6,
            dict(min_efficiency=.9, min_effective_target_flux_per_s=0., max_scattered_photons=6.),
            bootstrap=160, seed=83087)
        self.assertEqual(s["bootstrap_identity"], joint["bootstrap_identity"])
        self.assertEqual(d["bootstrap_identity"], s["bootstrap_identity"])
        repeated = {"nominal": scenarios["nominal"] * 3}
        many, dm = stats(repeated, 4)
        for key in d["scenarios"]["nominal"]:
            np.testing.assert_allclose(d["scenarios"]["nominal"][key], dm["scenarios"]["nominal"][key], rtol=1e-12, atol=1e-12)
        self.assertEqual(s["bootstrap_identity"], many["bootstrap_identity"])

    def test_recoil_residence_weights_are_combined_before_normalization(self):
        first = record([[.8, .1, 0]], speed=[10.])
        second = record([[.2, 0, .1]], speed=[20.])
        s, _ = stats({"nominal": [first, second]}, 1, phase_rad=0.)
        self.assertAlmostEqual(s["scenarios"]["nominal"]["normalized_P2"], (.9 + .2 / 2) / 1.5)
        self.assertAlmostEqual(s["scenarios"]["nominal"]["normalized_slope_per_rad"], .1 / 3)

    def test_uniform_thinning_preserves_signal_shape_and_background(self):
        source = {"nominal": [record([[.5, .2, 0]] * 4)]}
        c = detector()
        c.background_counts, c.electronic_noise_std_counts = 37., 3.
        full, _ = stats(source, 4, c)
        quarter, _ = stats(source, 4, c, flux_scale=.25)
        a, b = full["scenarios"]["nominal"], quarter["scenarios"]["nominal"]
        self.assertEqual(a["normalized_P2"], b["normalized_P2"])
        self.assertEqual(a["normalized_slope_per_rad"], b["normalized_slope_per_rad"])
        self.assertAlmostEqual(b["F2_counts"] - 37., .25 * (a["F2_counts"] - 37.))
        self.assertAlmostEqual(b["F2_variance"] - 46., .25 * (a["F2_variance"] - 46.))
        self.assertGreater(b["phase_noise_rad"], 2 * a["phase_noise_rad"])

    def test_flat_fringe_has_zero_information_and_unbounded_phase_noise(self):
        s, d = stats({"nominal": [record([[.5, 0, 0]] * 2)]}, 2)
        self.assertEqual(s["scenarios"]["nominal"]["readout_snr_per_rad"], 0.)
        self.assertIsNone(s["scenarios"]["nominal"]["phase_noise_rad"])
        self.assertIsNone(s["robust"]["worst_phase_noise_rad_95ci"])
        self.assertTrue(np.all(np.isinf(d["worst_phase_noise_rad"])))
        json.dumps(s, allow_nan=False)

    def test_empty_detector_does_not_create_a_finite_phase_resolution(self):
        c = detector()
        c.background_counts = 10.
        s, _ = stats({"nominal": [record([[.5, .2, 0]], weight=[0.])]}, 2, c)
        nominal = s["scenarios"]["nominal"]
        self.assertEqual(nominal["F2_counts"], 10.)
        self.assertEqual(nominal["readout_snr_per_rad"], 0.)
        self.assertIsNone(nominal["normalized_P2"])
        self.assertIsNone(nominal["phase_noise_rad"])
        json.dumps(s, allow_nan=False)

    def test_invalid_inputs_and_ambiguous_automatic_phase_are_rejected(self):
        source = {"nominal": [record([[.5, .2, 0]])]}
        for scale in (0., -1., 1.1, np.inf):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                stats(source, 1, flux_scale=scale)
        with self.assertRaises(ValueError):
            stats({"other": source["nominal"]}, 1)
        with self.assertRaises(ValueError):
            stats(source, 1, phase_rad=np.nan)
        malformed = copy.deepcopy(source)
        malformed["nominal"][0]["atom_ids"] = np.array([1])
        with self.assertRaises(ValueError):
            stats(malformed, 1)


if __name__ == "__main__":
    unittest.main()
