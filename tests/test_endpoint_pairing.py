"""Paired fluorescence-sensitivity intervals, including unbounded noise."""

import copy
import json
import unittest

import numpy as np

from rb87_beam.config import DetectionConfig
from rb87_beam.endpoint_comparison import _paired_endpoint
from rb87_beam.endpoint_statistics import endpoint_statistics


def endpoint(amplitude=.2, weight=1., flux_scale=1.):
    record = dict(atom_ids=np.arange(4),
        fringe_coefficients=np.tile([.5, amplitude, 0.], (4, 1)),
        detection_weights=np.full(4, weight), vz_m_s=np.full(4, 10.),
        target_population=np.full(4, .98), scattered_photons=np.full(4, 3.6))
    return endpoint_statistics({"nominal": [record]}, 4, 1e6,
        DetectionConfig(), bootstrap=160, seed=83087, flux_scale=flux_scale)


class EndpointPairingTests(unittest.TestCase):
    def test_identical_finite_strategies_have_exactly_zero_paired_difference(self):
        summary, draws = endpoint()
        result = _paired_endpoint(summary, draws, copy.deepcopy(summary), copy.deepcopy(draws))
        for key in ("worst_readout_snr_per_rad", "worst_phase_noise_rad"):
            self.assertEqual(result[key + "_difference"], 0.)
            self.assertEqual(result[key + "_difference_95ci"], [0., 0.])
        json.dumps(result, allow_nan=False)

    def test_identical_flat_fringes_have_no_fabricated_finite_noise_difference(self):
        summary, draws = endpoint(amplitude=0.)
        self.assertIsNone(summary["robust"]["worst_phase_noise_rad"])
        self.assertTrue(np.all(np.isinf(draws["worst_phase_noise_rad"])))
        result = _paired_endpoint(summary, draws, summary, draws)
        self.assertEqual(result["worst_readout_snr_per_rad_difference"], 0.)
        self.assertEqual(result["worst_readout_snr_per_rad_difference_95ci"], [0., 0.])
        self.assertIsNone(result["worst_phase_noise_rad_difference"])
        self.assertIsNone(result["worst_phase_noise_rad_difference_95ci"])
        json.dumps(result, allow_nan=False)

    def test_missing_detected_signal_has_zero_snr_but_undefined_phase_noise_delta(self):
        absent, da = endpoint(weight=0.)
        finite, df = endpoint()
        for a, ad, b, bd in ((absent, da, finite, df), (finite, df, absent, da)):
            result = _paired_endpoint(a, ad, b, bd)
            self.assertIsNotNone(result["worst_readout_snr_per_rad_difference"])
            self.assertIsNotNone(result["worst_readout_snr_per_rad_difference_95ci"])
            self.assertIsNone(result["worst_phase_noise_rad_difference"])
            self.assertIsNone(result["worst_phase_noise_rad_difference_95ci"])
            json.dumps(result, allow_nan=False)

    def test_mismatched_bootstrap_identity_is_rejected(self):
        summary, draws = endpoint()
        for key, wrong_value in (("seed", 19), ("counts_sha256", "different"),
                                 ("incident_samples", 5), ("bootstrap", 159)):
            altered = copy.deepcopy(draws)
            altered["bootstrap_identity"][key] = wrong_value
            with self.subTest(key=key), self.assertRaises(ValueError):
                _paired_endpoint(summary, draws, summary, altered)

    def test_summary_identity_must_match_draw_identity_on_both_sides(self):
        summary, draws = endpoint()
        changed = copy.deepcopy(summary)
        changed["bootstrap_identity"]["seed"] += 1
        missing = copy.deepcopy(summary)
        missing.pop("bootstrap_identity")
        for bad in (changed, missing):
            with self.assertRaises(ValueError):
                _paired_endpoint(bad, draws, summary, draws)
            with self.assertRaises(ValueError):
                _paired_endpoint(summary, draws, bad, draws)

    def test_different_scenario_sets_are_rejected_even_with_matching_counts(self):
        summary, draws = endpoint()
        changed = copy.deepcopy(draws)
        changed["scenarios"]["different"] = changed["scenarios"].pop("nominal")
        with self.assertRaises(ValueError):
            _paired_endpoint(summary, draws, summary, changed)

    def test_broadcastable_or_shortened_draw_arrays_are_rejected(self):
        summary, draws = endpoint()
        for key in ("worst_readout_snr_per_rad", "worst_phase_noise_rad"):
            for values in (draws[key][0], draws[key][:1], draws[key][:-1], draws[key][:, None]):
                changed = copy.deepcopy(draws)
                changed[key] = values
                with self.subTest(key=key, shape=np.shape(values)), self.assertRaises(ValueError):
                    _paired_endpoint(summary, draws, summary, changed)

    def test_equal_flux_identical_policies_stay_identical_after_thinning(self):
        s1, d1 = endpoint(flux_scale=.9)
        s2, d2 = endpoint(flux_scale=.9)
        delta = _paired_endpoint(s1, d1, s2, d2)
        self.assertEqual(delta["worst_readout_snr_per_rad_difference"], 0.)
        self.assertEqual(delta["worst_readout_snr_per_rad_difference_95ci"], [0., 0.])


if __name__ == "__main__":
    unittest.main()
