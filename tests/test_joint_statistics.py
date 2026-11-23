"""Independent statistical/physical checks for pump and Raman co-design."""

import copy
import json
import unittest

import numpy as np

from rb87_beam.joint_statistics import case_statistics, paired_comparison


LIMITS = {"min_efficiency": .8, "min_effective_target_flux_per_s": 0., "max_scattered_photons": 6.}


def replica(coefficients, speed=None, target=None, weight=None, ids=None, photons=None):
    coefficients = np.asarray(coefficients, dtype=float)
    n = len(coefficients)
    return dict(atom_ids=np.arange(n) if ids is None else np.asarray(ids),
                fringe_coefficients=coefficients,
                vz_m_s=np.ones(n) * 10 if speed is None else np.asarray(speed),
                target_population=np.ones(n) * .98 if target is None else np.asarray(target),
                detection_weights=np.ones(n) if weight is None else np.asarray(weight),
                scattered_photons=np.ones(n) * 3 if photons is None else np.asarray(photons))


def statistics(scenarios, n, **kwargs):
    return case_statistics(scenarios, n, 1e10, LIMITS, bootstrap=160, seed=71287, **kwargs)


class JointStatisticsTests(unittest.TestCase):
    def test_photon_weighted_coefficients_and_flux_follow_residence_time(self):
        # 10 m/s atom is bright with P=.8+.1cos(phi); 20 m/s atom is darker.
        source = replica([[.8, .1, 0], [.2, 0, .1]], speed=[10., 20.], target=[.9, .8])
        summary, _ = statistics({"nominal": [source]}, 2)
        s = summary["scenarios"]["nominal"]
        self.assertAlmostEqual(s["offset"], .6)
        self.assertAlmostEqual(s["cosine_coefficient"], 1 / 15)
        self.assertAlmostEqual(s["sine_coefficient"], 1 / 30)
        self.assertAlmostEqual(s["fringe_amplitude"], np.hypot(1 / 15, 1 / 30))
        self.assertAlmostEqual(s["fringe_contrast"], np.hypot(1 / 15, 1 / 30) / .6)
        self.assertAlmostEqual(s["effective_target_flux_per_s"] / 1e10, .85)
        self.assertAlmostEqual(s["detected_flux_per_s"], 1e10)

    def test_missing_source_launches_and_detector_losses_are_zero_flux(self):
        source = replica([[.5, .2, 0], [.5, .2, 0]], ids=[0, 2],
                         weight=[1., .5], target=[.8, .9])
        summary, _ = statistics({"nominal": [source]}, 4)
        s = summary["scenarios"]["nominal"]
        self.assertEqual(s["mean_extracted_samples"], 2)
        self.assertAlmostEqual(s["exit_flux_per_s"], .5e10)
        self.assertAlmostEqual(s["detected_flux_per_s"], .375e10)
        self.assertAlmostEqual(s["effective_target_flux_per_s"], 1e10 * (1 * .8 + .5 * .9) / 4)
        self.assertAlmostEqual(s["preparation_efficiency"], .85)
        self.assertAlmostEqual(s["score"], .4 * np.sqrt(.375))

    def test_repeated_recoil_paths_do_not_create_independent_source_samples(self):
        source = replica([[.2, .1, 0], [.4, .2, 0], [.6, .1, 0], [.7, .1, 0]],
                         target=[.9, .95, .99, .97])
        one, d1 = statistics({"nominal": [source]}, 4)
        many, d3 = statistics({"nominal": [source, source, source]}, 4)
        self.assertEqual(one["bootstrap_identity"], many["bootstrap_identity"])
        for metric, values in d1["scenarios"]["nominal"].items():
            np.testing.assert_allclose(values, d3["scenarios"]["nominal"][metric], atol=1e-5)
        self.assertAlmostEqual(one["robust"]["worst_score"], many["robust"]["worst_score"])
        np.testing.assert_allclose(one["robust"]["worst_score_95ci"], many["robust"]["worst_score_95ci"])

    def test_recoil_weighted_numerators_are_averaged_before_taking_ratios(self):
        first = replica([[.8, .1, 0]], speed=[10.])
        second = replica([[.2, 0, .1]], speed=[20.])
        summary, _ = statistics({"nominal": [first, second]}, 1)
        s = summary["scenarios"]["nominal"]
        # Averaging per-replica offsets would give .5, which is incorrect here.
        self.assertAlmostEqual(s["offset"], .6)
        self.assertAlmostEqual(s["cosine_coefficient"], 1 / 15)
        self.assertAlmostEqual(s["sine_coefficient"], 1 / 30)

    def test_worst_case_is_computed_inside_each_correlated_bootstrap_draw(self):
        first = replica([[.5, .24, 0], [.5, .02, 0], [.5, .24, 0], [.5, .02, 0]])
        second = replica([[.5, .02, 0], [.5, .24, 0], [.5, .02, 0], [.5, .24, 0]])
        summary, draws = statistics({"nominal": [first], "drift": [second]}, 4)
        x, y = draws["scenarios"]["nominal"]["score"], draws["scenarios"]["drift"]["score"]
        np.testing.assert_allclose(draws["worst_score"], np.minimum(x, y))
        np.testing.assert_allclose(x + y, .52)
        np.testing.assert_allclose(summary["robust"]["worst_score_95ci"], np.quantile(np.minimum(x, y), [.025, .975]))
        self.assertLess(summary["robust"]["worst_score_95ci"][1],
                        min(summary["scenarios"]["nominal"]["score_95ci"][1],
                            summary["scenarios"]["drift"]["score_95ci"][1]))

    def test_high_contrast_with_small_raman_acceptance_does_not_win(self):
        broad = replica(np.tile([.5, .2, 0], (4, 1)))
        narrow = replica(np.tile([.04, .038, 0], (4, 1)))
        b, _ = statistics({"nominal": [broad]}, 4)
        n, _ = statistics({"nominal": [narrow]}, 4)
        self.assertGreater(n["scenarios"]["nominal"]["fringe_contrast"], b["scenarios"]["nominal"]["fringe_contrast"])
        self.assertLess(n["robust"]["worst_score"], b["robust"]["worst_score"])
        self.assertLess(n["scenarios"]["nominal"]["modulated_flux_per_s"], b["scenarios"]["nominal"]["modulated_flux_per_s"])

    def test_constraints_apply_to_every_declared_scenario(self):
        good = replica(np.tile([.5, .2, 0], (4, 1)))
        drift = copy.deepcopy(good)
        drift["target_population"][:] = .7
        drift["scattered_photons"][:] = 7
        summary, _ = statistics({"nominal": [good], "drift": [drift]}, 4)
        self.assertFalse(summary["robust"]["feasible"])
        self.assertAlmostEqual(summary["robust"]["min_efficiency"], .7)
        self.assertAlmostEqual(summary["robust"]["max_scattered_photons"], 7)
        self.assertLess(summary["robust"]["constraint_margins"]["efficiency"]["value"], 0)
        self.assertLess(summary["robust"]["constraint_margins"]["scattered_photons"]["value"], 0)

    def test_contrast_floor_is_checked_in_the_worst_scenario(self):
        nominal = replica(np.tile([.5, .25, 0], (4, 1)))
        drift = replica(np.tile([.5, .15, 0], (4, 1)))
        limits = dict(LIMITS, min_fringe_contrast=.45)
        summary, _ = case_statistics({"nominal": [nominal], "drift": [drift]},
                                     4, 1e10, limits, bootstrap=160, seed=71287)
        self.assertAlmostEqual(summary["scenarios"]["nominal"]["fringe_contrast"], .5)
        self.assertAlmostEqual(summary["robust"]["worst_contrast"], .3)
        self.assertFalse(summary["robust"]["feasible"])
        self.assertAlmostEqual(summary["robust"]["constraint_margins"]["fringe_contrast"]["value"], -.15)

    def test_pairing_preserves_source_correlation_and_detects_mismatched_counts(self):
        source = replica([[.5, .1, 0], [.5, .12, 0], [.5, .16, 0], [.5, .18, 0]])
        changed = copy.deepcopy(source)
        changed["fringe_coefficients"][:, 1] += .02
        before, db = statistics({"nominal": [source]}, 4)
        after, da = statistics({"nominal": [changed]}, 4)
        paired = paired_comparison(before, db, after, da)
        np.testing.assert_allclose(paired["worst_score_improvement_95ci"], [.04, .04], atol=1e-14)
        np.testing.assert_allclose(paired["nominal_contrast_improvement_95ci"], [.04, .04], atol=1e-14)
        self.assertTrue(paired["worst_score_improvement_supported"])
        null = paired_comparison(before, db, before, db)
        self.assertEqual(null["worst_score_improvement_95ci"], [0., 0.])
        self.assertFalse(null["worst_score_improvement_supported"])
        wrong = copy.deepcopy(da)
        wrong["bootstrap_identity"]["seed"] += 1
        with self.assertRaises(ValueError):
            paired_comparison(before, db, after, wrong)
        wrong_summary = dict(after, incident_flux_per_s=2e10)
        with self.assertRaises(ValueError):
            paired_comparison(before, db, wrong_summary, da)

    def test_no_signal_contrast_remains_undefined_and_json_is_strict(self):
        dark = replica(np.zeros((4, 3)))
        summary, draws = statistics({"nominal": [dark]}, 4)
        self.assertIsNone(summary["scenarios"]["nominal"]["fringe_contrast"])
        self.assertIsNone(summary["scenarios"]["nominal"]["fringe_contrast_95ci"])
        self.assertEqual(summary["robust"]["worst_score"], 0)
        json.dumps(summary, allow_nan=False)
        paired = paired_comparison(summary, draws, summary, draws)
        self.assertIsNone(paired["nominal_contrast_improvement_95ci"])
        json.dumps(paired, allow_nan=False)

    def test_all_failed_launches_give_zero_flux_and_no_fabricated_efficiency(self):
        empty = replica(np.empty((0, 3)), ids=np.empty(0, dtype=int))
        summary, _ = statistics({"nominal": [empty]}, 3)
        s = summary["scenarios"]["nominal"]
        self.assertEqual(s["exit_flux_per_s"], 0)
        self.assertEqual(s["detected_flux_per_s"], 0)
        self.assertEqual(s["score"], 0)
        self.assertIsNone(s["preparation_efficiency"])
        self.assertFalse(summary["robust"]["feasible"])
        json.dumps(summary, allow_nan=False)

    def test_bad_ids_and_unphysical_weights_are_rejected(self):
        source = replica(np.tile([.5, .2, 0], (2, 1)))
        for key, value in (("atom_ids", [0, 0]), ("atom_ids", [0, 2]),
                           ("atom_ids", [.1, 1]), ("detection_weights", [1., -.1]),
                           ("target_population", [1., 1.1]), ("vz_m_s", [10., 0])):
            bad = dict(source, **{key: value})
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                statistics({"nominal": [bad]}, 2)


if __name__ == "__main__":
    unittest.main()
