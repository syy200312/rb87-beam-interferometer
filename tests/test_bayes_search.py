"""Checks of the independent constrained Bayesian search surrogate."""

import json
import unittest

import numpy as np

from rb87_beam.bayes_search import GaussianProcess, constrained_expected_improvement


class BayesianSearchChecks(unittest.TestCase):
    def test_gp_recovers_smooth_held_out_function_and_mean_only(self):
        train = np.linspace(0, 1, 17)[:, None]
        y = np.sin(2 * np.pi * train[:, 0]) + .2 * train[:, 0]
        points = np.linspace(.02, .98, 80)[:, None]
        expected = np.sin(2 * np.pi * points[:, 0]) + .2 * points[:, 0]
        model = GaussianProcess().fit(train, y)
        mean, std = model.predict(points)
        self.assertLess(np.max(abs(mean - expected)), .025)
        self.assertTrue(np.all(std >= 0))
        np.testing.assert_array_equal(model.predict(points, return_std=False), mean)

    def test_gp_handles_repeated_points_and_inconsistent_noisy_observations(self):
        train = np.array([[0.], [.5], [.5], [1.]])
        y = np.array([0., 1.02, .98, 0.])
        mean, std = GaussianProcess(noise=1e-3).fit(train, y).predict(np.array([[.5], [.75]]))
        self.assertTrue(np.all(np.isfinite(mean)))
        self.assertTrue(np.all(np.isfinite(std)))
        self.assertAlmostEqual(mean[0], 1, delta=.01)
        self.assertTrue(np.all(std > 0))

    def test_constraints_downweight_high_objective_in_infeasible_region(self):
        train = np.array([[0.], [.2], [.4], [.6], [.8], [1.]])
        y = train[:, 0]
        margin = (.52 - train[:, 0])[:, None]
        pool = np.array([[.3], [.45], [.7], [.9]])
        index, diagnostic = constrained_expected_improvement(train, y, margin, pool)
        self.assertLess(pool[index, 0], .52)
        self.assertEqual(diagnostic['mode'], 'constrained_expected_improvement')
        self.assertGreater(diagnostic['probability_feasible'][index], .9)

    def test_no_feasible_start_prioritizes_feasibility_without_fabricating_incumbent(self):
        train = np.array([[0.], [.25], [.5]])
        y = -train[:, 0]
        margins = np.column_stack((train[:, 0] - .6, np.full(3, 1.)))
        pool = np.array([[.1], [.3], [.65], [.85], [1.]])
        index, diagnostic = constrained_expected_improvement(train, y, margins, pool)
        self.assertEqual(diagnostic['mode'], 'feasibility_first')
        self.assertIsNone(diagnostic['best_observed_feasible'])
        self.assertGreater(pool[index, 0], .5)
        self.assertAlmostEqual(diagnostic['probability_feasible'][index],
                               max(diagnostic['probability_feasible']))

    def test_observed_points_excluded_and_results_reproducible(self):
        train = np.array([[0., 0.], [.5, .5], [1., 1.]])
        y = np.array([0., 1., 0.])
        margin = np.ones((3, 1))
        pool = np.array([[.5, .5], [.5 + 1e-11, .5], [.25, .25], [.75, .75]])
        first, diagnostic = constrained_expected_improvement(train, y, margin, pool, seed=42)
        second, again = constrained_expected_improvement(train, y, margin, pool, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(diagnostic, again)
        self.assertIn(first, (2, 3))
        self.assertEqual(diagnostic['excluded_observed_indices'], [0, 1])
        json.dumps(diagnostic, allow_nan=False)

    def test_exhausted_pool_and_invalid_normalization_are_rejected(self):
        train = np.array([[0.], [1.]])
        with self.assertRaisesRegex(ValueError, 'exhausted'):
            constrained_expected_improvement(train, np.array([0., 1.]), np.ones((2, 1)), train)
        with self.assertRaisesRegex(ValueError, 'normalized'):
            GaussianProcess().fit(np.array([[0.], [2.]]), np.array([0., 1.]))
        with self.assertRaises(ValueError):
            GaussianProcess(noise=-1)

    def test_unconstrained_search_accepts_empty_margin_matrix(self):
        train = np.array([[0.], [.5], [1.]])
        pool = np.array([[.25], [.75]])
        index, diagnostic = constrained_expected_improvement(train, np.array([0., 1., 0.]),
                                                             np.empty((3, 0)), pool)
        self.assertIn(index, (0, 1))
        np.testing.assert_array_equal(diagnostic['probability_feasible'], 1)


if __name__ == '__main__':
    unittest.main()
