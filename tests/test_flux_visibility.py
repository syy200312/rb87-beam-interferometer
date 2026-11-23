import json
import unittest

import numpy as np

from rb87_beam.config import SimulationConfig
from rb87_beam.flux_visibility import make_visibility_snapshot, scan_flux_visibility
from rb87_beam.physics import Ensemble, fit_fringe, fluorescence_scan


class FluxVisibilityTests(unittest.TestCase):
    def setUp(self):
        self.cfg = SimulationConfig()
        self.d = self.cfg.detection
        self.d.noise = False
        self.x = 2 * np.pi * np.arange(16) / 16
        self.w = np.array([1.0, 0.8, 0.6, 0.0])
        self.vel = np.array([5.0, 11.0, 15.0, 20.0])
        self.p = 0.4 + 0.25 * np.cos(self.x[:, None] + np.array([0.2, 0.5, 1.0, -2.0])[None, :])
        self.ex = 0.8
        self.targ = np.array([0.98, 0.97, 0.99, 1.0])
        self.e = Ensemble(np.zeros((4, 3)), np.c_[np.zeros((4, 2)), self.vel], 1e8 * self.ex)
        self.s = make_visibility_snapshot(
            self.x, self.p, self.w, self.vel, self.d, self.ex, self.targ
        )

    def test_same_ideal_counts_and_fringe_as_detector(self):
        # Apples-to-apples: no geometrical loss in the underlying model; self.w
        # already includes selection. Detector fluorescence is weighted by 1/v.
        self.d.aperture_radius_m = 10
        old = fluorescence_scan(
            self.p, self.e, self.w, self.d, self.cfg.raman, np.random.default_rng(3)
        )
        sweep = scan_flux_visibility(self.s, [1e8, 1e5], repeats=32, seed=5)
        first, small = sweep["profiles"]
        np.testing.assert_allclose(sweep["ideal_P2"], old["expected_photon_P2"], atol=1e-13)
        np.testing.assert_allclose(first["expected_F1_counts"], old["F1_counts"], rtol=1e-12)
        np.testing.assert_allclose(first["expected_F2_counts"], old["F2_counts"], rtol=1e-12)
        self.assertEqual(first["ideal_contrast"], small["ideal_contrast"])
        self.assertAlmostEqual(
            first["effective_target_flux_per_s"], 1e8 * self.ex * np.mean(self.w * self.targ)
        )
        self.assertAlmostEqual(first["detected_flux_per_s"], old["detected_flux_per_s"])
        self.assertGreater(first["fit_success_rate"], 0.95)
        self.assertLess(
            first["fitted_q975"] - first["fitted_q025"], small["fitted_q975"] - small["fitted_q025"]
        )

    def test_atom_and_photon_shot_noise_not_pure_photon_poisson(self):
        # For a F2 compound Poisson process with gamma-distributed arrival
        # *types*, Var(raw)=sum_i lambda_i(q_i+q_i²)+bg+electronic_noise².
        self.p[:] = 0.5
        s = make_visibility_snapshot(self.x, self.p, self.w, self.vel, self.d, self.ex)
        self.d.aperture_radius_m = 10
        self.d.noise = True
        d = self.d
        phi = 2e4
        self.e.flux_per_s = phi * self.ex
        lam = self.ex * phi * d.gate_s / 4 * self.w * 0.5
        var = (
            np.sum(lam * (s.q2 + s.q2**2)) + d.background_counts + d.electronic_noise_std_counts**2
        )
        observed = []
        for seed in range(48):
            rr = scan_flux_visibility(s, [phi], repeats=1, seed=seed)["profiles"][0]
            observed.extend(rr["example_F2_counts"])
        self.assertLess(abs(np.var(observed, ddof=1) / var - 1), 0.19)

    def test_no_spurious_zero_contrast_for_missing_fringe(self):
        self.d.background_counts = self.d.electronic_noise_std_counts = 0.0
        s = make_visibility_snapshot(self.x, self.p, self.w, self.vel, self.d)
        result = scan_flux_visibility(s, [1e-12], repeats=24)
        point = result["profiles"][0]
        self.assertEqual(point["n_fits"], 0)
        self.assertIsNone(point["fitted_median"])
        self.assertTrue(all(x is None for x in point["example_P2"]))
        json.dumps(result, allow_nan=False)

    def test_per_flux_rng_does_not_depend_on_flux_grid_order(self):
        a = scan_flux_visibility(self.s, [1e4, 1e6], repeats=12, seed=87)["profiles"][0]
        b = scan_flux_visibility(self.s, [1e6, 1e4], repeats=12, seed=87)["profiles"][1]
        self.assertEqual(a, b)
        f = fit_fringe(self.x, np.array(a["example_P2"], dtype=float))
        self.assertAlmostEqual(f["contrast"], a["example_contrast"])


if __name__ == "__main__":
    unittest.main()
