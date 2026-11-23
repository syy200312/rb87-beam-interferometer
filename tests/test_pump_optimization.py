"""Independent sampling, exact fixed-axis acceleration and statistical convergence tools."""

import copy
import tempfile
import unittest
from dataclasses import replace

import numpy as np

from rb87_beam.batch import atom_rng, source_prefix, generate_source_batch
from rb87_beam.optimization import ballistic_screen, clustered_statistics, scan_candidates
from rb87_beam.physics import Ensemble, run_simulation
from rb87_beam.presets import pylcp_mot_preset
from rb87_beam.pumping import OpticalPumping, recoil_noise
from rb87_beam.states import G_CLOCK


class PumpOptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = pylcp_mot_preset()
        cls.cfg.pumping.enabled = True
        cls.cfg.pumping.recoil_model = "none"
        cls.cfg.detection.noise = False
        cls.cfg.raman.scan_points = 9
        cls.model = OpticalPumping(cls.cfg.pumping, cls.cfg.raman)

    def test_per_atom_random_stream_is_independent_of_order_and_batch_size(self):
        first = {i: atom_rng(123, 10, i).normal(size=12) for i in range(12)}
        second = {i: atom_rng(123, 10, i).normal(size=12) for i in reversed(range(24))}
        for i in first:
            np.testing.assert_array_equal(first[i], second[i])
        self.assertFalse(np.array_equal(first[0], first[1]))

    def test_native_source_cache_resumes_without_changing_existing_atoms(self):
        cfg = copy.deepcopy(self.cfg)
        cfg.source.mot_length_m = .0001
        cfg.source.initial_radius_m = 1e-6
        cfg.source.initial_temperature_K = 0
        cfg.source.initial_sigma_vz_m_s = 0
        cfg.source.max_time_s = .001
        with tempfile.TemporaryDirectory() as folder:
            first = generate_source_batch(cfg, 2, folder, workers=1, seed=101, progress=None)
            larger = generate_source_batch(cfg, 4, folder, workers=2, seed=101, progress=None)
            np.testing.assert_array_equal(first.r_m, larger.r_m[:2])
            np.testing.assert_array_equal(first.v_m_s, larger.v_m_s[:2])
            np.testing.assert_array_equal(first.ground_density, larger.ground_density[:2])
            self.assertEqual(larger.source_metadata["sample_ids"], [0, 1, 2, 3])
            cfg.source.cooling_detuning_gamma -= 1
            with self.assertRaises(ValueError):
                generate_source_batch(cfg, 4, folder, workers=1, seed=101, progress=None)

    def test_incident_prefix_counts_failed_launches_in_flux(self):
        e = Ensemble(np.zeros((3, 3)), np.tile((0, 0, 15.), (3, 1)), 6e8,
            {"sample_ids": [0, 2, 4], "incident_flux_per_s": 1e9, "sampled": 5})
        small = source_prefix(e, 3)
        self.assertEqual(len(small.r_m), 2)
        self.assertAlmostEqual(small.flux_per_s, 2e9 / 3)
        self.assertEqual(small.source_metadata["sample_ids"], [0, 2])

    def test_fixed_axis_generator_rates_and_force_match_native(self):
        native = self.model
        fast = OpticalPumping(replace(self.cfg.pumping, rate_backend="fixed_axis"), self.cfg.raman)
        rng = np.random.default_rng(1)
        for _ in range(15):
            r = np.r_[rng.normal(0, .001, 2), rng.uniform(.002, .026)] / native.x0
            v = np.r_[rng.normal(0, .4, 2), 15.] / native.v0
            q1, rates1 = native.eqn.construct_evolution_matrix(r, v)
            q2, rates2 = fast.eqn.construct_evolution_matrix(r, v)
            np.testing.assert_allclose(q1, q2, atol=2e-15, rtol=3e-14)
            np.testing.assert_allclose(rates1["g->e"], rates2["g->e"], atol=2e-15, rtol=3e-14)
            pop = rng.dirichlet(np.ones(24))
            np.testing.assert_allclose(native.eqn.force(r, 0, pop, False),
                                       fast.eqn.force(r, 0, pop, False), atol=1e-14)

    def test_ballistic_population_screen_matches_native_motion_without_transverse_force(self):
        cfg = copy.deepcopy(self.cfg)
        rho = np.zeros((1, 8, 8), complex)
        rho[0, 3:, 3:] = np.eye(5) / 5
        source = Ensemble(np.zeros((1, 3)), np.array([[0., 0., 15.]]), 1e8, ground_density=rho)
        screened = ballistic_screen(cfg, source)
        final = self.model.prepare(source, np.random.default_rng(2))
        np.testing.assert_allclose(screened["ground_populations"], final["final_populations"], atol=2e-6)
        self.assertLess(abs(screened["mean_scattered_photons"] - final["scattered_photons"][0]), .03)

    def test_poisson_recoil_has_zero_mean_and_expected_covariance(self):
        rng = np.random.default_rng(97)
        absorption, stimulated = np.array([.7, .4]), np.array([.04, .03])
        direction = np.array([[1., 0, 0], [-1., 0, 0]])
        spontaneous = 1.1
        noise = np.array([recoil_noise(rng, absorption, stimulated, direction, spontaneous, 1.,
                                     "compound_poisson") for _ in range(25000)])
        np.testing.assert_allclose(noise.mean(axis=0), 0, atol=.025)
        expected = np.array([absorption.sum() + stimulated.sum(), 0, 0]) + spontaneous / 3
        np.testing.assert_allclose(noise.var(axis=0), expected, rtol=.035)

    def test_bootstrap_uses_source_clusters_not_recoil_pseudoreplication(self):
        cfg = copy.deepcopy(self.cfg)
        cfg.pumping.enabled = False
        n = 12
        rho = np.zeros((n, 8, 8), complex)
        rho[:, G_CLOCK, G_CLOCK] = np.linspace(.8, .99, n)
        rho[:, 0, 0] = 1 - rho[:, G_CLOCK, G_CLOCK]
        e = Ensemble(np.zeros((n, 3)), np.tile((0, 0, 15.), (n, 1)), 1e9,
                     {"sample_ids": list(range(n))}, ground_density=rho)
        result = run_simulation(cfg, e)
        one, draw1 = clustered_statistics([(result, np.ones(n))], n, 1e9, bootstrap=100)
        three, draw3 = clustered_statistics([(result, np.ones(n))] * 3, n, 1e9, bootstrap=100)
        np.testing.assert_allclose(draw1, draw3, atol=1e-6)
        self.assertAlmostEqual(one["fringe_contrast"], result["expected_fit"]["contrast"], places=10)
        self.assertAlmostEqual(one["preparation_efficiency"], .895)
        np.testing.assert_allclose(one["preparation_efficiency_95ci"], three["preparation_efficiency_95ci"])

    def test_scan_grid_respects_physical_geometry_and_declared_power_ratios(self):
        candidates = scan_candidates(self.cfg)
        self.assertEqual(len(candidates), 588)
        for row in candidates:
            cfg = copy.deepcopy(self.cfg)
            cfg.pumping = type(cfg.pumping)(**row["pumping"])
            cfg.validate()
            self.assertLess(cfg.pumping.transfer_power_W + cfg.pumping.dark_power_W, .002)


if __name__ == "__main__":
    unittest.main()
