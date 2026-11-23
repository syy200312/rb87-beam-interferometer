"""Selection rules, physical transfer, motion/phase continuity and reproducibility."""

import copy
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from rb87_beam.config import SimulationConfig
from rb87_beam.io import read_stateful_ensemble, save_stateful_ensemble
from rb87_beam.physics import Ensemble, GAMMA, flight_time, trajectory, run_simulation
from rb87_beam.presets import pylcp_mot_preset
from rb87_beam.pumping import OpticalPumping
from rb87_beam.states import G_CLOCK, E_CLOCK


def configuration():
    c = pylcp_mot_preset()
    c.pumping.enabled = True
    c.pumping.recoil_model = "none"
    c.detection.noise = False
    c.raman.scan_points = 9
    return c


def incoming():
    rho = np.zeros((1, 8, 8), complex)
    rho[:, 3:, 3:] = np.eye(5) / 5
    return Ensemble(np.zeros((1, 3)), np.array([[.03, -.01, 15.]]), 1e8, ground_density=rho)


class OpticalPumpingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = configuration()
        cls.model = OpticalPumping(cls.cfg.pumping, cls.cfg.raman)
        cls.prepared = cls.model.prepare(incoming(), np.random.default_rng(87))

    def test_sigma_dark_state_and_pi_leakage_are_physical_selection_rules(self):
        m = self.model
        r = np.array([0, 0, m.options.center_z_m]) / m.x0
        q, rates = m.eqn.construct_evolution_matrix(r, np.zeros(3))
        np.testing.assert_allclose(q.sum(axis=0), 0, atol=2e-14)
        # Excited index zero is Fprime=0,m=0. No resonant target excitation.
        self.assertAlmostEqual(rates["g->e"][:, G_CLOCK, 0].sum(), 0, places=14)
        self.assertGreater(rates["g->e"][:, 0, 0].sum(), .01)
        # All 16 excited levels remain: off-resonant target scattering is nonzero.
        self.assertGreater(rates["g->e"][:, G_CLOCK, :].sum(), 1e-5)
        leaking = OpticalPumping(replace(m.options, pi_power_fraction=.01), self.cfg.raman)
        _, leak_rates = leaking.eqn.construct_evolution_matrix(r, np.zeros(3))
        self.assertGreater(leak_rates["g->e"][:, G_CLOCK, 0].sum(), 1e-3)

    def test_two_frequencies_transfer_population_without_reset_or_flux_creation(self):
        out = self.prepared
        self.assertGreater(out["diagnostics"]["preparation_efficiency"], .9)
        self.assertLess(out["diagnostics"]["preparation_efficiency"], .9999)
        self.assertEqual(out["ensemble"].flux_per_s, incoming().flux_per_s)
        self.assertGreater(out["scattered_photons"][0], 5)
        self.assertLess(out["final_populations"][0, 3:].sum(), .02)
        np.testing.assert_allclose(out["population_trace"].sum(axis=1), 1, atol=1e-8)
        self.assertGreater(out["population_trace"][-1, G_CLOCK], out["population_trace"][0, G_CLOCK])
        self.assertGreaterEqual(out["population_trace"].min(), -1e-9)

    def test_transfer_only_spreads_F1_population_and_pi_leakage_reduces_dark_accumulation(self):
        r = np.array([0, 0, self.cfg.pumping.center_z_m]) / self.model.x0
        initial = np.r_[np.zeros(3), np.ones(5) / 5, np.zeros(16)]
        values = []
        for opts in (self.cfg.pumping, replace(self.cfg.pumping, dark_power_W=0),
                     replace(self.cfg.pumping, pi_power_fraction=.01)):
            model = OpticalPumping(opts, self.cfg.raman)
            q, _ = model.eqn.construct_evolution_matrix(r, np.zeros(3))
            final = expm(q * .001 * GAMMA) @ initial
            ground = final[:8] + model.branching @ final[8:]
            values.append(ground[G_CLOCK])
        self.assertGreater(values[0], .9)
        self.assertLess(values[1], .6)
        self.assertLess(values[2], values[0] - .01)

    def test_no_light_preserves_coherence_ballistics_and_absolute_Raman_phase(self):
        cfg = copy.deepcopy(self.cfg)
        cfg.pumping.transfer_power_W = cfg.pumping.dark_power_W = 0
        cfg.raman.chirp_rad_s2 = 1e5
        source = incoming()
        source.ground_density[:] = 0
        amplitude = np.array([np.sqrt(.7), 1j * np.sqrt(.3)])
        source.ground_density[0][np.ix_([G_CLOCK, E_CLOCK], [G_CLOCK, E_CLOCK])] = np.outer(amplitude, amplitude.conj())
        output = OpticalPumping(cfg.pumping, cfg.raman).prepare(source, np.random.default_rng(1))["ensemble"]
        t = flight_time(output, cfg.detection.z_m)
        r1, v1 = trajectory(source, t, cfg.raman)
        r2, v2 = trajectory(output, t, cfg.raman)
        np.testing.assert_allclose(r1, r2, atol=2e-9)
        np.testing.assert_allclose(v1, v2, atol=1e-9)
        np.testing.assert_array_equal(output.ground_density, source.ground_density)
        cfg.pumping.enabled = False
        before, after = run_simulation(cfg, source), run_simulation(cfg, output)
        np.testing.assert_allclose(before["per_atom_P2"], after["per_atom_P2"], atol=3e-7)

    def test_checkpoint_replays_prepared_motion_states_and_fringe(self):
        cfg = copy.deepcopy(self.cfg)
        cfg.pumping.enabled = False
        prepared = self.prepared["ensemble"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prepared.h5"
            save_stateful_ensemble(prepared, path)
            restored = read_stateful_ensemble(path)
            np.testing.assert_array_equal(restored.reference_time_s, prepared.reference_time_s)
            self.assertEqual(restored.reference_plane_z_m, prepared.reference_plane_z_m)
            first, second = run_simulation(cfg, prepared), run_simulation(cfg, restored)
            np.testing.assert_array_equal(first["per_atom_P2"], second["per_atom_P2"])
            np.testing.assert_array_equal(first["detection"]["F2_counts"], second["detection"]["F2_counts"])

    def test_pumping_handoff_uses_real_density_and_can_be_followed_by_projection(self):
        cfg = copy.deepcopy(self.cfg)
        cfg.raman.state_handling = "project_F1_m0"
        cfg.raman.preparation_efficiency = 1
        result = run_simulation(cfg, incoming())
        target = result["ensemble"].ground_density[:, G_CLOCK, G_CLOCK].real.mean()
        self.assertAlmostEqual(result["detection"]["detected_flux_per_s"], incoming().flux_per_s * target)
        self.assertIsNotNone(result["pumping"])
        with self.assertRaises(ValueError):
            self.model.prepare(Ensemble(np.zeros((1, 3)), np.array([[0., 0., 15.]]), 1e8), np.random.default_rng(1))
        with self.assertRaises(ValueError):
            self.model.prepare(self.prepared["ensemble"], np.random.default_rng(1))

    def test_recoil_is_reproducible_and_changes_actual_exit_velocity(self):
        model = OpticalPumping(replace(self.cfg.pumping, recoil_model="diffusion"), self.cfg.raman)
        first = model.prepare(incoming(), np.random.default_rng(42))["ensemble"]
        second = model.prepare(incoming(), np.random.default_rng(42))["ensemble"]
        np.testing.assert_array_equal(first.v_m_s, second.v_m_s)
        self.assertGreater(np.linalg.norm(first.v_m_s - self.prepared["ensemble"].v_m_s), 1e-4)
        self.assertTrue(np.all(first.reference_time_s > 0))

    def test_population_solution_converges_with_tighter_steps_and_tolerances(self):
        opts = replace(self.cfg.pumping, max_step_s=1e-5, rtol=1e-8, population_atol=1e-12)
        output = OpticalPumping(opts, self.cfg.raman).prepare(incoming(), np.random.default_rng(87))
        np.testing.assert_allclose(output["final_populations"], self.prepared["final_populations"], atol=2e-6)

    def test_configuration_roundtrip_and_bad_region_or_polarization(self):
        restored = SimulationConfig.from_dict(self.cfg.to_dict())
        self.assertEqual(restored.pumping, self.cfg.pumping)
        for key, value in (("pi_power_fraction", 1.1), ("waist_z_m", .02),
                           ("bias_field_T", 0), ("transfer_power_W", -1), ("recoil_model", "jumps")):
            c = copy.deepcopy(self.cfg)
            setattr(c.pumping, key, value)
            with self.assertRaises(ValueError):
                c.validate()


if __name__ == "__main__":
    unittest.main()
