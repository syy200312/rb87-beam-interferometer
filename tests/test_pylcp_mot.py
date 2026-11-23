"""Physical regressions for native PyLCP optics, populations and state-preserving handoff."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
from scipy.linalg import expm
from scipy.special import jv

from rb87_beam import SimulationConfig, run_simulation
from rb87_beam.io import read_stateful_ensemble, save_stateful_ensemble, export_result
from rb87_beam.physics import Ensemble, GAMMA, HYPERFINE_HZ, HBAR, K, MASS, raman_zone, preparation_weights
from rb87_beam.pylcp_mot import MOT2DPlus, fm_spectrum
from rb87_beam.states import GROUND_LABELS, EXCITED_LABELS, G_CLOCK, E_CLOCK, rotation_to_lab_z


def config_for_states():
    config = SimulationConfig()
    config.detection.noise = False
    config.raman.acceleration_m_s2 = (0, 0, 0)
    config.raman.selection_sigma_vx_m_s = 0
    config.raman.scan_points = 9
    config.raman.state_handling = "inherit"
    return config


def ensemble_with_density(rho):
    rho = np.asarray(rho, dtype=complex)
    return Ensemble(np.zeros((len(rho), 3)), np.tile((0., 0., 11.), (len(rho), 1)), 1e6,
                    ground_density=rho)


class NativeMOTChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = SimulationConfig()
        cls.config.source.mode = "pylcp"
        cls.config.pylcp.recoil_model = "none"
        cls.model = MOT2DPlus(cls.config.source, cls.config.pylcp)

    def test_physical_hyperfine_intervals_basis_and_units(self):
        model = self.model
        splitting = (model.ground_energies[3] - model.ground_energies[0]) * GAMMA / (2 * np.pi)
        self.assertAlmostEqual(splitting / HYPERFINE_HZ, 1, places=8)
        self.assertEqual(model.hamiltonian.n, 24)
        self.assertAlmostEqual(model.v0 / model.mass_scaled, HBAR * K / MASS, places=13)
        np.testing.assert_allclose(model.eqn.decay_rates["g->e"], 1, atol=1e-13)

    def test_native_generator_is_conservative_and_respects_selection_rules(self):
        model = self.model
        q, _ = model.eqn.construct_evolution_matrix(np.array([1e-4, 0, .02]) / model.x0, np.zeros(3))
        np.testing.assert_allclose(q.sum(axis=0), 0, atol=1e-13)
        off_diagonal = q.copy()
        np.fill_diagonal(off_diagonal, 0)
        self.assertGreaterEqual(off_diagonal.min(), 0)
        np.testing.assert_allclose(model.branching.sum(axis=0), 1, atol=1e-13)
        self.assertTrue(np.all(model.dipole[:, :3, EXCITED_LABELS[:, 0] == 3] == 0))

    def test_hollow_return_and_gaussian_are_not_overwritten_by_base_class(self):
        model = self.model
        center = np.array([0., 0., .02]) / model.x0
        outside_hole = np.array([.001, 0., .02]) / model.x0
        returns = [(record, beam) for record, beam in zip(model.beam_records, model.beams.beam_vector)
                   if record["axis"] == 2 and record["sign"] == -1]
        self.assertGreater(len(returns), 0)
        for _, beam in returns:
            self.assertEqual(beam.intensity(center), 0)
            self.assertGreater(beam.intensity(outside_hole), 0)
        beam = model.beams.beam_vector[0]
        self.assertLess(beam.intensity(np.array([0., .01, .02]) / model.x0), beam.intensity(center))

    def test_both_transverse_axes_are_restoring(self):
        model = self.model
        for axis in (0, 1):
            position = np.array([0., 0., .02])
            position[axis] = 1e-4
            r = position / model.x0
            model.eqn.construct_evolution_matrix(r, np.zeros(3))
            force = model.eqn.force(r, 0, model.initial_populations(), return_details=False)
            self.assertLess(force[axis], 0)

    def test_repump_reduces_hyperfine_dark_population(self):
        options = replace(self.config.pylcp, repump_saturation=0, repump_axial_saturation=0)
        dark_model = MOT2DPlus(self.config.source, options)
        initial = np.r_[np.zeros(3), np.ones(5) / 5, np.zeros(16)]
        final = []
        for model in (self.model, dark_model):
            q, _ = model.eqn.construct_evolution_matrix(np.array([0., 0., .02]) / model.x0, np.zeros(3))
            populations = expm(q * .003 / model.t0) @ initial
            self.assertAlmostEqual(populations.sum(), 1, places=9)
            final.append(populations[:3].sum())
        self.assertGreater(final[1], .9)
        self.assertLess(final[0], final[1] / 100)

    def test_fm_bessel_weights_and_truncation(self):
        shift, power = fm_spectrum(3e6, .5, 3)
        n = np.arange(-3, 4)
        np.testing.assert_allclose(power, jv(n, .5) ** 2)
        np.testing.assert_allclose(shift * GAMMA / (2 * np.pi), n * 3e6, atol=1e-8)
        self.assertLess(power.sum(), 1)
        self.assertGreater(power.sum(), .999999)
        _, truncated = fm_spectrum(3e6, 2.0, 0)
        self.assertAlmostEqual(truncated.sum(), jv(0, 2.0) ** 2)

    def test_fm_optics_retain_total_power_and_repump_family(self):
        options = replace(self.config.pylcp, cooling_fm_frequency_Hz=3e6, cooling_fm_index=.5)
        model = MOT2DPlus(self.config.source, options)
        cooling_power = sum(record["power_saturation"] for record in model.beam_records if record["family"] == "cooling")
        self.assertAlmostEqual(cooling_power, 4 * self.config.source.cooling_saturation * model.spectra["cooling"]["retained_power_fraction"])
        self.assertTrue(any(record["family"] == "repump" for record in model.beam_records))

    def test_free_decay_branching_and_quantization_rotation(self):
        model = self.model
        population = np.zeros(24)
        population[-1] = 1  # Fprime=3,m=+3 -> F=2,m=+2 only
        rho, axis = model.ground_after_decay(population, np.array([0., 0., .04]))
        self.assertAlmostEqual(rho[-1, -1].real, 1)
        np.testing.assert_allclose(axis, (0, 0, 1))
        rotation = rotation_to_lab_z((1, 0, 0))
        np.testing.assert_allclose(rotation.conj().T @ rotation, np.eye(8), atol=1e-14)
        self.assertAlmostEqual(abs(rotation[G_CLOCK, G_CLOCK]) ** 2, 0, places=12)
        self.assertAlmostEqual(abs(rotation[E_CLOCK, E_CLOCK]) ** 2, .25, places=12)

    def test_native_source_exit_and_stateful_checkpoint_replay(self):
        config = config_for_states()
        config.source.mode, config.source.n_atoms = "pylcp", 2
        config.source.initial_temperature_K = 0
        config.source.initial_radius_m = 1e-6
        config.source.initial_sigma_vz_m_s = 0
        config.source.mot_length_m = .001
        config.source.max_time_s = .003
        config.pylcp.recoil_model = "none"
        result = run_simulation(config)
        ensemble = result["ensemble"]
        self.assertEqual(len(ensemble.r_m), 2)
        self.assertEqual(ensemble.ground_density.shape, (2, 8, 8))
        self.assertEqual(ensemble.exit_populations.shape, (2, 24))
        self.assertLess(result["max_state_norm_error"], 1e-11)
        np.testing.assert_allclose(result["per_atom_ground_populations"].sum(axis=2), 1, atol=1e-11)
        with tempfile.TemporaryDirectory() as directory:
            export_result(result, directory, plot=False)
            path = Path(directory) / "source_ensemble.h5"
            loaded = read_stateful_ensemble(path)
            np.testing.assert_array_equal(loaded.ground_density, ensemble.ground_density)
            np.testing.assert_array_equal(loaded.exit_time_s, ensemble.exit_time_s)
            replay = run_simulation(config, loaded)
            np.testing.assert_array_equal(replay["detection"]["F2_counts"], result["detection"]["F2_counts"])
            with h5py.File(path, "r+") as handle:
                handle["ground_labels_F_mF"][0] = (2, 2)
            with self.assertRaises(ValueError):
                read_stateful_ensemble(path)

    def test_existing_module_exposes_upgraded_source(self):
        from MOT_ALL import MOT2DPlus_module
        self.assertIs(MOT2DPlus_module, MOT2DPlus)


class RamanStateTransferChecks(unittest.TestCase):
    def test_mixture_is_preserved_without_a_pure_state_reset(self):
        rho = np.zeros((4, 8, 8), dtype=complex)
        for i, index in enumerate((G_CLOCK, E_CLOCK, 0, 7)):
            rho[i, index, index] = 1
        ensemble = ensemble_with_density(rho)
        config = config_for_states()
        result = run_simulation(config, ensemble)
        ideal = .5 * (1 - np.cos(result["phase_rad"]))
        np.testing.assert_allclose(result["per_atom_P2"][:, 0], ideal, atol=1e-10)
        np.testing.assert_allclose(result["per_atom_P2"][:, 1], 1 - ideal, atol=1e-10)
        np.testing.assert_allclose(result["per_atom_P2"][:, 2], 0, atol=1e-10)
        np.testing.assert_allclose(result["per_atom_P2"][:, 3], 1, atol=1e-10)
        np.testing.assert_allclose(preparation_weights(ensemble, config.raman), 1)
        np.testing.assert_allclose(result["per_atom_ground_populations"].sum(axis=2), 1, atol=1e-12)

    def test_clock_coherence_matches_direct_wavefunction_propagation(self):
        config = config_for_states()
        amplitudes = np.array([np.sqrt(.3), 1j * np.sqrt(.7)])
        rho = np.zeros((1, 8, 8), dtype=complex)
        rho[0][np.ix_([G_CLOCK, E_CLOCK], [G_CLOCK, E_CLOCK])] = np.outer(amplitudes, amplitudes.conj())
        ensemble = ensemble_with_density(rho)
        result = run_simulation(config, ensemble)
        state = amplitudes[None, :]
        for zone in (0, 1):
            state = raman_zone(state, ensemble, config.raman, zone)
        expected = [abs(raman_zone(state, ensemble, config.raman, 2, phase)[0, 1]) ** 2 for phase in result["phase_rad"]]
        np.testing.assert_allclose(result["per_atom_P2"][:, 0], expected, atol=1e-11)

    def test_state_projection_loses_flux_instead_of_creating_prepared_atoms(self):
        config = config_for_states()
        config.raman.state_handling = "project_F1_m0"
        rho = np.zeros((1, 8, 8), dtype=complex)
        rho[0, G_CLOCK, G_CLOCK], rho[0, 7, 7] = .1, .9
        ensemble = ensemble_with_density(rho)
        result = run_simulation(config, ensemble)
        self.assertAlmostEqual(result["detection"]["detected_flux_per_s"], ensemble.flux_per_s * .1 * .8)
        np.testing.assert_allclose(result["per_atom_P2"][:, 0], .5 * (1 - np.cos(result["phase_rad"])), atol=1e-10)
        np.testing.assert_allclose(result["per_atom_ground_populations"].sum(axis=2), 1, atol=1e-12)

    def test_invalid_density_and_missing_state_are_rejected(self):
        config = config_for_states()
        with self.assertRaises(ValueError):
            run_simulation(config, Ensemble(np.zeros((1, 3)), np.array([[0., 0., 11.]]), 1e6))
        rho = np.zeros((1, 8, 8), dtype=complex)
        rho[0, 0, 0], rho[0, 1, 1] = 1.1, -.1
        with self.assertRaises(ValueError):
            ensemble_with_density(rho).validate()


if __name__ == "__main__":
    unittest.main()
