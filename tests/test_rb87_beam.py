"""Physical checks for the new baseline; run without importing legacy test scripts."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from rb87_beam import SimulationConfig, run_simulation
from rb87_beam.io import export_result, read_legacy_h5
from rb87_beam.physics import (Ensemble, KB, MASS, K_EFF, calibrated_source,
                               fit_fringe, fluorescence_scan, mot_scattering_rates,
                               pulse_step, raman_zone)


def mono_ensemble(vz=11.0, vx=0.0):
    return Ensemble(np.zeros((1, 3)), np.array([[vx, 0.0, vz]]), 1e6)


def ideal_config():
    c = SimulationConfig()
    c.detection.noise = False
    c.raman.acceleration_m_s2 = (0.0, 0.0, 0.0)
    c.raman.selection_sigma_vx_m_s = 0.0
    c.raman.scan_points = 17
    return c


class PhysicsChecks(unittest.TestCase):
    def test_unitary_and_dark_free_evolution(self):
        rng = np.random.default_rng(2)
        state = rng.normal(size=(100, 2)) + 1j * rng.normal(size=(100, 2))
        state /= np.linalg.norm(state, axis=1)[:, None]
        rate = rng.normal(0, 1e6, 100)
        out = pulse_step(state, 2e5, 0.4, rate, 2e-4)
        np.testing.assert_allclose(np.sum(abs(out) ** 2, axis=1), 1, atol=2e-14)
        dark = pulse_step(state, 0, 0.4, rate, 2e-4)
        np.testing.assert_allclose(dark, state, atol=2e-14)

    def test_resonant_rabi_and_detuned_transition(self):
        state = np.array([[1.0, 0.0]], dtype=complex)
        omega, duration, detuning = 2e4, 1e-4, 3e4
        out = pulse_step(state, omega, 0, detuning, duration)
        exact = omega ** 2 / (omega ** 2 + detuning ** 2) * np.sin(np.hypot(omega, detuning) * duration / 2) ** 2
        self.assertAlmostEqual(abs(out[0, 1]) ** 2, exact, places=13)
        out = pulse_step(state, omega, 0, 0, np.pi / omega)
        self.assertAlmostEqual(abs(out[0, 1]) ** 2, 1, places=13)

    def test_three_zone_ideal_mz_both_envelopes(self):
        c = ideal_config()
        for envelope in ("gaussian", "rectangular"):
            c.raman.envelope = envelope
            result = run_simulation(c, mono_ensemble())
            exact = 0.5 * (1 - np.cos(result["phase_rad"]))
            np.testing.assert_allclose(result["detection"]["observed_P2"], exact, atol=1e-10)
            self.assertLess(result["max_state_norm_error"], 1e-11)

    def test_acceleration_scale_and_rotation_sign(self):
        c = ideal_config()
        base = run_simulation(c, mono_ensemble())
        c.raman.acceleration_m_s2 = (0.02, 0, 0)
        accelerated = run_simulation(c, mono_ensemble())
        shift = np.angle(np.exp(1j * (accelerated["fit"]["phase_rad"] - base["fit"]["phase_rad"])))
        short_pulse = K_EFF * 0.02 * (c.raman.zone_spacing_m / 11) ** 2
        # Finite Gaussian pulses add ~1.2% here; this is a short-pulse comparison.
        self.assertLess(abs(shift / short_pulse - 1), 0.02)
        c.raman.acceleration_m_s2 = (0, 0, 0)
        c.raman.rotation_rad_s = (0, -0.02 / 22, 0)
        rotated = run_simulation(c, mono_ensemble())
        np.testing.assert_allclose(rotated["detection"]["observed_P2"],
                                   accelerated["detection"]["observed_P2"], atol=1e-10)

    def test_velocity_dependent_area(self):
        c = ideal_config()
        for vz in (8.0, 11.0, 15.0):
            result = raman_zone(np.array([[1., 0.]], dtype=complex), mono_ensemble(vz), c.raman, 0)
            exact = np.sin(np.pi / 4 * 11 / vz) ** 2
            self.assertAlmostEqual(abs(result[0, 1]) ** 2, exact, places=8)

    def test_reference_doppler_retuned_for_tilt(self):
        c = ideal_config()
        angle = 0.01
        c.raman.k_direction = (np.cos(angle), 0, -np.sin(angle))
        result = run_simulation(c, mono_ensemble())
        np.testing.assert_allclose(result["detection"]["observed_P2"],
                                   0.5 * (1 - np.cos(result["phase_rad"])), atol=1e-10)

    def test_friction_and_restoring_force(self):
        s = SimulationConfig().source
        rates = mot_scattering_rates(np.array([[0.0001, 0, 0]]), np.zeros((1, 3)), s)
        self.assertLess(rates[0, 0] - rates[0, 1], 0)
        rates = mot_scattering_rates(np.zeros((1, 3)), np.array([[0.1, 0, 0]]), s)
        self.assertLess(rates[0, 0] - rates[0, 1], 0)
        self.assertEqual(mot_scattering_rates(np.zeros((1, 3)), np.zeros((1, 3)), s)[0, 5], 0)

    def test_spectral_splitting_conserves_power(self):
        s = SimulationConfig().source
        r, v = np.array([[0.0001, 0, 0.01]]), np.array([[1., 0, 3.]])
        original = mot_scattering_rates(r, v, s)
        s.cooling_spectrum_gamma = (0, 0, 0)
        s.cooling_spectrum_weights = (0.2, 0.5, 0.3)
        np.testing.assert_allclose(mot_scattering_rates(r, v, s), original, rtol=1e-13)

    def test_empirical_temperature_and_fwhm(self):
        s = SimulationConfig().source
        s.n_atoms = 50000
        e = calibrated_source(s, np.random.default_rng(3))
        inferred_T = MASS * np.mean(np.var(e.v_m_s[:, :2], axis=0)) / KB
        self.assertLess(abs(inferred_T / s.calibrated_transverse_temperature_K - 1), 0.02)
        inferred_width = np.std(e.v_m_s[:, 2]) * np.sqrt(8 * np.log(2))
        self.assertLess(abs(inferred_width / s.calibrated_fwhm_vz_m_s - 1), 0.02)

    def test_fluorescence_residence_time_weighting(self):
        c = ideal_config()
        e = Ensemble(np.zeros((2, 3)), np.array([[0., 0, 10.], [0, 0, 20.]]), 1e6)
        result = fluorescence_scan(np.array([[0., 1.]]), e, np.ones(2), c.detection, c.raman, np.random.default_rng(1))
        self.assertAlmostEqual(result["expected_atom_P2"][0], 0.5)
        self.assertAlmostEqual(result["expected_photon_P2"][0], 1 / 3)
        self.assertAlmostEqual(result["observed_P2"][0], 1 / 3)

    def test_fringe_fit_phase_wrapping_and_zero_contrast(self):
        phases = np.linspace(0, 2 * np.pi, 41, endpoint=False)
        result = fit_fringe(phases, 0.4 + 0.1 * np.cos(phases + 0.7))
        self.assertAlmostEqual(result["phase_rad"], 0.7)
        self.assertAlmostEqual(result["contrast"], 0.25)
        self.assertIsNone(fit_fringe(phases, np.full_like(phases, 0.3))["phase_rad"])

    def test_seed_reproducibility_and_mot_pipeline(self):
        c = SimulationConfig()
        c.source.n_atoms = 100
        c.raman.scan_points = 5
        first, second = run_simulation(c), run_simulation(c)
        np.testing.assert_array_equal(first["detection"]["F2_counts"], second["detection"]["F2_counts"])
        self.assertGreater(len(first["ensemble"].r_m), 0)
        self.assertGreater(first["detection"]["expected_atoms_per_gate"], 0)

    def test_invalid_configuration(self):
        for key, value in (("n_atoms", 0), ("n_atoms", 5.5), ("dt_s", 0), ("cooling_saturation", -1)):
            c = SimulationConfig()
            setattr(c.source, key, value)
            with self.assertRaises(ValueError):
                c.validate()
        with self.assertRaises(ValueError):
            SimulationConfig.from_dict({"unknown": True})

    def test_time_slicing_converges(self):
        c = ideal_config()
        c.raman.two_photon_detuning_Hz = 5000
        values = []
        for steps in (24, 48, 96):
            c.raman.integration_steps = steps
            state = raman_zone(np.array([[1., 0.]], dtype=complex), mono_ensemble(), c.raman, 0)
            values.append(abs(state[0, 1]) ** 2)
        self.assertLess(abs(values[2] - values[1]), abs(values[1] - values[0]) / 2)
        self.assertLess(abs(values[2] - values[1]), 3e-4)

    def test_unprepared_spectators_reduce_fringe(self):
        c = ideal_config()
        c.raman.remove_unprepared = False
        c.raman.preparation_efficiency = 0.6
        c.raman.spectator_F2_fraction = 0.5
        result = run_simulation(c, mono_ensemble())
        self.assertAlmostEqual(result["fit"]["contrast"], 0.6, places=9)

    def test_hdf_unit_conversion_and_right_handed_axes(self):
        import h5py
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "source.h5"
            with h5py.File(path, "w") as handle:
                g = handle.create_group("sol_0")
                g.create_dataset("r", data=np.array([[0., 1.], [0., 2.], [0., 3.]]))
                g.create_dataset("v", data=np.array([[0., 0.], [0., 100.], [0., 0.]]))
            e = read_legacy_h5(path, 0.001, 0.01, 0.002, 1e6, "x,-z,y")
            np.testing.assert_allclose(e.r_m, [[0.001, -0.003, 0]])
            np.testing.assert_allclose(e.v_m_s, [[0, 0, 10]])
            with self.assertRaises(ValueError):
                read_legacy_h5(path, 0.001, 0.01, 0.002, 1e6, "xzy")

    def test_artifact_export(self):
        result = run_simulation(ideal_config(), mono_ensemble())
        with tempfile.TemporaryDirectory() as folder:
            export_result(result, folder, plot=False)
            self.assertTrue((Path(folder) / "summary.json").is_file())
            self.assertTrue((Path(folder) / "fringe.csv").is_file())


if __name__ == "__main__":
    unittest.main()
