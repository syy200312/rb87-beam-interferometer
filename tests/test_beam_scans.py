"""Independent analytic checks for beam Rabi and chirp-rate experiments."""

import copy
import unittest

import numpy as np

from rb87_beam.config import SimulationConfig
from rb87_beam.physics import Ensemble, K_EFF, run_simulation
from rb87_beam.scans import run_chirp_scan, run_rabi_scan
from rb87_beam.states import E_CLOCK, G_CLOCK


def scan_config():
    config = SimulationConfig()
    config.pumping.enabled = False
    config.raman.state_handling = "ideal_preparation"
    config.raman.preparation_efficiency = 1.0
    config.raman.selection_sigma_vx_m_s = 0.0
    config.raman.acceleration_m_s2 = (0.0, 0.0, 0.0)
    config.raman.envelope = "rectangular"
    config.raman.integration_steps = 48
    config.raman.scan_points = 9
    config.detection.noise = False
    config.rabi_scan.points = 13
    config.chirp_scan.points = 17
    return config


def mono_ensemble(velocities=(11.0,), density=None):
    velocities = np.asarray(velocities)
    velocity = np.column_stack((np.zeros(len(velocities)), np.zeros(len(velocities)), velocities))
    return Ensemble(np.zeros_like(velocity), velocity, 1e6, ground_density=density)


def instantaneous_pulses(config, ensemble, chirp_Hz_s):
    """Closed-form zero-width pulses, with actual transit times and velocity-scaled areas."""
    out = []
    for alpha in chirp_Hz_s:
        state = np.zeros((len(ensemble.r_m), 2), dtype=complex)
        state[:, 0] = 1
        for j, nominal_area in enumerate((np.pi / 2, np.pi, np.pi / 2)):
            time = (config.raman.first_zone_z_m + j * config.raman.zone_spacing_m) / ensemble.v_m_s[:, 2]
            phi = (-np.pi * alpha * time ** 2
                   - 2 * np.pi * config.raman.two_photon_detuning_Hz * time
                   + config.raman.phase_offsets_rad[j])
            area = nominal_area * config.raman.reference_vz_m_s / ensemble.v_m_s[:, 2]
            cosine, sine = np.cos(area / 2), np.sin(area / 2)
            g = cosine * state[:, 0] - 1j * sine * np.exp(-1j * phi) * state[:, 1]
            e = cosine * state[:, 1] - 1j * sine * np.exp(1j * phi) * state[:, 0]
            state = np.column_stack((g, e))
        out.append(abs(state[:, 1]) ** 2)
    return np.asarray(out)


class BeamScanPhysicsChecks(unittest.TestCase):
    def test_resonant_area_scan_matches_sine_squared_for_both_envelopes(self):
        config = scan_config()
        for envelope in ("rectangular", "gaussian"):
            with self.subTest(envelope=envelope):
                config.raman.envelope = envelope
                result = run_rabi_scan(config, mono_ensemble())
                expected = np.sin(np.pi * result["reference_area_pi"] / 2) ** 2
                np.testing.assert_allclose(result["per_atom_P2"][:, 0], expected, atol=3e-8)
                np.testing.assert_allclose(result["detection"]["expected_photon_P2"], expected, atol=3e-8)
                self.assertLess(result["max_state_norm_error"], 1e-11)

    def test_detuned_rectangular_scan_matches_generalized_rabi_solution(self):
        config = scan_config()
        config.raman.two_photon_detuning_Hz = 5000.0
        result = run_rabi_scan(config, mono_ensemble())
        duration = config.raman.rectangular_width_m / config.raman.reference_vz_m_s
        omega = np.pi * result["reference_area_pi"] / duration
        delta = 2 * np.pi * config.raman.two_photon_detuning_Hz
        generalized = np.hypot(omega, delta)
        expected = (omega / generalized) ** 2 * np.sin(generalized * duration / 2) ** 2
        np.testing.assert_allclose(result["per_atom_P2"][:, 0], expected, atol=1e-11)
        self.assertEqual(result["per_atom_P2"][0, 0], 0.0)

    def test_mixed_clock_population_and_spectators_survive_rabi_scan(self):
        config = scan_config()
        config.raman.state_handling = "inherit"
        rho = np.zeros((1, 8, 8), dtype=complex)
        rho[0, G_CLOCK, G_CLOCK], rho[0, E_CLOCK, E_CLOCK] = .6, .1
        rho[0, 0, 0], rho[0, 7, 7] = .1, .2
        result = run_rabi_scan(config, mono_ensemble(density=rho))
        transferred = np.sin(np.pi * result["reference_area_pi"] / 2) ** 2
        np.testing.assert_allclose(result["per_atom_P2"][:, 0], .3 + .5 * transferred, atol=1e-11)
        populations = result["per_atom_ground_populations"]
        np.testing.assert_allclose(populations[:, 0, 0], .1, atol=1e-12)
        np.testing.assert_allclose(populations[:, 0, 7], .2, atol=1e-12)
        np.testing.assert_allclose(populations.sum(axis=2), 1, atol=1e-12)
        self.assertAlmostEqual(result["detection"]["detected_flux_per_s"], 1e6)

    def test_initial_clock_coherence_is_propagated_without_reset(self):
        config = scan_config()
        config.raman.state_handling = "inherit"
        amplitudes = np.array([np.sqrt(.4), 1j * np.sqrt(.6)])
        rho = np.zeros((1, 8, 8), dtype=complex)
        rho[0][np.ix_([G_CLOCK, E_CLOCK], [G_CLOCK, E_CLOCK])] = np.outer(amplitudes, amplitudes.conj())
        result = run_rabi_scan(config, mono_ensemble(density=rho))
        angle = np.pi * result["reference_area_pi"] / 2
        exact = abs(np.cos(angle) * amplitudes[1] - 1j * np.sin(angle) * amplitudes[0]) ** 2
        np.testing.assert_allclose(result["per_atom_P2"][:, 0], exact, atol=1e-11)

    def test_transit_trace_ends_at_the_corresponding_area_scan_result(self):
        config = scan_config()
        result = run_rabi_scan(config, mono_ensemble())
        trace = result["transit"]
        times = np.asarray(trace["reference_elapsed_s"])
        self.assertEqual(times[0], 0.0)
        self.assertTrue(np.all(np.diff(times) > 0))
        duration = config.raman.rectangular_width_m / config.raman.reference_vz_m_s
        self.assertAlmostEqual(times[-1], duration)
        target = trace["reference_area_pi"]
        ids = np.flatnonzero(np.isclose(result["reference_area_pi"], target))
        self.assertEqual(len(ids), 1)
        expected = result["detection"]["expected_photon_P2"][ids[0]]
        self.assertAlmostEqual(trace["expected_photon_P2"][-1], expected, places=11)
        self.assertAlmostEqual(trace["expected_atom_P2"][0], 0.0, places=11)

    def test_short_pulse_chirp_period_units_and_acceleration_compensation(self):
        config = scan_config()
        config.raman.rectangular_width_m = 1e-6
        config.raman.acceleration_m_s2 = (.02, 0.0, 0.0)
        time = config.raman.zone_spacing_m / config.raman.reference_vz_m_s
        compensation = K_EFF * .02 / (2 * np.pi)
        config.chirp_scan.center_Hz_s = compensation
        config.chirp_scan.half_range_Hz_s = 1 / time ** 2
        snapshot = copy.deepcopy(config.to_dict())
        result = run_chirp_scan(config, mono_ensemble())
        np.testing.assert_allclose(result["chirp_rad_s2"], 2 * np.pi * result["chirp_Hz_s"], rtol=1e-14)
        self.assertAlmostEqual(result["reference_T_s"], time)
        self.assertAlmostEqual(result["reference_chirp_Hz_s"], compensation)
        phase = (K_EFF * .02 - 2 * np.pi * result["chirp_Hz_s"]) * time ** 2
        expected = .5 * (1 - np.cos(phase))
        np.testing.assert_allclose(result["per_atom_P2"][:, 0], expected, atol=3e-4)
        self.assertLess(result["per_atom_P2"][len(expected) // 2, 0], 1e-7)
        self.assertEqual(config.to_dict(), snapshot)

    def test_velocity_distribution_uses_each_atoms_flight_times_and_pulse_areas(self):
        config = scan_config()
        config.raman.rectangular_width_m = 1e-7
        config.chirp_scan.points = 9
        config.chirp_scan.half_range_Hz_s = 1e6
        ensemble = mono_ensemble((11.0, 13.0))
        result = run_chirp_scan(config, ensemble)
        expected = instantaneous_pulses(config, ensemble, result["chirp_Hz_s"])
        np.testing.assert_allclose(result["per_atom_P2"], expected, atol=6e-5)
        self.assertGreater(np.max(abs(expected[:, 0] - expected[:, 1])), .05)

    def test_downstream_checkpoint_reference_time_keeps_chirp_clock_consistent(self):
        config = scan_config()
        config.raman.rectangular_width_m = 1e-6
        config.chirp_scan.points = 9
        config.chirp_scan.half_range_Hz_s = 1e6
        original = mono_ensemble((11.0, 13.0))
        downstream = copy.deepcopy(original)
        downstream.reference_plane_z_m = .01
        downstream.r_m[:, 2] = .01
        downstream.reference_time_s = .01 / downstream.v_m_s[:, 2]
        downstream.exit_time_s = np.array([.2, .4])  # MOT residence is not Raman flight time.
        first, second = run_chirp_scan(config, original), run_chirp_scan(config, downstream)
        np.testing.assert_allclose(first["per_atom_P2"], second["per_atom_P2"], atol=1e-11)

    def test_horizontal_raman_axis_has_zero_gravity_chirp_compensation(self):
        config = scan_config()
        config.raman.acceleration_m_s2 = (0.0, -9.80665, 0.0)
        config.chirp_scan.points = 9
        config.chirp_scan.half_range_Hz_s = 1e6
        result = run_chirp_scan(config, mono_ensemble())
        self.assertEqual(result["reference_chirp_Hz_s"], 0.0)

    def test_existing_phase_scan_is_not_changed_by_new_experiment_settings(self):
        config = scan_config()
        before = run_simulation(config, mono_ensemble())
        config.rabi_scan.area_max_pi = 8.0
        config.rabi_scan.transit_area_pi = 6.0
        config.chirp_scan.center_Hz_s = 12345.0
        config.chirp_scan.half_range_Hz_s = 1e5
        after = run_simulation(config, mono_ensemble())
        np.testing.assert_array_equal(before["per_atom_P2"], after["per_atom_P2"])
        self.assertEqual(before["fit"], after["fit"])

    def test_finite_gaussian_pulses_converge_with_detuning_and_nonzero_chirp(self):
        config = scan_config()
        config.raman.envelope = "gaussian"
        config.raman.two_photon_detuning_Hz = 3500.0
        config.chirp_scan.points = 9
        config.chirp_scan.center_Hz_s = .7e6
        config.chirp_scan.half_range_Hz_s = 1e6
        values = []
        for steps in (32, 64, 128):
            config.raman.integration_steps = steps
            values.append(run_chirp_scan(config, mono_ensemble())["per_atom_P2"][:, 0])
        coarse = np.linalg.norm(values[1] - values[0])
        fine = np.linalg.norm(values[2] - values[1])
        self.assertLess(fine, coarse / 3)  # second-order midpoint convergence
        self.assertLess(np.max(abs(values[2] - values[1])), .001)


if __name__ == "__main__":
    unittest.main()
