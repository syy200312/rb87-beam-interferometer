"""Power-derived Raman propagation, spin transport and scan export checks."""

import copy
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rb87_beam.config import SimulationConfig
from rb87_beam.io import read_stateful_ensemble
from rb87_beam.physics import (Ensemble, HBAR, K_EFF, MASS,
    raman_optical_frequency_offset, raman_peak_frequency, raman_power_calibration,
    raman_zone, run_simulation)
from rb87_beam.raman_calibration import calibrate_powers, solve_powers
from rb87_beam.scan_io import export_scan_result
from rb87_beam.scans import run_chirp_scan, run_rabi_scan
from rb87_beam.states import E_CLOCK, G_CLOCK, raman_frame_density


def power_config(detuning=0.):
    config = SimulationConfig()
    config.pumping.enabled = False
    r = config.raman
    r.coupling_model = "rb87_d2_power"
    r.peak_rabi_Hz = 0.
    r.waist_y_m, r.waist_z_m = .006, 18e-6
    r.first_zone_z_m, r.zone_spacing_m = .03, .01425
    r.reference_vz_m_s = 15.
    r.two_photon_detuning_Hz = detuning
    r.detuning_reference_Fprime = 1
    r.polarization1 = r.polarization2 = "sigma_plus"
    r.quantization_axis = r.k_direction
    r.spin_transport = "adiabatic_from_z"
    r.acceleration_m_s2 = (0., 0., 0.)
    r.state_handling, r.selection_sigma_vx_m_s = "inherit", 0.
    r.integration_steps, r.scan_points = 64, 9
    target = r.reference_vz_m_s / (4 * r.waist_z_m * np.sqrt(np.pi / 2))
    powers = solve_powers(target, r.waist_y_m, r.waist_z_m,
                         two_photon_detuning_Hz=raman_optical_frequency_offset(r))
    r.beam1_power_W, r.beam2_power_W = powers["power1_W"], powers["power2_W"]
    config.detection.noise = False
    config.rabi_scan.points = 9
    config.rabi_scan.area_max_pi = config.rabi_scan.transit_area_pi = 2.
    config.chirp_scan.points, config.chirp_scan.half_range_Hz_s = 9, 2e5
    config.validate()
    return config


def checkpoint(n=1, mixed=False):
    positions = np.zeros((n, 3))
    positions[:, 2] = .016
    velocity = np.zeros((n, 3))
    velocity[:, 2] = 15.
    density = np.zeros((n, 8, 8), complex)
    if mixed:
        density[:, G_CLOCK, G_CLOCK] = .6
        density[:, E_CLOCK, E_CLOCK] = .1
        density[:, 0, 0], density[:, 7, 7] = .1, .2
    else:
        density[:, G_CLOCK, G_CLOCK] = 1.
    return Ensemble(positions, velocity, 1e6,
        source_metadata=dict(sample_ids=list(range(n)), mode="provided_ensemble",
                             optical_pumping=dict(preparation_efficiency=.6 if mixed else 1.)),
        ground_density=density, reference_plane_z_m=.016,
        reference_time_s=.016 / velocity[:, 2])


def equivalent_effective(config):
    effective = copy.deepcopy(config)
    calibration = raman_power_calibration(config.raman)
    effective.raman.coupling_model = "effective"
    effective.raman.beam1_power_W = effective.raman.beam2_power_W = 0.
    effective.raman.spin_transport = "none"
    effective.raman.peak_rabi_Hz = calibration["peak_rabi_Hz"]
    effective.raman.differential_light_shift_Hz += calibration["differential_light_shift_Hz"]
    effective.raman.phase_offsets_rad = tuple(
        phase + calibration["coupling_phase_rad"] for phase in config.raman.phase_offsets_rad)
    return effective


class RamanPowerIntegrationTests(unittest.TestCase):
    def test_physical_powers_change_the_propagated_resonant_pulse_area(self):
        config, atoms = power_config(), checkpoint()
        initial = np.array([[1., 0.]], complex)
        one = raman_zone(initial, atoms, config.raman, 0)
        stronger = copy.deepcopy(config.raman)
        stronger.beam1_power_W *= 2
        stronger.beam2_power_W *= 2
        two = raman_zone(initial, atoms, stronger, 0)
        middle = raman_zone(initial, atoms, config.raman, 1)
        self.assertAlmostEqual(abs(one[0, 1]) ** 2, .5, delta=1e-7)
        self.assertAlmostEqual(abs(two[0, 1]) ** 2, 1., delta=1e-7)
        np.testing.assert_allclose(middle, two, atol=1e-10)
        self.assertAlmostEqual(raman_peak_frequency(stronger) / raman_peak_frequency(config.raman), 2.)

    def test_one_color_change_introduces_stark_detuning_and_middle_zone_doubles_it(self):
        config, atoms = power_config(), checkpoint()
        config.raman.beam1_power_W *= 2
        calibration = raman_power_calibration(config.raman)
        self.assertGreater(abs(calibration["differential_light_shift_Hz"]), 1e5)
        effective = equivalent_effective(config)
        initial = np.array([[1., 0.]], complex)
        for zone, factor in enumerate((1., 2., 1.)):
            expected = copy.deepcopy(effective.raman)
            # Legacy effective mode has one independently supplied shift per
            # zone; compare against the physical zone's explicitly doubled one.
            expected.differential_light_shift_Hz *= factor
            actual = raman_zone(initial, atoms, config.raman, zone)
            replay = raman_zone(initial, atoms, expected, zone)
            np.testing.assert_allclose(actual, replay, atol=2e-11)
        pure_coupling = copy.deepcopy(effective.raman)
        pure_coupling.differential_light_shift_Hz = 0.
        wrong = raman_zone(initial, atoms, pure_coupling, 0)
        correct = raman_zone(initial, atoms, config.raman, 0)
        self.assertGreater(abs(abs(wrong[0, 1]) ** 2 - abs(correct[0, 1]) ** 2), .05)

    def test_power_mode_requires_explicit_geometry_transport_and_one_coupling_source(self):
        config = power_config()
        for key, value in (("peak_rabi_Hz", 100e3), ("spin_transport", "none"),
                           ("quantization_axis", (0., 0., 1.)), ("envelope", "rectangular")):
            with self.subTest(key=key):
                invalid = copy.deepcopy(config)
                setattr(invalid.raman, key, value)
                with self.assertRaises(ValueError):
                    invalid.validate()

    def test_optical_carrier_offset_includes_recoil_and_reference_doppler(self):
        config = power_config(detuning=-25e3)
        r = config.raman
        r.k_direction = (np.sqrt(1 - .01 ** 2), 0., .01)
        r.quantization_axis = r.k_direction
        expected = (r.two_photon_detuning_Hz + HBAR * K_EFF ** 2 / (4 * np.pi * MASS)
                    + K_EFF * .01 * r.reference_vz_m_s / (2 * np.pi))
        self.assertAlmostEqual(raman_optical_frequency_offset(r), expected, places=10)
        calibration = raman_power_calibration(r)
        direct = calibrate_powers(r.beam1_power_W, r.beam2_power_W, r.waist_y_m, r.waist_z_m,
                                 two_photon_detuning_Hz=expected)
        self.assertEqual(calibration["peak_rabi_Hz"], direct["peak_rabi_Hz"])
        high, low = calibration["optical_frequency_offsets_Hz"]
        self.assertAlmostEqual(high - low - calibration["hyperfine_ground_splitting_Hz"], expected, delta=1e-6)

    def test_adiabatic_transport_preserves_mixture_and_never_changes_input_checkpoint(self):
        config, atoms = power_config(), checkpoint(mixed=True)
        original = copy.deepcopy(atoms)
        transported = raman_frame_density(atoms, config.raman)
        np.testing.assert_allclose(transported, original.ground_density, atol=1e-15)
        self.assertAlmostEqual(np.trace(transported[0] @ transported[0]).real, .42)
        result = run_rabi_scan(config, atoms)
        transfer = np.sin(np.pi * result["reference_area_pi"] / 2) ** 2
        np.testing.assert_allclose(result["per_atom_P2"][:, 0], .3 + .5 * transfer, atol=1e-7)
        populations = result["per_atom_ground_populations"]
        np.testing.assert_allclose(populations[:, 0, 0], .1, atol=1e-13)
        np.testing.assert_allclose(populations[:, 0, 7], .2, atol=1e-13)
        np.testing.assert_allclose(populations.sum(axis=2), 1., atol=1e-13)
        np.testing.assert_array_equal(atoms.ground_density, original.ground_density)
        np.testing.assert_array_equal(atoms.r_m, original.r_m)
        np.testing.assert_array_equal(atoms.v_m_s, original.v_m_s)
        np.testing.assert_array_equal(atoms.reference_time_s, original.reference_time_s)
        self.assertEqual(atoms.source_metadata, original.source_metadata)
        self.assertFalse(result["raman_spin_transport"]["reset_to_m0"])

    def test_phase_rabi_and_chirp_replay_match_the_same_effective_coupling(self):
        config, atoms = power_config(detuning=-25e3), checkpoint(n=2, mixed=True)
        atoms.r_m[:, 0] = [-.0002, .0002]
        atoms.r_m[:, 1] = [-.0001, .0001]
        atoms.v_m_s[:, 0] = [-.01, .015]
        atoms.v_m_s[:, 2] = [14.7, 15.4]
        atoms.reference_time_s = .016 / atoms.v_m_s[:, 2]
        effective = equivalent_effective(config)
        snapshot = config.to_dict()
        for run in (run_simulation, run_rabi_scan, run_chirp_scan):
            with self.subTest(experiment=run.__name__):
                actual, replay = run(config, atoms), run(effective, atoms)
                np.testing.assert_allclose(actual["per_atom_P2"], replay["per_atom_P2"], atol=2e-11)
                np.testing.assert_allclose(actual["detection"]["expected_photon_P2"],
                                           replay["detection"]["expected_photon_P2"], atol=2e-11)
                np.testing.assert_allclose(actual["per_atom_ground_populations"],
                                           replay["per_atom_ground_populations"], atol=2e-11)
        self.assertEqual(config.to_dict(), snapshot)

    def test_rabi_power_exports_are_milliwatts_and_reproduce_each_scan_point(self):
        config, atoms = power_config(), checkpoint(mixed=True)
        result = run_rabi_scan(config, atoms)
        for i, (p1, p2) in enumerate(zip(result["beam1_power_mW"], result["beam2_power_mW"])):
            r = copy.deepcopy(config.raman)
            r.beam1_power_W, r.beam2_power_W = p1 * 1e-3, p2 * 1e-3
            self.assertAlmostEqual(raman_peak_frequency(r), result["peak_rabi_Hz"][i], delta=1e-7)
        with tempfile.TemporaryDirectory() as directory:
            export_scan_result(result, directory, plot=False)
            with (Path(directory) / "rabi.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), config.rabi_scan.points)
            with np.load(Path(directory) / "scan.npz", allow_pickle=False) as saved:
                for key in ("beam1_power_mW", "beam2_power_mW", "reference_area_pi", "peak_rabi_Hz"):
                    np.testing.assert_allclose([float(row[key]) for row in rows], result[key], rtol=1e-14, atol=0.)
                    np.testing.assert_array_equal(saved[key], result[key])
            restored = read_stateful_ensemble(Path(directory) / "source_ensemble.h5")
            np.testing.assert_array_equal(restored.ground_density, atoms.ground_density)
            self.assertEqual(restored.reference_plane_z_m, .016)

    def test_actual_zone_and_scan_power_must_stay_in_the_elimination_regime(self):
        config, atoms = power_config(), checkpoint()
        initial = np.array([[1., 0.]], complex)
        base = raman_power_calibration(config.raman)
        # Choose a first-zone ratio of 0.08: first zone is valid, doubling its
        # optical powers in the middle zone raises the ratio above 0.1.
        scale = (.08 / base["max_single_photon_rabi_to_detuning_ratio"]) ** 2
        stronger = copy.deepcopy(config.raman)
        stronger.beam1_power_W *= scale
        stronger.beam2_power_W *= scale
        raman_zone(initial, atoms, stronger, 0)
        with self.assertRaisesRegex(ValueError, "Omega/Delta"):
            raman_zone(initial, atoms, stronger, 1)
        with self.assertRaisesRegex(ValueError, "Omega/Delta"):
            raman_zone(initial, atoms, config.raman, 0, amplitude_scale=100.)


if __name__ == "__main__":
    unittest.main()
