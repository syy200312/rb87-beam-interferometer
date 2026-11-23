"""Physical budget, shared-source and acquisition-objective checks for co-design."""

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rb87_beam.batch import subset_ensemble
from rb87_beam.joint_optimization import (
    BOUNDS, _source_digest, apply_scenario, assert_independent_sources,
    ballistic_response, decode, encode, engineering_margins, evaluate_screen,
    first_peak_Hz, native_cases, parameters, validation_subset)
from rb87_beam.joint_statistics import case_statistics
from rb87_beam.physics import Ensemble, raman_zone
from rb87_beam.presets import pylcp_mot_preset
from rb87_beam.scans import run_rabi_scan


def baseline_config():
    config = pylcp_mot_preset()
    config.pumping.enabled = True
    config.pumping.transfer_power_W = 60e-6
    config.pumping.dark_power_W = 20e-6
    config.pumping.transfer_detuning_Hz = config.pumping.dark_detuning_Hz = 1.5e6
    config.pumping.waist_z_m = .001
    config.pumping.rate_backend = "fixed_axis"
    config.pumping.diffusion_quadrature_order = 3
    config.pumping.recoil_chunk_s = 25e-6
    config.raman.peak_rabi_Hz = first_peak_Hz(config.raman)
    config.raman.acceleration_m_s2 = (0., 0., 0.)
    config.raman.selection_sigma_vx_m_s = 0.
    config.raman.integration_steps = 48
    config.raman.scan_points = 9
    config.detection.noise = False
    return config


def source_ensemble(n=2):
    density = np.zeros((n, 8, 8), complex)
    density[:, 3:, 3:] = np.eye(5) / 5
    velocity = np.column_stack((np.linspace(-.02, .02, n), np.zeros(n), np.linspace(15., 15.2, n)))
    return Ensemble(np.zeros((n, 3)), velocity, 1e9,
                    {"sample_ids": list(range(n)), "mode": "provided_ensemble"}, ground_density=density)


def study_settings():
    return dict(scenarios=[dict(name="nominal")], native_raman_steps=32,
        incident_flux_per_s=1e9,
        constraints=dict(min_efficiency=0., min_effective_target_flux_per_s=1.,
                         max_scattered_photons=20., min_fringe_contrast=0.,
                         max_pump_power_W=160e-6, max_raman_peak_Hz=500e3,
                         max_relative_raman_power=1.5, min_rayleigh_m=.001))


class JointOptimizationChecks(unittest.TestCase):
    def test_explicit_peak_keeps_intensity_fixed_when_zone_width_changes(self):
        config = baseline_config()
        config.pumping.enabled = False
        config.raman.peak_rabi_Hz = 100e3
        config.raman.state_handling = "ideal_preparation"
        velocity = config.raman.reference_vz_m_s
        ensemble = Ensemble(np.zeros((1, 3)), np.array([[0., 0., velocity]]), 1e6)
        initial = np.array([[1., 0.]], complex)
        probabilities = []
        for width in (25e-6, 50e-6):
            config.raman.waist_z_m = width
            for zone in (0, 1, 2):
                state = raman_zone(initial, ensemble, config.raman, zone)
                area = 2 * np.pi * 100e3 * (1, 2, 1)[zone] * width * np.sqrt(np.pi / 2) / velocity
                self.assertAlmostEqual(abs(state[0, 1]) ** 2, np.sin(area / 2) ** 2, delta=2e-8)
            probabilities.append(abs(raman_zone(initial, ensemble, config.raman, 0)[0, 1]) ** 2)
        self.assertGreater(abs(probabilities[1] - probabilities[0]), .3)

    def test_area_calibrated_rabi_scan_remains_correct_with_explicit_peak(self):
        config = baseline_config()
        config.pumping.enabled = False
        config.raman.peak_rabi_Hz = 173e3
        config.raman.state_handling = "ideal_preparation"
        config.raman.preparation_efficiency = 1.
        config.rabi_scan.points = 13
        ensemble = Ensemble(np.zeros((1, 3)), np.array([[0., 0., 15.]]), 1e6)
        result = run_rabi_scan(config, ensemble)
        expected = np.sin(np.pi * result["reference_area_pi"] / 2) ** 2
        np.testing.assert_allclose(result["per_atom_P2"][:, 0], expected, atol=3e-8)
        width = config.raman.waist_z_m * np.sqrt(np.pi / 2)
        expected_frequency = result["reference_area_pi"] * 15 / (2 * width)
        np.testing.assert_allclose(result["peak_rabi_Hz"], expected_frequency, atol=1e-8)

    def test_zero_peak_preserves_legacy_area_calibration(self):
        config = baseline_config()
        config.raman.peak_rabi_Hz = 0.
        ensemble = Ensemble(np.zeros((1, 3)), np.array([[0., 0., 15.]]), 1e6)
        initial = np.array([[1., 0.]], complex)
        for width in (20e-6, 40e-6):
            config.raman.waist_z_m = width
            state = raman_zone(initial, ensemble, config.raman, 0)
            self.assertAlmostEqual(abs(state[0, 1]) ** 2, .5, delta=3e-9)

    def test_decode_respects_relative_gaussian_power_and_keeps_fixed_experiment(self):
        baseline = baseline_config()
        original = baseline.to_dict()
        values = parameters(baseline, baseline)
        values.update(raman_waist_um=35., raman_relative_power=1.2)
        candidate = decode(encode(values), baseline)
        ratio = first_peak_Hz(candidate.raman) * candidate.raman.waist_z_m / (
            first_peak_Hz(baseline.raman) * baseline.raman.waist_z_m)
        self.assertAlmostEqual(ratio, 1.2)
        for name in ("zone_spacing_m", "first_zone_z_m", "reference_vz_m_s",
                     "selection_sigma_vx_m_s", "state_handling", "k_direction"):
            self.assertEqual(getattr(candidate.raman, name), getattr(baseline.raman, name))
        self.assertEqual(candidate.detection, baseline.detection)
        self.assertEqual(candidate.source, baseline.source)
        self.assertEqual(baseline.to_dict(), original)
        self.assertEqual(candidate.raman.state_handling, "inherit")
        self.assertEqual(candidate.raman.selection_sigma_vx_m_s, 0.)

    def test_peak_budget_covers_middle_zone_and_power_budget_covers_width(self):
        baseline = baseline_config()
        values = parameters(baseline, baseline)
        values.update(raman_relative_power=1.2, raman_waist_um=35.)
        candidate = decode(encode(values), baseline)
        constraints = study_settings()["constraints"]
        constraints["max_raman_peak_Hz"] = 1.5 * first_peak_Hz(candidate.raman)
        constraints["max_relative_raman_power"] = 1.1
        margin = engineering_margins(candidate, baseline, constraints)
        self.assertLess(first_peak_Hz(candidate.raman), constraints["max_raman_peak_Hz"])
        self.assertLess(margin[1], 0.)  # Middle pi zone needs twice the first-zone peak.
        self.assertLess(margin[2], 0.)  # Increased width/peak product exceeds the power budget.

    def test_decode_bounds_keep_pumping_and_raman_regions_separate(self):
        baseline = baseline_config()
        for point in (np.zeros(len(BOUNDS)), np.ones(len(BOUNDS))):
            candidate = decode(point, baseline)
            candidate.validate()
            end = candidate.pumping.center_z_m + candidate.pumping.half_widths * candidate.pumping.waist_z_m
            beginning = candidate.raman.first_zone_z_m - candidate.raman.envelope_half_widths * candidate.raman.waist_z_m
            self.assertLess(end, beginning)
            self.assertLess(candidate.pumping.transfer_power_W + candidate.pumping.dark_power_W, 160e-6)
        with self.assertRaises(ValueError):
            decode(np.full(len(BOUNDS), 1.01), baseline)

    def test_scenario_changes_incoming_speed_before_pumping_without_mutating_inputs(self):
        config = baseline_config()
        source = source_ensemble()
        original_config, original_velocity = config.to_dict(), source.v_m_s.copy()
        original_density, original_digest = source.ground_density.copy(), _source_digest(source)
        perturbed, atoms = apply_scenario(config, source, dict(name="drift", source_vz_factor=.98,
            pump_power_factor=.95, raman_power_factor=1.05, pump_frequency_shift_Hz=-.2e6,
            raman_frequency_shift_Hz=1e4))
        self.assertEqual(atoms.reference_plane_z_m, 0.)
        np.testing.assert_allclose(atoms.v_m_s[:, 2], original_velocity[:, 2] * .98)
        np.testing.assert_array_equal(atoms.ground_density, original_density)
        self.assertNotEqual(_source_digest(atoms), original_digest)
        self.assertAlmostEqual(perturbed.pumping.transfer_power_W, config.pumping.transfer_power_W * .95)
        self.assertAlmostEqual(perturbed.raman.peak_rabi_Hz, config.raman.peak_rabi_Hz * 1.05)
        np.testing.assert_array_equal(source.v_m_s, original_velocity)
        self.assertEqual(config.to_dict(), original_config)
        self.assertEqual(_source_digest(source), original_digest)
        atoms.reference_plane_z_m = .016
        atoms.r_m[:, 2] = .016
        atoms.reference_time_s = .016 / atoms.v_m_s[:, 2]
        with self.assertRaisesRegex(ValueError, "MOT-exit"):
            apply_scenario(config, atoms, dict(name="already_prepared"))

    def test_low_acceptance_high_contrast_does_not_beat_large_modulated_signal(self):
        constraints = study_settings()["constraints"]
        def summary(offset, amplitude):
            record = dict(atom_ids=np.arange(2), fringe_coefficients=np.tile((offset, amplitude, 0.), (2, 1)),
                          detection_weights=np.ones(2), vz_m_s=np.full(2, 15.),
                          target_population=np.full(2, .98), scattered_photons=np.full(2, 3.6))
            return case_statistics({"nominal": [record]}, 2, 1e9, constraints, bootstrap=20)[0]
        broad, narrow = summary(.4, .2), summary(.01, .009)
        self.assertGreater(narrow["scenarios"]["nominal"]["fringe_contrast"],
                           broad["scenarios"]["nominal"]["fringe_contrast"])
        self.assertLess(narrow["robust"]["worst_score"], broad["robust"]["worst_score"])
        self.assertEqual(narrow["robust"]["min_detected_flux_per_s"], broad["robust"]["min_detected_flux_per_s"])

    def test_speed_and_angular_divergence_vary_independently_before_pumping(self):
        config, source = baseline_config(), source_ensemble(5)
        source.v_m_s[:, 0] = [-.18, .04, .23, .09, -.07]
        source.v_m_s[:, 1] = [.13, -.04, .16, -.11, .02]
        source.r_m[:, 0] = np.linspace(-.001, .001, 5)
        original = copy.deepcopy(source)
        angles = original.v_m_s[:, :2] / original.v_m_s[:, 2, None]
        center, spread = angles.mean(axis=0), angles.std(axis=0)
        for speed, divergence in ((1., .8), (.94, 1.), (.94, 1.2), (1.06, .8)):
            with self.subTest(speed=speed, divergence=divergence):
                _, atoms = apply_scenario(config, source, dict(name="beam_envelope",
                    source_vz_factor=speed, source_divergence_factor=divergence))
                changed_angles = atoms.v_m_s[:, :2] / atoms.v_m_s[:, 2, None]
                np.testing.assert_allclose(changed_angles.mean(axis=0), center, rtol=1e-13)
                np.testing.assert_allclose(changed_angles.std(axis=0), divergence * spread, rtol=1e-13)
                np.testing.assert_allclose(changed_angles,
                    center + divergence * (angles - center), rtol=1e-13)
                np.testing.assert_allclose(atoms.v_m_s[:, 2], speed * original.v_m_s[:, 2])
                np.testing.assert_array_equal(atoms.r_m, original.r_m)
                np.testing.assert_array_equal(atoms.ground_density, original.ground_density)
                self.assertEqual(atoms.source_metadata, original.source_metadata)
                self.assertEqual(atoms.flux_per_s, original.flux_per_s)
                self.assertEqual(atoms.reference_plane_z_m, 0.)
        np.testing.assert_array_equal(source.v_m_s, original.v_m_s)
        np.testing.assert_array_equal(source.r_m, original.r_m)
        np.testing.assert_array_equal(source.ground_density, original.ground_density)
        self.assertEqual(_source_digest(source), _source_digest(original))

    def test_divergence_identity_preserves_cache_and_speed_only_keeps_legacy_behavior(self):
        config, source = baseline_config(), source_ensemble(5)
        source.v_m_s[:, 1] = [.17, -.12, .06, .02, -.04]
        original = _source_digest(source)
        for scenario in (dict(name="nominal"), dict(name="nominal", source_vz_factor=1.,
                                                   source_divergence_factor=1.)):
            _, atoms = apply_scenario(config, source, scenario)
            np.testing.assert_array_equal(atoms.v_m_s, source.v_m_s)
            self.assertEqual(_source_digest(atoms), original)
        _, legacy = apply_scenario(config, source, dict(name="speed_only", source_vz_factor=.93))
        np.testing.assert_array_equal(legacy.v_m_s[:, :2], source.v_m_s[:, :2])
        np.testing.assert_array_equal(legacy.v_m_s[:, 2], .93 * source.v_m_s[:, 2])
        angles = source.v_m_s[:, :2] / source.v_m_s[:, 2, None]
        np.testing.assert_allclose(legacy.v_m_s[:, :2] / legacy.v_m_s[:, 2, None], angles / .93)
        self.assertEqual(_source_digest(source), original)

    def test_source_envelope_factors_must_be_positive_finite_scalars(self):
        config, source = baseline_config(), source_ensemble()
        for key in ("source_vz_factor", "source_divergence_factor"):
            for value in (0., -1., np.nan, np.inf, -np.inf, None, [1.], True):
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                    apply_scenario(config, source, dict(name="invalid", **{key: value}))

    def test_ballistic_joint_screen_preserves_real_populations_and_source(self):
        config = baseline_config()
        source = source_ensemble()
        original = _source_digest(source)
        record = ballistic_response(config, source)
        self.assertTrue(np.all(record["target_population"] > 0))
        self.assertTrue(np.all(record["target_population"] < 1 - 1e-6))
        self.assertTrue(np.all(record["scattered_photons"] > 0))
        np.testing.assert_array_equal(record["detection_weights"], 1)
        settings = study_settings()
        settings["scenarios"].append(dict(name="beam_slow", source_vz_factor=.98))
        summary, margins = evaluate_screen(config, source, config, settings)
        self.assertEqual(set(summary["scenarios"]), {"nominal", "beam_slow"})
        self.assertTrue(np.all(np.isfinite(margins)))
        self.assertGreater(summary["robust"]["worst_score"], 0.)
        self.assertEqual(_source_digest(source), original)

    def test_native_cache_reuses_same_pumping_for_different_raman_designs(self):
        config = baseline_config()
        source = source_ensemble(1)
        alternate = copy.deepcopy(config)
        alternate.raman.peak_rabi_Hz *= .8
        settings = study_settings()
        cases = {"baseline": config, "raman_only": alternate}
        with tempfile.TemporaryDirectory() as folder:
            before, _ = native_cases(cases, source, config, folder, settings,
                incident_samples=1, replicas=1, workers=1, seed=97, bootstrap=20, progress=None)
            cache = list((Path(folder) / "pump_cache").glob("*/replica_0/atom_*.npz"))
            self.assertEqual(len(cache), 1)
            content = cache[0].read_bytes()
            after, _ = native_cases(cases, source, config, folder, settings,
                incident_samples=1, replicas=1, workers=1, seed=97, bootstrap=20, progress=None)
            self.assertEqual(cache[0].read_bytes(), content)
            self.assertEqual(before, after)
            self.assertEqual(before["baseline"]["scenarios"]["nominal"]["preparation_efficiency"],
                             before["raman_only"]["scenarios"]["nominal"]["preparation_efficiency"])
            self.assertNotAlmostEqual(before["baseline"]["robust"]["worst_score"],
                                      before["raman_only"]["robust"]["worst_score"], places=6)
            # A larger source has a new whole-source digest, while atom 0 has
            # the same physical entry state and must retain its recoil path.
            old_folders = set((Path(folder) / "pump_cache").iterdir())
            larger = source_ensemble(2)
            np.testing.assert_array_equal(source.v_m_s[0], larger.v_m_s[0])
            self.assertNotEqual(_source_digest(source), _source_digest(larger))
            events = []
            native_cases(cases, larger, config, folder, settings, incident_samples=2,
                         replicas=1, workers=1, seed=97, bootstrap=20, progress=events.append)
            self.assertIn("1 missing", next(message for message in events if "missing" in message))
            extended_folders = set((Path(folder) / "pump_cache").iterdir()) - old_folders
            self.assertEqual(len(extended_folders), 1)
            extended = extended_folders.pop() / "replica_0"
            self.assertEqual((extended / "atom_000000.npz").read_bytes(), content)
            self.assertTrue((extended / "atom_000001.npz").is_file())
            with np.load(extended / "atom_000000.npz", allow_pickle=False) as original_atom:
                old_input = str(original_atom["input_digest"])
            # Reusing the numerical ID is insufficient after its input changes.
            changed = copy.deepcopy(larger)
            changed.v_m_s[0, 0] += .015
            before_changed = set((Path(folder) / "pump_cache").iterdir())
            events = []
            native_cases(cases, changed, config, folder, settings, incident_samples=2,
                         replicas=1, workers=1, seed=97, bootstrap=20, progress=events.append)
            self.assertIn("1 missing", next(message for message in events if "missing" in message))
            changed_folders = set((Path(folder) / "pump_cache").iterdir()) - before_changed
            self.assertEqual(len(changed_folders), 1)
            changed_folder = changed_folders.pop() / "replica_0"
            with np.load(changed_folder / "atom_000000.npz", allow_pickle=False) as changed_atom:
                self.assertNotEqual(str(changed_atom["input_digest"]), old_input)
            # Unmodified atom 1 is still eligible for warm reuse.
            self.assertEqual((changed_folder / "atom_000001.npz").read_bytes(),
                             (extended / "atom_000001.npz").read_bytes())

    def test_same_source_overlapping_subsets_cannot_be_declared_independent(self):
        source = source_ensemble(3)
        training = subset_ensemble(source, [0, 2])
        validation = subset_ensemble(source, [1, 2])
        validation.source_metadata["sample_ids"] = [100, 101]  # Renaming IDs cannot remove overlap.
        self.assertNotEqual(_source_digest(training), _source_digest(validation))
        with self.assertRaisesRegex(ValueError, "share source atom trajectories"):
            assert_independent_sources(training, validation)
        independent = copy.deepcopy(validation)
        independent.v_m_s[:, 0] += .001
        assert_independent_sources(training, independent)

    def test_validation_subset_counts_failed_launches_but_rejects_unsimulated_atoms(self):
        source = source_ensemble(3)
        source.flux_per_s = .6e9
        source.source_metadata.update(sample_ids=[0, 2, 4], sampled=5, incident_flux_per_s=1e9)
        subset = validation_subset(source, 3)
        self.assertEqual(subset.source_metadata["sample_ids"], [0, 2])
        self.assertAlmostEqual(subset.flux_per_s, 2e9 / 3)
        self.assertAlmostEqual(validation_subset(source, 5).flux_per_s, .6e9)
        with self.assertRaisesRegex(ValueError, "more launches than were simulated"):
            validation_subset(source, 6)


if __name__ == "__main__":
    unittest.main()
