"""Checks against the original finite-pulse chirp scan and direct state propagation."""

import copy
import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rb87_beam.chirp_noise_propagation import chirp_probabilities
from rb87_beam.chirp_noise_study import run
from rb87_beam.config import SimulationConfig
from rb87_beam.physics import Ensemble, K_EFF, populations_from_unitary, raman_zone
from rb87_beam.raman_noise_metrics import make_detector
from rb87_beam.raman_noise_study import NoiseSettings
from rb87_beam.scans import run_chirp_scan
from rb87_beam.states import E_CLOCK, G_CLOCK


def example():
    c = SimulationConfig()
    c.pumping.enabled = False
    c.detection.noise = False
    c.raman.integration_steps = 24
    c.raman.selection_sigma_vx_m_s = 0.
    c.raman.two_photon_detuning_Hz = 1700.
    c.raman.differential_light_shift_Hz = -400.
    c.raman.phase_offsets_rad = (.13, -.04, .38)
    # The configured base chirp is intentionally different from all scanned
    # values. The scan axis gives absolute slopes, not increments.
    c.raman.chirp_rad_s2 = 2*np.pi * 1.75e5
    c.chirp_scan.half_range_Hz_s = 6e5
    c.chirp_scan.points = 7
    positions = np.array([[0., 0., .002], [0., 4e-4, .002], [.02, 0., .002]])
    velocities = np.array([[.003, .01, 10.3], [-.002, 0., 11.7], [0., 0., 12.]])
    rho = np.zeros((3, 8, 8), dtype=complex)
    rho[:, G_CLOCK, G_CLOCK] = (.75, .55, .61)
    rho[:, E_CLOCK, E_CLOCK] = .15
    rho[:, 7, 7] = 1. - rho[:, G_CLOCK, G_CLOCK] - .15
    rho[:, G_CLOCK, E_CLOCK] = .03 + .02j
    rho[:, E_CLOCK, G_CLOCK] = .03 - .02j
    atoms = Ensemble(positions, velocities, 3e8, ground_density=rho, reference_plane_z_m=.002,
                     reference_time_s=np.array([.00019, .00021, .00018]))
    return c, atoms


def records():
    return {
        "phase": np.array([[0, 0, 0], [.02, -.08, .013], [-.03, .04, .05]]),
        "vibration": 1e-9 * np.array([[0, 0, 0], [-3., 1.1, .7], [1.4, -2., 3.]]),
        "magnetic": np.array([[0, 0, 0], [-.005, .003, .01], [.002, .006, -.007]]),
        "intensity": np.array([[1., 1., 1.], [1.019, .98, 1.012], [1.002, 1.015, .96]])}


def direct(ensemble, raman, alpha, draws, row, bias):
    r = copy.deepcopy(raman)
    r.chirp_rad_s2 = 2*np.pi*alpha
    b = draws["magnetic"][row]
    clock = 575.15 * (2*bias*b + b*b)
    phase = draws["phase"][row] - K_EFF * draws["vibration"][row]
    span = (2*r.envelope_half_widths*r.waist_z_m if r.envelope == "gaussian"
            else r.rectangular_width_m)
    dark = (r.zone_spacing_m - span) / ensemble.v_m_s[:, 2]
    state = np.zeros((len(ensemble.r_m), 2), complex)
    state[:, 0] = 1.
    for j in range(3):
        state = raman_zone(state, ensemble, r, j, scanned_phase=phase[j],
                           amplitude_scale=draws["intensity"][row, j], clock_shift_Hz=clock[j])
        if j < 2:
            half_angle = .5*np.pi * (clock[j] + clock[j+1]) * dark
            state *= np.exp(1j * half_angle[:, None] * np.array([1., -1.]))
    return populations_from_unitary(state[None], ensemble, r)[0][0]


class ChirpNoiseComparisonTests(unittest.TestCase):
    def test_absolute_chirp_matches_legacy_per_atom_and_aperture_weighted_signal(self):
        config, atoms = example()
        draws = {key: arr[:1] for key, arr in records().items()}
        reference = run_chirp_scan(config, atoms)
        detector = make_detector(atoms, config)
        self.assertAlmostEqual(detector.accepted_flux_per_s, 2e8)
        self.assertAlmostEqual(detector.target_flux_per_s, 1.30e8)
        photon_weights = detector.atom_counts * detector.q2
        for i, slope in enumerate(reference["chirp_Hz_s"]):
            nominal, noisy, _ = chirp_probabilities(atoms, config.raman, slope, draws, .1)
            np.testing.assert_allclose(nominal, reference["per_atom_P2"][i], rtol=0, atol=2e-11)
            np.testing.assert_allclose(noisy[0], nominal, rtol=0, atol=2e-11)
            mean = np.average(nominal, weights=photon_weights)
            self.assertAlmostEqual(mean, reference["detection"]["expected_photon_P2"][i], places=10)

    def test_chirped_noisy_transfer_matches_all_three_direct_pulses_and_dark_intervals(self):
        config, atoms = example()
        draws = records()
        for envelope, alpha in (("gaussian", 0.), ("rectangular", -6e5), ("gaussian", 6e5)):
            config.raman.envelope = envelope
            nominal, noisy, err = chirp_probabilities(atoms, config.raman, alpha, draws, .1,
                                                      check_indices=(1, 2))
            self.assertEqual(noisy.shape, (3, 3))
            np.testing.assert_allclose(nominal, noisy[0], atol=2e-11, rtol=0)
            expected = [direct(atoms, config.raman, alpha, draws, row, .1) for row in (1, 2)]
            np.testing.assert_allclose(noisy[1:], expected, rtol=0, atol=1e-5)
            self.assertLess(err, 1e-5)

    def test_replacing_chirp_with_third_pulse_phase_does_not_reproduce_atomic_response(self):
        c, atoms = example()
        alpha = 6e5
        draws = {key: value[:1].copy() for key, value in records().items()}
        true_chirp, _, _ = chirp_probabilities(atoms, c.raman, alpha, draws, .1)
        t_ref = c.raman.zone_spacing_m / c.raman.reference_vz_m_s
        draws["phase"][0, 2] = -2*np.pi*alpha*t_ref**2
        phase_only = direct(atoms, c.raman, 0., draws, 0, .1)
        self.assertGreater(np.max(np.abs(true_chirp - phase_only)), .005)

    def test_zero_technical_rms_leaves_counting_noise_in_both_scans(self):
        c, atoms = example()
        settings = NoiseSettings(phase_rms_rad=0., phase_zone_correlation=.35,
            mirror_displacement_rms_m=0., mirror_zone_correlation=.8, bias_field_G=.1,
            field_rms_G=0., field_zone_correlation=.5, fractional_power_rms=0.,
            log_power_zone_correlation=.65, gate_s=.01, record_period_s=.02,
            operating_phase_rad=4.7, rotation_step_rad_s=5e-7, records=32, scans=8,
            phase_points=11, seed=121)
        with tempfile.TemporaryDirectory() as folder:
            summary = run(c, atoms, settings, folder, workers=1, progress=None, plot=False)
            with (Path(folder)/"chirp_noise_fringes.csv").open(encoding="utf-8", newline="") as source:
                rows = list(csv.DictReader(source))
        self.assertEqual(len(rows), c.chirp_scan.points)
        self.assertAlmostEqual(summary["accepted_flux_per_s"], 2e8)
        self.assertAlmostEqual(summary["target_flux_per_s"], 1.30e8)
        self.assertAlmostEqual(summary["third_zone_phase_offset_rad"], c.raman.phase_offsets_rad[2])
        self.assertAlmostEqual(summary["baseline"]["visibility"], summary["combined"]["visibility"], places=10)
        for row in rows:
            self.assertAlmostEqual(float(row["combined_minus_baseline_P2"]), 0., places=10)
            for case in ("baseline", "combined"):
                self.assertEqual(int(row[case + "_valid_gates"]), settings.records)
                self.assertGreater(float(row[case + "_q975"]), float(row[case + "_q025"]))


if __name__ == "__main__":
    unittest.main()
