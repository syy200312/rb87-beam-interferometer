import tempfile
import unittest
from pathlib import Path

import numpy as np

from rb87_beam.low_power_report import certify_family, paired_metrics, power_runs, write_report


def record(ids, photons, weights):
    n = len(ids)
    return dict(atom_ids=np.array(ids), target_population=np.ones(n),
                scattered_photons=np.asarray(photons), detection_weights=np.asarray(weights),
                fringe_coefficients=np.array([[.5, .1, 0.]] * n), vz_m_s=np.full(n, 10.))


class LowPowerReportTests(unittest.TestCase):
    def test_paired_photon_reduction_and_missing_incident_atom(self):
        ids = np.arange(31)
        before = record(ids, 3.0 + ids/7, np.ones(31))
        after = record(ids, 1.5 + ids/7, np.ones(31))
        row = paired_metrics([before, before], [after, after], 32, 1e5, 7e4,
                             bootstrap=250, retain_draws=True)
        certify_family([row], 7e4)
        self.assertEqual(row["candidate_flux_per_s"], 1e5*31/32)
        self.assertAlmostEqual(row["photon_saving_family_lo95"], 1.5)
        self.assertEqual(row["contrast_change"], 0.)
        self.assertTrue(row["advantage"])
        self.assertTrue(row["matched_flux_1pct"])
        self.assertFalse(row["baseline_advantage"])

    def test_saving_is_not_a_win_when_flux_fails(self):
        ids = np.arange(20)
        before = record(ids, np.full(20, 4), np.ones(20))
        after = record(ids, np.full(20, 2), np.full(20, .79))
        row = paired_metrics([before], [after], 20, 100., 80., bootstrap=128, retain_draws=True)
        certify_family([row], 80.)
        self.assertEqual(row["photon_saving"], 2.)
        self.assertFalse(row["advantage"])
        self.assertFalse(row["matched_flux_1pct"])

    def test_power_groups_do_not_bridge_missing_and_failed_nodes(self):
        rows = [dict(power_uW=p, length_mm=8, detuning_MHz=-1.5, advantage=a)
                for p, a in [(10, True), (20, False), (40, True), (80, True), (320, True)]]
        runs = power_runs(rows, [10,20,40,80,160,320])
        self.assertEqual([r["powers_uW"] for r in runs], [[10], [40,80], [320]])

    def test_no_supported_and_no_rows_are_reportable(self):
        ids = np.arange(8)
        rec = record(ids, np.full(8, 3.5), np.ones(8))
        row = paired_metrics([rec], [rec], 8, 100., 99., bootstrap=32, retain_draws=True)
        certify_family([row], 99.)
        settings = dict(power_uW=50, length_mm=4, detuning_MHz=-1.5)
        meta = dict(baseline=settings, incident_samples=8, replicas=1,
                    min_flux_per_s=99, incident_flux_per_s=100)
        with tempfile.TemporaryDirectory() as temp:
            r = write_report(temp, [dict(row, **settings)], [50], meta)
            self.assertFalse(r["candidate_advantage"])
            self.assertFalse(r["baseline_advantage"])
            self.assertTrue((Path(temp) / "comparison.png").is_file())
            self.assertIn("没有达到判据", (Path(temp) / "report.md").read_text(encoding="utf-8"))
            write_report(temp, [], [50], meta)
            self.assertFalse((Path(temp) / "comparison.png").exists())
