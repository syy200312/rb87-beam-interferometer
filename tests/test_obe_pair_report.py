import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rb87_beam.obe_pair_report import summarize_obe_pairs, write_obe_report


def record(ids, target=.98, acceptance=1., scatter=4., duplicates=1):
    n = len(ids)
    val = np.ones((duplicates,n))
    return dict(atom_ids=np.array(ids), target_population=val*target,
                accepted_target=val*target*acceptance, scattered_photons=val*scatter)


def case(pair, sc, ref, new):
    return dict(pair=pair, scenario=sc, baseline=ref, candidate=new)


class ObePairedStatisticsTest(unittest.TestCase):
    def test_both_ratios_and_joint_decision_including_counterexample(self):
        ids = np.arange(32)
        good = record(ids, target=.98, scatter=3)
        ref = record(ids, target=.979, scatter=5)
        # Ratio differs by >0.3% but <1%; inject the same atom in both groups.
        other = record(ids, target=.987, scatter=4)
        cases = [case(p, s, ref, c) for p in ("matched80","winner55")
                 for s,c in (("nominal",good),("high",other))]
        records, summary = summarize_obe_pairs(cases,32,100.,95.,bootstrap=128)
        self.assertAlmostEqual(records[0]["flux_ratio"],.979/.98)
        self.assertAlmostEqual(records[2]["flux_ratio"],.98/.979)
        self.assertAlmostEqual(records[0]["photon_saving_sim_lo95"],2)
        self.assertEqual(summary["pairs"]["matched80"]["joint_status"],"excluded")
        self.assertEqual(summary["pairs"]["matched80"]["excluded_scenarios"],["high"])
        self.assertEqual(summary["pairs"]["winner55"]["joint_status"],"supported")
        self.assertEqual(summary["pairs"]["winner55"]["worst"]["flux_ratio"]["scenario"],"high")

    def test_uneven_extraction_and_per_replica_normalization(self):
        ids = np.delete(np.arange(64), [4,12,50,54])
        b = record(ids, target=1., acceptance=.8, scatter=5 + ids/10.)
        a = record(ids[::-1], target=1., acceptance=.8, scatter=4 + ids[::-1]/10., duplicates=3)
        a["scattered_photons"] += np.array([[-2.],[0.],[2.]])
        r, s = summarize_obe_pairs([case("matched80","nominal",b,a)],64,100.,50.,bootstrap=256)
        self.assertEqual(s["extracted_samples"],60)
        self.assertAlmostEqual(r[0]["baseline_efficiency"], 1.)
        self.assertAlmostEqual(r[0]["baseline_flux_per_s"], 75.)
        self.assertAlmostEqual(r[0]["baseline_photons"], 5+ids.mean()/10.)
        self.assertAlmostEqual(r[0]["photon_saving_sim_lo95"],1.)
        self.assertEqual(r[0]["joint_status"],"supported")

    def test_missing_undefined_ratio_is_unresolved_not_certified(self):
        ids = np.arange(4)
        rec = record(ids, target=0., scatter=3.)
        r, s = summarize_obe_pairs([case("winner55","nominal",rec,rec)],4,100.,95.,bootstrap=32)
        self.assertIsNone(r[0]["flux_ratio_sim_hi95"])
        self.assertEqual(r[0]["flux_ratio_status"],"unresolved")
        self.assertEqual(s["pairs"]["winner55"]["joint_status"],"excluded")
        self.assertFalse(r[0]["joint_point"])

    def test_mismatched_ids_replicates_or_scenarios_may_not_be_silently_paired(self):
        b = record([0, 1, 3])
        c = record([0, 1, 2])
        with self.assertRaisesRegex(ValueError,"source ID sets"):
            summarize_obe_pairs([case("matched80","nominal",b,c)],4,100.,90.)
        with self.assertRaisesRegex(ValueError,"same declared"):
            summarize_obe_pairs([case("matched80","nominal",b,b), case("winner55","high",b,b)],
                                4,100.,90.,bootstrap=16)
        wrong = copy.deepcopy(b)
        wrong["accepted_target"][0,0] = .999
        with self.assertRaisesRegex(ValueError,"clock-state"):
            summarize_obe_pairs([case("matched80","nominal",b,wrong)],4,100.,90.,bootstrap=16)

    def test_export_single_pair_and_negative_saving(self):
        ids = np.arange(16)
        b, c = record(ids, scatter=4.), record(ids, scatter=5.)
        with tempfile.TemporaryDirectory() as temp:
            rows, summary = write_obe_report(temp, [case("matched80","nominal",b,c)],16,
                                             100.,95.,bootstrap=32)
            self.assertTrue((Path(temp)/"report.md").is_file())
            self.assertTrue((Path(temp)/"obe_summary.json").is_file())
            self.assertLess(rows[0]["photon_reduction_sim_hi95"],0)
            self.assertAlmostEqual(rows[0]["retained_fraction_of_prior"],-.25/.079)
            self.assertEqual(summary["pairs"]["matched80"]["excluded_scenarios"],["nominal"])
