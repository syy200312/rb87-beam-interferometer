import tempfile
import unittest

from rb87_beam.low_power_matched import select_matched, write_matched


def row(family, power, flux, photons):
    return dict(family=family, power_uW=power, length_mm=4, detuning_MHz=0,
                flux_per_s=flux, photons=photons)


class MatchedFluxTests(unittest.TestCase):
    def test_selection_ignores_cheaper_flux_mismatch_and_applies_both_floors(self):
        anchor = row("3:1", 80, 100, 3.6)
        candidates = [row("1:1", 55, 99.69, 1.0), row("1:1", 60, 99.80, 2.0),
                      row("1:1", 65, 100.3, 3.0), row("1:1", 70, 100.0, 4.0)]
        self.assertEqual(select_matched(candidates, anchor, 99.9), candidates[2])
        self.assertIsNone(select_matched(candidates, anchor, 100.2))
        self.assertIsNone(select_matched(candidates[:1], anchor, 90))

    def test_training_match_is_not_held_out_evidence(self):
        anchor = row("3:1", 80, 100, 3.6)
        with tempfile.TemporaryDirectory() as out:
            result = write_matched(out, row("1:1", 85, 100, 4), anchor, None)
        self.assertIsNone(result["validation"])


if __name__ == "__main__":
    unittest.main()
