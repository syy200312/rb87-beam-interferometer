"""Control-group selection and declared-domain checks for endpoint co-design."""

import copy
import itertools
import unittest

import numpy as np

from rb87_beam.endpoint_comparison import (
    _prefix_data, raman_library, scenario_grid, select_methods)
from rb87_beam.joint_optimization import BOUNDS


def constraints():
    return dict(min_efficiency=.97, min_effective_target_flux_per_s=9.5e9,
                max_scattered_photons=6.)


def candidate(case, family, efficiency, photons, lower, point,
              flux=9.7e9, feasible=True):
    return dict(case=case, family=family,
        validation=dict(robust=dict(min_efficiency=efficiency,
            min_effective_target_flux_per_s=flux,
            max_scattered_photons=photons, feasible=feasible)),
        endpoint=dict(robust=dict(worst_readout_snr_per_rad=point,
                                 worst_readout_snr_per_rad_95ci=[lower, point + 10.])))


def library():
    return [candidate("A0", "higher_efficiency", .985, 5.5, 18., 40.),
            candidate("A1", "higher_efficiency", .985, 5.5, 20., 30.),
            candidate("B0", "lower_scattering", .975, 3.5, 17., 100.),
            candidate("B1", "lower_scattering", .975, 3.5, 25., 28.)]


class EndpointComparisonTests(unittest.TestCase):
    def test_both_sequential_rules_reoptimize_raman_using_the_common_endpoint_objective(self):
        rows = library()
        original = copy.deepcopy(rows)
        selected = select_methods(rows, constraints())
        self.assertEqual(selected["sequential_efficiency"]["case"], "A1")
        self.assertEqual(selected["sequential_scattering"]["case"], "B1")
        self.assertEqual(selected["joint"]["case"], "B1")
        # The noisy high point estimate B0 must not beat the better lower bound.
        self.assertNotEqual(selected["joint"]["case"], "B0")
        self.assertEqual(rows, original)

    def test_stage_a_pump_choice_is_independent_of_terminal_response(self):
        rows = library()
        first = select_methods(rows, constraints())
        for row in rows:
            # Reverse the relative end-signal ranking of the two pump families.
            score = 60. if row["family"] == "higher_efficiency" else 2.
            row["endpoint"]["robust"] = dict(worst_readout_snr_per_rad=score + 1.,
                                            worst_readout_snr_per_rad_95ci=[score, score + 2.])
        second = select_methods(rows, constraints())
        for method in ("sequential_efficiency", "sequential_scattering"):
            self.assertEqual(first[method]["family"], second[method]["family"])
        self.assertNotEqual(first["joint"]["family"], second["joint"]["family"])

    def test_stage_a_respects_flux_and_scattering_budgets_before_raman_ranking(self):
        rows = library() + [
            candidate("excess_scatter", "excess_scatter", .999, 6.1, 100., 110., feasible=False),
            candidate("starved", "starved", .999, .1, 200., 210., flux=9.4e9, feasible=False),
            candidate("bad_preparation", "bad_preparation", .969, .01, 300., 310., feasible=False)]
        selected = select_methods(rows, constraints())
        self.assertEqual(selected["sequential_efficiency"]["family"], "higher_efficiency")
        self.assertEqual(selected["sequential_scattering"]["family"], "lower_scattering")
        self.assertEqual(selected["joint"]["case"], "B1")

    def test_failed_second_stage_cannot_silently_change_the_stage_a_choice(self):
        rows = library()
        for row in rows:
            if row["family"] == "higher_efficiency":
                row["validation"]["robust"]["feasible"] = False
        with self.assertRaisesRegex(RuntimeError, "Stage A selected higher_efficiency"):
            select_methods(rows, constraints())

    def test_infeasible_pump_library_cannot_relax_the_common_constraints(self):
        rows = library()
        for row in rows:
            row["validation"]["robust"]["max_scattered_photons"] = 6.1
            row["validation"]["robust"]["feasible"] = False
        with self.assertRaisesRegex(RuntimeError, "No pump family"):
            select_methods(rows, constraints())

    def test_validation_grid_has_all_crossed_speed_divergence_pairs_and_one_nominal(self):
        scenarios = scenario_grid()
        self.assertEqual(len(scenarios), 9)
        self.assertEqual(len({row["name"] for row in scenarios}), 9)
        self.assertEqual({(row["source_vz_factor"], row["source_divergence_factor"])
                          for row in scenarios},
                         set(itertools.product((.95, 1., 1.05), (.9, 1., 1.1))))
        nominal = [row for row in scenarios if row["name"] == "nominal"]
        self.assertEqual(len(nominal), 1)
        self.assertEqual(nominal[0]["source_vz_factor"], 1.)
        self.assertEqual(nominal[0]["source_divergence_factor"], 1.)

    def test_shared_raman_library_is_a_unique_cartesian_set_inside_declared_bounds(self):
        rows = raman_library()
        keys = ("raman_waist_um", "raman_relative_power", "raman_detuning_kHz")
        values = {tuple(row[key] for key in keys) for row in rows}
        axes = [{row[key] for row in rows} for key in keys]
        self.assertEqual(len(rows), 75)
        self.assertEqual(len(values), len(rows))
        self.assertEqual(values, set(itertools.product(*axes)))
        for row in rows:
            for key in keys:
                self.assertGreaterEqual(row[key], BOUNDS[key][0])
                self.assertLessEqual(row[key], BOUNDS[key][1])

    def test_sample_prefix_keeps_failed_launch_ids_absent_and_pairs_all_atom_fields(self):
        record = dict(atom_ids=np.array([0, 2, 5]),
                      fringe_coefficients=np.array([[.2, .1, 0.], [.4, .2, .1], [.6, .1, 0.]]),
                      vz_m_s=np.array([10., 12., 15.]),
                      target_population=np.array([.97, .98, .99]))
        original = copy.deepcopy(record)
        result = _prefix_data({"nominal": [record]}, 3)["nominal"][0]
        np.testing.assert_array_equal(result["atom_ids"], [0, 2])
        for key in record:
            np.testing.assert_array_equal(result[key], original[key][:2])
            np.testing.assert_array_equal(record[key], original[key])


if __name__ == "__main__":
    unittest.main()
