"""Paired, source-identical time-step check, independent of density-matrix internals.

CLI: python -m rb87_beam.obe_convergence_check coarse.json fine.json output_dir
JSON inputs can be case arrays or objects with a 'cases' member.
"""
import argparse
import json
from pathlib import Path

import numpy as np


_PAIRS = {"matched80": (True, .003), "winner55": (False, .01)}


def _load(data):
    if isinstance(data, (str, Path)):
        data = json.loads(Path(data).read_text(encoding="utf-8-sig"))
    return data["cases"] if isinstance(data, dict) else data


def _index(cases):
    out = {}
    for case in _load(cases):
        key = (case.get("pair", "matched80"), case["scenario"])
        if key[0] not in _PAIRS or not isinstance(key[1], str) or not key[1] or key in out:
            raise ValueError("Unknown pair, empty scenario, or duplicate case: %r" % (key,))
        out[key] = case
    if not out:
        raise ValueError("No cases in resolution comparison")
    return out


def _arrays(rec):
    ids = np.asarray(rec["atom_ids"])
    if (ids.ndim != 1 or not ids.size or ids.dtype.kind not in "iu" or
            len(np.unique(ids)) != ids.size or np.min(ids) < 0):
        raise ValueError("atom_ids must be nonempty, distinct, nonnegative integers")
    order = np.argsort(ids)
    fields = []
    for field in ("target_population", "accepted_target", "scattered_photons"):
        raw = np.atleast_2d(np.asarray(rec[field], dtype=float))
        if (raw.ndim != 2 or not raw.shape[0] or raw.shape[1] != ids.size or
                not np.all(np.isfinite(raw)) or np.min(raw) < -1e-10 or
                (field != "scattered_photons" and np.max(raw) > 1 + 1e-10)):
            raise ValueError("Invalid per-atom values for " + field)
        fields.append(raw[:, order])
    if len(set(f.shape for f in fields)) != 1 or np.any(fields[1] > fields[0] + 1e-10):
        raise ValueError("Inconsistent replicas or acceptance greater than population")
    return tuple(ids[order]), fields


def _compare_case(key, coarse, fine):
    b_over_c, tol = _PAIRS[key[0]]
    sets, metrics = [], []
    for case in (coarse, fine):
        ids_b, b = _arrays(case["baseline"])
        ids_c, c = _arrays(case["candidate"])
        if ids_b != ids_c or b[0].shape != c[0].shape:
            raise ValueError("Recipe inputs must use identical source IDs and replica counts")
        sets.append((ids_b, b[0].shape[0]))
        fb, fc = b[1].mean(), c[1].mean()
        pb, pc = b[2].mean(), c[2].mean()
        if min(fb, fc, pb) <= 0:
            raise ValueError("Undefined flux ratio or relative photon saving")
        # Normalization and count of source failures cancel in this paired ratio.
        metrics.append(dict(flux_ratio=float(fb / fc if b_over_c else fc / fb),
                            scattering_reduction_fraction=float((pb-pc)/pb),
                            photons_saved_per_extracted=float(pb-pc)))
    if sets[0] != sets[1]:
        raise ValueError("Time-step comparison requires identical source IDs and replica counts")
    m0, m1 = metrics
    changes = {k: abs(m1[k]-v) for k, v in m0.items()}
    ratio = m1["flux_ratio"]
    distance = min(abs(ratio-(1-tol)), abs(ratio-(1+tol)))
    inband = lambda r: 1-tol <= r <= 1+tol
    sign = lambda r: r["scattering_reduction_fraction"] > 0
    return dict(pair=key[0], scenario=key[1], extracted_ids=len(sets[0][0]), replicas=sets[0][1],
                ratio_orientation="baseline/candidate" if b_over_c else "candidate/baseline",
                ratio_tolerance=tol, coarse=m0, fine=m1, absolute_change=changes,
                fine_distance_to_nearest_flux_ratio_band_edge=distance,
                fine_ratio_inside_match_band=inband(ratio),
                flux_match_class_changed=inband(ratio) != inband(m0["flux_ratio"]),
                flux_match_within_one_refinement_change=distance <= changes["flux_ratio"],
                fine_scattering_saving_positive=sign(m1), saving_sign_changed=sign(m1) != sign(m0),
                fine_saving_within_one_refinement_change_of_zero=(
                    abs(m1["scattering_reduction_fraction"]) <= changes["scattering_reduction_fraction"]))


def compare_resolutions(coarse_cases, fine_cases, output_dir):
    """Compare exactly paired lists of cases, independent from the OBE solver.

    Relative photon saving is (baseline-candidate)/baseline (fraction, not percent).
    Coarse/fine labels must be assigned by caller from solver settings. Caller must
    establish shared physical source, optical parameters, and stochastic streams.
    Checks here verify source IDs, repeats, case names and defined observables.
    """
    coarse, fine = _index(coarse_cases), _index(fine_cases)
    if coarse.keys() != fine.keys():
        raise ValueError("Different sets of scenarios/pairs: " + repr(sorted(coarse.keys() ^ fine.keys())))
    rows = [_compare_case(k, coarse[k], fine[k]) for k in sorted(coarse)]
    maximum = {}
    for k in rows[0]["absolute_change"]:
        worst = max(rows, key=lambda r: r["absolute_change"][k])
        maximum[k] = {**{x: worst[x] for x in ("pair", "scenario")}, "value": worst["absolute_change"][k]}
    result = dict(n_cases=len(rows), max_absolute_change=maximum,
                  flux_match_flips=sum(r["flux_match_class_changed"] for r in rows),
                  scattering_sign_flips=sum(r["saving_sign_changed"] for r in rows),
                  note=("Two-resolution differences measure sensitivity to these settings, not a "
                        "rigorous truncation-error bound or sampling interval. Flux checks here "
                        "concern only the ratio, not the common absolute flux floor."),
                  cases=rows)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out/"obe_convergence.json").write_text(json.dumps(result, allow_nan=False, indent=2,
                                                       ensure_ascii=False)+"\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("coarse", "fine", "output_dir"):
        p.add_argument(arg)
    args = p.parse_args()
    summary = compare_resolutions(args.coarse, args.fine, args.output_dir)
    print(json.dumps(summary["max_absolute_change"], indent=2))
