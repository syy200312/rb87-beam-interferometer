"""Predeclare and evaluate a discrete tolerance grid around a frozen design.

The grid never reselects a design or an operating phase. Native pumping paths
are read from the parent study; only the Raman response is recomputed. Passing
27 sampled nodes does not establish a continuous feasible parameter box.
"""

import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from .batch import source_prefix
from .config import SimulationConfig
from .endpoint_comparison import _parallel_rows, _read, _save
from .io import read_stateful_ensemble
from .joint_optimization import (BOUNDS, _source_digest, apply_scenario,
    assert_independent_sources, engineering_margins, first_peak_Hz, parameters)


def _signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def local_jobs(selected, baseline):
    """Return the prescribed 3x3x3 nodes with one frozen training phase."""
    config = SimulationConfig.from_dict(selected["config"])
    center = parameters(config, baseline)
    phase = selected["endpoint"]["operating_phase_rad"]
    if not np.isfinite(phase):
        raise ValueError("The selected training phase must be finite")
    axes = {
        "raman_waist_um": [round(center["raman_waist_um"] + x, 12) for x in (0., 1., 2.)],
        "raman_relative_power": [round(center["raman_relative_power"] + x, 12) for x in (-.05, 0., .05)],
        "raman_detuning_kHz": [round(center["raman_detuning_kHz"] + x, 12) for x in (0., 2.5, 5.)],
    }
    for key, values in axes.items():
        if min(values) < BOUNDS[key][0] or max(values) > BOUNDS[key][1]:
            raise ValueError("Prescribed local grid exceeds original bounds for " + key)
    jobs = []
    for i, (width, power, detuning) in enumerate(itertools.product(*axes.values())):
        c = copy.deepcopy(config)
        c.raman.waist_z_m = width * 1e-6
        c.raman.peak_rabi_Hz = (first_peak_Hz(baseline.raman) * power *
                              baseline.raman.waist_z_m / c.raman.waist_z_m)
        c.raman.differential_light_shift_Hz = (baseline.raman.differential_light_shift_Hz *
            c.raman.peak_rabi_Hz / first_peak_Hz(baseline.raman))
        c.raman.two_photon_detuning_Hz = detuning * 1e3
        c.validate()
        jobs.append(["local_%03d" % i, selected["family"], c.to_dict(), float(phase)])
    return axes, jobs


def _parent_inputs(study_dir, output):
    selection = _read(output / "selection.json")
    parent_protocol = _read(output / "protocol.json")
    identity = {key: value for key, value in parent_protocol.items() if key != "signature"}
    if _signature(identity) != parent_protocol["signature"]:
        raise ValueError("The parent comparison protocol was modified")
    if selection["protocol_signature"] != parent_protocol["signature"]:
        raise ValueError("Parent selection and protocol differ")
    old = _read(study_dir / "study.json")
    base = SimulationConfig.from_dict(old["baseline_config"])
    training = read_stateful_ensemble(old["settings"]["training_source_path"])
    if _source_digest(training) != parent_protocol["training_source_digest"]:
        raise ValueError("Training source changed after the parent selection")
    return selection, parent_protocol, old, base, training


def _engineering(config, source, baseline, settings):
    margins = [engineering_margins(apply_scenario(config, source, s)[0], baseline,
                                  settings["constraints"]) for s in settings["scenarios"]]
    return np.min(margins, axis=0).tolist()


def freeze(study_dir, output):
    path = output / "local_region_protocol.json"
    if not path.exists() and (output / "study.json").exists():
        raise ValueError("First freeze is forbidden after validation study.json exists")
    selection, parent, old, base, training = _parent_inputs(study_dir, output)
    axes, jobs = local_jobs(selection["selected"]["joint"], base)
    settings = copy.deepcopy(parent["settings"])
    for _, _, config, _ in jobs:
        candidate = SimulationConfig.from_dict(config)
        for scenarios in (settings["scenarios"], parent["validation_grid"]):
            scenario_settings = dict(settings, scenarios=scenarios)
            if min(_engineering(candidate, training, base, scenario_settings)) < 0:
                raise ValueError("A prescribed local node violates an engineering budget")
    identity = dict(
        parent_protocol_signature=parent["signature"], selection_signature=_signature(selection),
        selected_case_ids={m: row["case"] for m, row in selection["selected"].items()},
        selected_joint_case=selection["selected"]["joint"]["case"],
        selected_joint_config=selection["selected"]["joint"]["config"],
        operating_phase_rad=selection["selected"]["joint"]["endpoint"]["operating_phase_rad"],
        baseline_config=base.to_dict(), axes=axes, jobs=jobs,
        training_source_digest=_source_digest(training), training_settings=settings,
        training_atoms=len(training.r_m), training_recoil_replicas=6, training_bootstrap=400,
        validation_grid=parent["validation_grid"], validation_bootstrap=1200,
        validation_source_path=old["source_split"]["validation_path"],
        validation_data_rule="Use the exact source prefix, replicas and settings of the completed parent study",
        rule="Report all 27 predefined nodes; no reselection; one training-selected phase for every node",
        criterion="All common constraints on the validation grid and >=95% of the frozen selected validation SNR",
        interpretation="Discrete local tolerance diagnostic; not a guaranteed continuous parameter box")
    protocol = dict(signature=_signature(identity), **identity)
    if path.exists():
        if _read(path) != protocol:
            raise ValueError("Local grid protocol changed after freeze; use a new study")
    else:
        if (output / "study.json").exists():
            raise ValueError("Parent validation completed before the local grid could be frozen")
        _save(path, protocol)
    print("Frozen 27 local Raman nodes at one unchanged training phase", flush=True)
    return protocol


def _checked_previous(path, jobs, identity):
    if not path.exists():
        return None
    saved = _read(path)
    if saved["identity"] != identity:
        raise ValueError("Local-response cache inputs changed")
    expected = {job[0]: job for job in jobs}
    for name, row in saved["rows"].items():
        if name not in expected or row["family"] != expected[name][1] or row["config"] != expected[name][2]:
            raise ValueError("Cached local response does not match its frozen config")
        if row["endpoint"]["operating_phase_rad"] != expected[name][3]:
            raise ValueError("Cached local response used a different phase")
    return saved["rows"]


def validate(study_dir, output, workers):
    path = output / "local_region_protocol.json"
    if not path.exists():
        raise ValueError("Freeze the local grid before the parent validation completes")
    protocol = _read(path)
    if _signature({k: v for k, v in protocol.items() if k != "signature"}) != protocol["signature"]:
        raise ValueError("Local region protocol was modified after freeze")
    selection, parent, old, base, training = _parent_inputs(study_dir, output)
    if (protocol["parent_protocol_signature"] != parent["signature"]
            or protocol["selection_signature"] != _signature(selection)
            or protocol["baseline_config"] != base.to_dict()):
        raise ValueError("Frozen local region and current parent selection differ")
    study = _read(output / "study.json")
    expected_ids = protocol["selected_case_ids"]
    if study["selection"]["case_ids"] != expected_ids:
        raise ValueError("The completed parent study uses different method selections")
    for method, selected in selection["selected"].items():
        actual = study["strategies"][method]
        if (actual["case"] != selected["case"] or actual["config"] != selected["config"]
                or actual["endpoint"]["operating_phase_rad"] != selected["endpoint"]["operating_phase_rad"]):
            raise ValueError("The completed parent study changed a frozen method or phase")
    settings = copy.deepcopy(study["settings"])
    if (settings["scenarios"] != protocol["validation_grid"]
            or settings["constraints"] != protocol["training_settings"]["constraints"]
            or settings["incident_flux_per_s"] != protocol["training_settings"]["incident_flux_per_s"]):
        raise ValueError("Parent validation changed the frozen domain or common budgets")
    pool = read_stateful_ensemble(protocol["validation_source_path"])
    n, replicas = settings["validation_atoms"], settings["recoil_replicas"]
    if n > pool.source_metadata.get("sampled", len(pool.r_m)):
        raise ValueError("Parent validation sample count exceeds the source pool")
    source = source_prefix(pool, n)
    if (study["source_split"]["validation_digest"] != _source_digest(source)
            or study["source_split"]["training_digest"] != _source_digest(training)):
        raise ValueError("Local-grid sources differ from the completed parent study")
    assert_independent_sources(training, source)
    jobs = protocol["jobs"]
    families = {job[1] for job in jobs}
    data = {}
    for stage, folder, stage_settings, samples, recoil, bootstrap, ensemble in (
        ("training", study_dir / "native_training", protocol["training_settings"],
         protocol["training_atoms"], 6, 400, training),
        ("validation", output / "native_validation", settings, n, replicas, 1200, source)):
        identity = dict(protocol_signature=protocol["signature"], settings=stage_settings,
                        source_digest=_source_digest(ensemble), samples=samples,
                        replicas=recoil, bootstrap=bootstrap)
        cache = output / ("local_region_%s_rows.json" % stage)
        previous = _checked_previous(cache, jobs, identity)
        rows = _parallel_rows(jobs, folder, families, stage_settings, recoil,
                              samples, bootstrap, workers, previous)
        for name, _, config, _ in jobs:
            margins = _engineering(SimulationConfig.from_dict(config), ensemble, base, stage_settings)
            rows[name]["validation"]["robust"]["engineering_margins"] = margins
            rows[name]["validation"]["robust"]["feasible"] &= min(margins) >= 0
        _save(cache, dict(identity=identity, rows=rows))
        data[stage] = rows
    selected_snr = study["strategies"]["joint"]["endpoint"]["robust"]["worst_readout_snr_per_rad"]
    nodes = []
    for name, family, config, _ in jobs:
        train, valid = data["training"][name], data["validation"][name]
        nodes.append(dict(case=name, family=family,
            parameters=parameters(SimulationConfig.from_dict(config), base),
            training=dict(validation=train["validation"], endpoint=train["endpoint"]),
            validation=valid["validation"], endpoint=valid["endpoint"],
            passes_all_grid_constraints=valid["validation"]["robust"]["feasible"],
            within_95percent_selected_validation_snr=(
                valid["endpoint"]["robust"]["worst_readout_snr_per_rad"] >= .95 * selected_snr)))
    description = dict(protocol_signature=protocol["signature"], axes=protocol["axes"],
        node_count=len(nodes), operating_phase_rad=protocol["operating_phase_rad"],
        selected_case_unchanged=protocol["selected_joint_case"],
        rule=protocol["rule"], criterion=protocol["criterion"], interpretation=protocol["interpretation"])
    _save(output / "local_work_region.json", dict(grid=description, nodes=nodes))
    # Re-read before the small derived-field update so unrelated report evidence
    # added while the response jobs ran is preserved.
    current = _read(output / "study.json")
    if current["selection"] != study["selection"] or current["source_split"] != study["source_split"]:
        raise ValueError("Parent study changed while local-region responses were running")
    current["local_work_region"], current["local_work_region_grid"] = nodes, description
    _save(output / "study.json", current)
    print("Saved all 27 discrete local-grid results without changing the selected design", flush=True)
    return nodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("freeze", "validate"), required=True)
    parser.add_argument("--output", default="beam_outputs/endpoint_comparison")
    parser.add_argument("--study", default="beam_outputs/joint_optimization")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    output, study_dir = Path(args.output).resolve(), Path(args.study).resolve()
    if args.stage == "freeze":
        freeze(study_dir, output)
    else:
        validate(study_dir, output, args.workers)


if __name__ == "__main__":
    main()
