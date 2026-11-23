"""Controlled finite-library comparison of sequential and endpoint co-design.

Native prepared training trajectories are reused from an existing study. Both
methods have exactly the same pump and Raman candidate library. Selection is
frozen before evaluation on independent MOT source trajectories and new joint
speed/divergence perturbations. This is not a global optimization claim.
"""

import argparse
import copy
import hashlib
import itertools
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .batch import source_prefix
from .config import SimulationConfig
from .io import read_stateful_ensemble
from .joint_optimization import (apply_scenario, assert_independent_sources,
    decode, engineering_margins, encode, native_cases, parameters,
    response_record, write_json, _source_digest)
from .joint_statistics import case_statistics, paired_comparison


_PREPARED = None
_WORK = None


def scenario_grid():
    return [dict(name="nominal" if s == d == 1 else "v%03d_d%03d" % (round(100*s), round(100*d)),
                 source_vz_factor=s, source_divergence_factor=d)
            for s, d in itertools.product((.95, 1., 1.05), (.9, 1., 1.1))]


def raman_library():
    return [dict(raman_waist_um=w, raman_relative_power=p, raman_detuning_kHz=d)
            for w, p, d in itertools.product((18., 20., 22., 25., 35.),
                                             (.85, 1., 1.1, 1.2, 1.4), (-25., 0., 25.))]


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _clean(value):
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def _save(path, value):
    write_json(path, _clean(value))


def _init_worker(prepared_dir, families, settings, replicas, incident_samples, bootstrap):
    global _PREPARED, _WORK
    _WORK = dict(settings=settings, incident_samples=incident_samples, bootstrap=bootstrap)
    _PREPARED = {}
    for family in families:
        _PREPARED[family] = {}
        for scenario in settings["scenarios"]:
            name = scenario["name"]
            records = []
            for replica in range(replicas):
                folder = Path(prepared_dir) / family / name
                e = read_stateful_ensemble(folder / ("prepared_%d.h5" % replica))
                with np.load(folder / ("response_%d.npz" % replica), allow_pickle=False) as data:
                    photons = data["scattered_photons"].copy()
                records.append((e, photons))
            _PREPARED[family][name] = records


def _response_data(config, family):
    c = copy.deepcopy(config)
    c.raman.integration_steps = _WORK["settings"]["native_raman_steps"]
    c.raman.scan_points = 5
    result = {}
    for scenario in _WORK["settings"]["scenarios"]:
        varied = copy.deepcopy(c)
        factor = scenario.get("raman_power_factor", 1.)
        varied.raman.peak_rabi_Hz *= factor
        varied.raman.differential_light_shift_Hz *= factor
        varied.raman.two_photon_detuning_Hz += scenario.get("raman_frequency_shift_Hz", 0.)
        result[scenario["name"]] = [response_record(varied, e, photons)
            for e, photons in _PREPARED[family][scenario["name"]]]
    return result


def _evaluate(job):
    from .endpoint_statistics import endpoint_statistics
    name, family, config_data, phase = job
    config = SimulationConfig.from_dict(config_data)
    data = _response_data(config, family)
    n, settings = _WORK["incident_samples"], _WORK["settings"]
    metrics, _ = case_statistics(data, n, settings["incident_flux_per_s"],
        settings["constraints"], bootstrap=_WORK["bootstrap"], seed=83087)
    endpoint, _ = endpoint_statistics(data, n, settings["incident_flux_per_s"],
        config.detection, phase_rad=phase, bootstrap=_WORK["bootstrap"], seed=83087)
    return dict(case=name, family=family, config=config_data, validation=metrics, endpoint=endpoint)


def _rank(row):
    robust = row["endpoint"]["robust"]
    interval = robust["worst_readout_snr_per_rad_95ci"]
    return ((interval[0] if interval else -np.inf), robust["worst_readout_snr_per_rad"])


def select_methods(rows, pump_constraints):
    """Stage A cannot consult contrast or endpoint response when choosing a pump."""
    families = {}
    for row in rows:
        families.setdefault(row["family"], row["validation"]["robust"])
    pump_feasible = [f for f, r in families.items()
        if r["min_efficiency"] >= pump_constraints["min_efficiency"]
        and r["min_effective_target_flux_per_s"] >= pump_constraints["min_effective_target_flux_per_s"]
        and r["max_scattered_photons"] <= pump_constraints["max_scattered_photons"]]
    if not pump_feasible:
        raise RuntimeError("No pump family passes the common preparation constraints")
    efficiency = max(pump_feasible, key=lambda f: (families[f]["min_efficiency"],
                                                 -families[f]["max_scattered_photons"]))
    scattering = min(pump_feasible, key=lambda f: (families[f]["max_scattered_photons"],
                                                   -families[f]["min_efficiency"]))
    feasible = [row for row in rows if row["validation"]["robust"]["feasible"]]
    if not feasible:
        raise RuntimeError("No Raman candidate passes common constraints; do not relax them silently")
    selected = {"joint": max(feasible, key=_rank)}
    for method, family in (("sequential_efficiency", efficiency), ("sequential_scattering", scattering)):
        candidates = [row for row in feasible if row["family"] == family]
        if not candidates:
            raise RuntimeError("Stage A selected %s but no feasible Raman point exists" % family)
        selected[method] = max(candidates, key=_rank)
    return selected


def _parallel_rows(jobs, folder, families, settings, replicas, n, bootstrap, workers, previous=None):
    rows = dict(previous or {})
    missing = [job for job in jobs if job[0] not in rows]
    if missing:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                initargs=(str(folder), list(families), settings, replicas, n, bootstrap)) as pool:
            futures = {pool.submit(_evaluate, job): job[0] for job in missing}
            for done, future in enumerate(as_completed(futures), 1):
                row = future.result()
                rows[row["case"]] = row
                if done % 10 == 0 or done == len(missing):
                    print("Endpoint response candidates: %d/%d" % (done, len(missing)), flush=True)
    return rows


def select(study_dir, output, workers):
    study = _read(study_dir / "study.json")
    base = SimulationConfig.from_dict(study["baseline_config"])
    source = read_stateful_ensemble(study["settings"]["training_source_path"])
    if _source_digest(source) != study["source_split"]["native_training_digest"]:
        raise ValueError("Cached native-training source does not match the historical study")
    alternative = next(row for row in study["search_history"] if row["case"] == "joint_019")
    families = {"baseline": base, "joint_019": decode(alternative["point"], base)}
    settings = copy.deepcopy(study["settings"])
    settings.update(training_atoms=len(source.r_m), native_raman_steps=192,
                    training_replicas=6, bootstrap=400)
    jobs = []
    for family, config in families.items():
        for i, controls in enumerate(raman_library()):
            c = decode(encode(dict(parameters(config, base), **controls)), base)
            # Actual engineering margins are checked on every training drift.
            margins = [engineering_margins(apply_scenario(c, source, s)[0], base,
                          settings["constraints"]) for s in settings["scenarios"]]
            if np.min(margins) < 0:
                continue
            jobs.append(("%s_r%03d" % (family, i), family, c.to_dict(), None))
    identity = dict(training_source_digest=_source_digest(source), jobs=jobs, settings=settings,
        objective="maximize lower 95% source-cluster bound of minimum terminal ratio-readout SNR per radian",
        region_rule="same-pump Raman nodes within 95% of selected lower confidence bound; maximum seven nodes frozen before validation",
        sequential_efficiency="maximize worst preparation efficiency subject to common pump constraints, then optimize Raman",
        sequential_scattering="minimize worst pumping scatter subject to common pump constraints, then optimize Raman",
        validation_grid=scenario_grid())
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    manifest = output / "protocol.json"
    if manifest.exists() and _read(manifest)["signature"] != signature:
        raise ValueError("Protocol changed: choose a new output directory")
    _save(manifest, dict(signature=signature, **identity))
    cached = _read(output / "training_candidates.json") if (output / "training_candidates.json").exists() else {}
    rows = _parallel_rows(jobs, study_dir / "native_training", families, settings, 6,
                          len(source.r_m), 400, workers, cached)
    _save(output / "training_candidates.json", rows)
    selected = select_methods(list(rows.values()), settings["constraints"])
    joint = selected["joint"]
    region = sorted((r for r in rows.values() if r["family"] == joint["family"]
        and r["validation"]["robust"]["feasible"] and _rank(r)[0] >= .95 * _rank(joint)[0]),
        key=_rank, reverse=True)[:7]
    selection = dict(protocol_signature=signature, selected=selected,
                     region_candidates=[r["case"] for r in region],
                     family_configs={k: c.to_dict() for k, c in families.items()},
                     validation_used_for_selection=False)
    _save(output / "selection.json", selection)
    for method, row in selected.items():
        _save(output / (method + "_config.json"), row["config"])
        print("Frozen %s: %s; SNR lower bound %.4f" % (method, row["case"], _rank(row)[0]), flush=True)
    return selection


def _paired_endpoint(before, bd, after, ad):
    identities = [obj.get("bootstrap_identity") for obj in (before, bd, after, ad)]
    if any(value is None or value != identities[0] for value in identities):
        raise ValueError("Endpoint pairing needs identical source bootstrap draws")
    if set(bd["scenarios"]) != set(ad["scenarios"]):
        raise ValueError("Endpoint pairing needs identical declared scenarios")
    result = {}
    for key in ("worst_readout_snr_per_rad", "worst_phase_noise_rad"):
        if any(np.shape(d[key]) != (identities[0]["bootstrap"],) for d in (bd, ad)):
            raise ValueError("Endpoint pairing draw lengths do not match the bootstrap identity")
        with np.errstate(invalid="ignore"):
            delta = np.asarray(ad[key]) - np.asarray(bd[key])
        valid = np.all(np.isfinite(delta))
        interval = np.quantile(delta, [.025, .975]).tolist() if valid else None
        a, b = after["robust"][key], before["robust"][key]
        result[key + "_difference"] = a - b if a is not None and b is not None else None
        result[key + "_difference_95ci"] = interval
    return result


def _prefix_data(data, n):
    return {name: [{k: np.asarray(v)[np.asarray(r["atom_ids"]) < n] for k, v in r.items()}
                    for r in replicas] for name, replicas in data.items()}


def validate(study_dir, output, n, replicas, workers):
    from .endpoint_statistics import endpoint_statistics
    old = _read(study_dir / "study.json")
    selection = _read(output / "selection.json")
    rows = _read(output / "training_candidates.json")
    protocol = _read(output / "protocol.json")
    if selection["protocol_signature"] != protocol["signature"]:
        raise ValueError("Frozen selection and comparison protocol differ")
    identity = {k: v for k, v in protocol.items() if k != "signature"}
    if hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest() != protocol["signature"]:
        raise ValueError("Comparison protocol was modified after selection")
    for row in selection["selected"].values():
        if row != rows.get(row["case"]):
            raise ValueError("Frozen selected candidate does not match its training record")
    if any(name not in rows for name in selection["region_candidates"]):
        raise ValueError("A frozen region candidate is missing from training records")
    base = SimulationConfig.from_dict(old["baseline_config"])
    training = read_stateful_ensemble(old["settings"]["training_source_path"])
    if _source_digest(training) != protocol["training_source_digest"]:
        raise ValueError("Training source changed after selection")
    pool = read_stateful_ensemble(old["source_split"]["validation_path"])
    if n > pool.source_metadata["sampled"]:
        raise ValueError("Validation sample count exceeds the independently simulated source pool")
    source = source_prefix(pool, n)
    assert_independent_sources(training, source)
    settings = copy.deepcopy(old["settings"])
    settings.update(scenarios=scenario_grid(), native_raman_steps=192,
                    validation_atoms=n, recoil_replicas=replicas, bootstrap=1200)
    cases = set(selection["region_candidates"] + [r["case"] for r in selection["selected"].values()])
    families = {rows[case]["family"] for case in cases}
    configs = {family: SimulationConfig.from_dict(selection["family_configs"][family]) for family in families}
    folder = output / "native_validation"
    native_cases(configs, source, base, folder, settings, n, replicas, workers, 270087, bootstrap=128)
    jobs = [(name, rows[name]["family"], rows[name]["config"],
             rows[name]["endpoint"]["operating_phase_rad"]) for name in sorted(cases)]
    validation = _parallel_rows(jobs, folder, families, settings, replicas, n, 1200, workers)
    _init_worker(folder, families, settings, replicas, n, 1200)
    strategies, metric_draws, endpoint_draws, response = {}, {}, {}, {}
    for method, trained in selection["selected"].items():
        row = copy.deepcopy(validation[trained["case"]])
        config = SimulationConfig.from_dict(row["config"])
        data = _response_data(config, row["family"])
        response[method] = data
        for scenario, records in data.items():
            target = output / "selected_responses" / row["case"] / scenario
            target.mkdir(parents=True, exist_ok=True)
            for replica, record in enumerate(records):
                np.savez_compressed(target / ("response_%d.npz" % replica), **record)
        row["validation"], metric_draws[method] = case_statistics(data, n, settings["incident_flux_per_s"],
            settings["constraints"], bootstrap=1200, seed=83087)
        row["endpoint"], endpoint_draws[method] = endpoint_statistics(data, n, settings["incident_flux_per_s"],
            config.detection, phase_rad=trained["endpoint"]["operating_phase_rad"], bootstrap=1200, seed=83087)
        row["parameters"] = parameters(config, base)
        strategies[method] = row
    paired = {}
    matched = {}
    for method in ("sequential_efficiency", "sequential_scattering"):
        b, a = strategies[method], strategies["joint"]
        comparison = paired_comparison(b["validation"], metric_draws[method], a["validation"], metric_draws["joint"])
        comparison.update(_paired_endpoint(b["endpoint"], endpoint_draws[method], a["endpoint"], endpoint_draws["joint"]))
        matched_summaries, matched_draws = {}, {}
        for selected in (method, "joint"):
            result, draw = {}, {}
            for scenario in settings["scenarios"]:
                name = scenario["name"]
                target_flux = min(strategies[x]["validation"]["scenarios"][name]["effective_target_flux_per_s"]
                                  for x in (method, "joint"))
                actual = strategies[selected]["validation"]["scenarios"][name]["effective_target_flux_per_s"]
                # Uniform independent thinning: no state/velocity filtering and unchanged per-atom scatter.
                es, ed = endpoint_statistics({"nominal": response[selected][name]}, n, settings["incident_flux_per_s"],
                    base.detection, phase_rad=strategies[selected]["endpoint"]["operating_phase_rad"],
                    bootstrap=1200, seed=83087, flux_scale=min(1., target_flux / actual))
                result[name] = dict(es["scenarios"]["nominal"], matched_effective_flux_per_s=target_flux,
                                    arrival_rate_scale=min(1., target_flux / actual))
                draw[name] = ed["scenarios"]["nominal"]["readout_snr_per_rad"]
            vals = np.min(np.stack(list(draw.values())), axis=0)
            matched_summaries[selected] = dict(scenarios=result,
                worst_readout_snr_per_rad=min(r["readout_snr_per_rad"] for r in result.values()))
            matched_draws[selected] = vals
        change = matched_draws["joint"] - matched_draws[method]
        comparison["matched_flux_worst_snr_difference"] = (matched_summaries["joint"]["worst_readout_snr_per_rad"]
            - matched_summaries[method]["worst_readout_snr_per_rad"])
        comparison["matched_flux_worst_snr_difference_95ci"] = np.quantile(change, [.025, .975]).tolist()
        comparison["matched_flux_note"] = "Uniform thinning per scenario to common point-estimate effective flux; fixed during bootstrap, not equal scatter"
        paired[method] = comparison
        matched[method] = matched_summaries
    convergence = {}
    for count in sorted({min(128, n), n}):
        convergence[str(count)] = {}
        for method, row in strategies.items():
            data = _prefix_data(response[method], count)
            ms, _ = case_statistics(data, count, settings["incident_flux_per_s"], settings["constraints"], bootstrap=600, seed=83087)
            es, _ = endpoint_statistics(data, count, settings["incident_flux_per_s"], base.detection,
                phase_rad=row["endpoint"]["operating_phase_rad"], bootstrap=600, seed=83087)
            convergence[str(count)][method] = dict(validation=ms, endpoint=es)
    region = []
    joint_snr = strategies["joint"]["endpoint"]["robust"]["worst_readout_snr_per_rad"]
    for case in selection["region_candidates"]:
        r = validation[case]
        region.append(dict(case=case, family=r["family"],
            parameters=parameters(SimulationConfig.from_dict(r["config"]), base),
            validation=r["validation"], endpoint=r["endpoint"],
            passes_all_grid_constraints=r["validation"]["robust"]["feasible"],
            within_95percent_selected_validation_snr=r["endpoint"]["robust"]["worst_readout_snr_per_rad"] >= .95*joint_snr))
    angle = source.v_m_s[:, :2] / source.v_m_s[:, 2, None]
    output_study = dict(settings=settings, strategies=strategies, paired=paired,
        matched_flux=matched, convergence=convergence, work_region=region,
        selection=dict(case_ids={m: r["case"] for m, r in selection["selected"].items()},
                       rule=_read(output / "protocol.json")["objective"],
                       shared_raman_candidates=len(raman_library()), shared_pump_families=len(selection["family_configs"])),
        source_split=dict(training_atoms=len(training.r_m), training_recoil_replicas=6,
            validation_atoms=n, validation_recoil_replicas=replicas,
            training_digest=_source_digest(training), validation_digest=_source_digest(source),
            independent_trajectories=True, newly_generated_blind_source=False,
            nominal_mean_vz_m_s=float(source.v_m_s[:, 2].mean()),
            nominal_rms_divergence_mrad=float(1000*np.sqrt(np.var(angle, axis=0).sum())),
            note="Existing independent source pool reused only after this round's selection; not new blind laboratory data"),
        limitations=["Only two previously shortlisted pump families, finite Raman grid; not global optimality",
            "Same effective-flux floor and pumping-scatter cap, not equal realized scatter",
            "Raman spontaneous scattering absent; source flux normalization is assumed, not measured",
            "Speed/divergence scaled at MOT exit before native pumping; MOT parameters are not regenerated",
            "Terminal signal is expected fluorescence with compound-Poisson noise propagation, not experimental feedback",
            "Scenario grid and source-cluster bootstrap do not guarantee a continuous operating region",
            "Each selected training quadrature phase is held fixed on all validation scenarios",
            "Nested prefix convergence reuses the perturbation defined on the full validation sample, not independent repeated datasets"])
    _save(output / "study.json", output_study)
    return output_study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default="beam_outputs/joint_optimization")
    parser.add_argument("--output", default="beam_outputs/endpoint_comparison")
    parser.add_argument("--stage", choices=("all", "select", "validate", "report"), default="all")
    parser.add_argument("--atoms", type=int, default=256)
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if min(args.atoms, args.replicas, args.workers) < 1:
        parser.error("atoms, replicas, workers must be positive")
    directory, output = Path(args.study).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.stage in ("all", "select"):
        select(directory, output, args.workers)
    if args.stage in ("all", "validate"):
        validate(directory, output, args.atoms, args.replicas, args.workers)
    if args.stage in ("all", "validate", "report"):
        from .endpoint_report import write_report
        write_report(_read(output / "study.json"), output)


if __name__ == "__main__":
    main()
