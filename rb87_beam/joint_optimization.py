"""Constrained multi-fidelity joint pumping/Raman optimization.

Low-fidelity Bayesian screening never qualifies a recommendation. Candidates
are selected again on native recoil/motion training runs, frozen, then compared
on an independent source pool. All robustness claims refer to the declared
finite perturbation scenarios, not an entire continuous uncertainty box.
"""

import argparse
import copy
import hashlib
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from .batch import subset_ensemble, source_prefix
from .bayes_search import constrained_expected_improvement
from .config import SimulationConfig
from .io import read_stateful_ensemble, save_stateful_ensemble
from .optimization import ballistic_screen, _model, _pump_one, pump_input_digest
from .physics import (Ensemble, WAVELENGTH, flight_time, trajectory, run_simulation)
from .pumping import PUMP_ALGORITHM_VERSION
from .states import G_CLOCK


BOUNDS = {
    "pump_total_uW": (50., 130.), "transfer_fraction": (.55, .85),
    "pump_detuning_MHz": (.5, 2.5), "pump_length_mm": (3., 7.),
    "raman_waist_um": (18., 35.), "raman_relative_power": (.85, 1.4),
    "raman_detuning_kHz": (-25., 25.),
}
SCENARIOS = [
    dict(name="nominal"),
    dict(name="power_low", pump_power_factor=.95, raman_power_factor=.95),
    dict(name="power_high", pump_power_factor=1.05, raman_power_factor=1.05),
    dict(name="frequency_low", pump_frequency_shift_Hz=-.2e6, raman_frequency_shift_Hz=-1e4),
    dict(name="frequency_high", pump_frequency_shift_Hz=.2e6, raman_frequency_shift_Hz=1e4),
    dict(name="beam_slow", source_vz_factor=.98),
    dict(name="beam_fast", source_vz_factor=1.02),
]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def load_config(path):
    return SimulationConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def first_peak_Hz(r):
    if r.peak_rabi_Hz > 0:
        return float(r.peak_rabi_Hz)
    width = r.waist_z_m * np.sqrt(np.pi / 2) if r.envelope == "gaussian" else r.rectangular_width_m
    return float(r.reference_vz_m_s / (4 * width))


def parameters(config, baseline):
    p, r = config.pumping, config.raman
    total = p.transfer_power_W + p.dark_power_W
    return dict(pump_total_uW=total * 1e6, transfer_fraction=p.transfer_power_W / total,
        pump_detuning_MHz=p.transfer_detuning_Hz / 1e6,
        pump_length_mm=2 * p.half_widths * p.waist_z_m * 1e3,
        raman_waist_um=r.waist_z_m * 1e6,
        raman_relative_power=first_peak_Hz(r) * r.waist_z_m /
                             (first_peak_Hz(baseline.raman) * baseline.raman.waist_z_m),
        raman_detuning_kHz=r.two_photon_detuning_Hz / 1e3)


def encode(values):
    return np.array([(values[key] - low) / (high - low) for key, (low, high) in BOUNDS.items()])


def decode(point, baseline):
    if np.shape(point) != (len(BOUNDS),) or not np.all(np.isfinite(point)) or np.any(np.asarray(point) < 0) or np.any(np.asarray(point) > 1):
        raise ValueError("Joint design coordinates must be finite and lie in the unit box")
    v = {key: float(low + x * (high - low)) for x, (key, (low, high)) in zip(point, BOUNDS.items())}
    c = copy.deepcopy(baseline)
    p, r = c.pumping, c.raman
    p.enabled = True
    p.transfer_power_W = v["pump_total_uW"] * 1e-6 * v["transfer_fraction"]
    p.dark_power_W = v["pump_total_uW"] * 1e-6 * (1 - v["transfer_fraction"])
    p.transfer_detuning_Hz = p.dark_detuning_Hz = v["pump_detuning_MHz"] * 1e6
    p.waist_z_m = v["pump_length_mm"] * 1e-3 / (2 * p.half_widths)
    r.waist_z_m = v["raman_waist_um"] * 1e-6
    r.peak_rabi_Hz = (first_peak_Hz(baseline.raman) * v["raman_relative_power"] *
                      baseline.raman.waist_z_m / r.waist_z_m)
    r.differential_light_shift_Hz = (baseline.raman.differential_light_shift_Hz *
                                    r.peak_rabi_Hz / first_peak_Hz(baseline.raman))
    r.two_photon_detuning_Hz = v["raman_detuning_kHz"] * 1e3
    c.validate()
    return c


def apply_scenario(config, source, scenario):
    c, e = copy.deepcopy(config), copy.deepcopy(source)
    p, r = c.pumping, c.raman
    pf, rf = scenario.get("pump_power_factor", 1.), scenario.get("raman_power_factor", 1.)
    p.transfer_power_W *= pf
    p.dark_power_W *= pf
    p.transfer_detuning_Hz += scenario.get("pump_frequency_shift_Hz", 0.)
    p.dark_detuning_Hz += scenario.get("pump_frequency_shift_Hz", 0.)
    r.peak_rabi_Hz = first_peak_Hz(r) * rf
    r.differential_light_shift_Hz *= rf
    r.two_photon_detuning_Hz += scenario.get("raman_frequency_shift_Hz", 0.)
    if source.reference_plane_z_m != 0:
        raise ValueError("Joint optimization needs a MOT-exit input, not an already pumped checkpoint")
    speed = scenario.get("source_vz_factor", 1.)
    divergence = scenario.get("source_divergence_factor", 1.)
    for key, value in (("source_vz_factor", speed), ("source_divergence_factor", divergence)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value) or value <= 0):
            raise ValueError(key + " must be a positive finite scalar")
    if "source_divergence_factor" in scenario:
        # Independently vary longitudinal speed and the centered angular spread
        # at the MOT exit. The mean beam angle remains fixed before pumping.
        if divergence != 1.:
            e.validate()
            angles = e.v_m_s[:, :2] / e.v_m_s[:, 2, None]
            center = angles.mean(axis=0)
            angles = center + divergence * (angles - center)
            e.v_m_s[:, :2] = angles * e.v_m_s[:, 2, None]
        if speed != 1.:
            e.v_m_s *= speed
    elif speed != 1.:
        # Legacy speed-only scenarios keep the transverse velocities fixed.
        e.v_m_s[:, 2] *= speed
    c.validate()
    e.validate()
    return c, e


def engineering_margins(config, baseline, constraints):
    r, p = config.raman, config.pumping
    return [1 - (p.transfer_power_W + p.dark_power_W) / constraints["max_pump_power_W"],
            1 - 2 * first_peak_Hz(r) / constraints["max_raman_peak_Hz"],
            1 - (first_peak_Hz(r) * r.waist_z_m) /
                (first_peak_Hz(baseline.raman) * baseline.raman.waist_z_m * constraints["max_relative_raman_power"]),
            np.pi * r.waist_z_m ** 2 / WAVELENGTH / constraints["min_rayleigh_m"] - 1]


def metric_margins(summary, constraints):
    r = summary["robust"]
    return [r["min_efficiency"] - constraints["min_efficiency"],
            r["min_effective_target_flux_per_s"] / constraints["min_effective_target_flux_per_s"] - 1,
            1 - r["max_scattered_photons"] / constraints["max_scattered_photons"],
            (r["worst_contrast"] or 0.) - constraints["min_fringe_contrast"]]


def _density(pop):
    result = np.zeros((len(pop), 8, 8), complex)
    result[:, np.arange(8), np.arange(8)] = pop
    return result


def response_record(config, ensemble, photons):
    c = copy.deepcopy(config)
    c.pumping.enabled, c.detection.noise = False, False
    result = run_simulation(c, ensemble)
    design = np.column_stack((np.ones(len(result["phase_rad"])), np.cos(result["phase_rad"]), np.sin(result["phase_rad"])))
    return dict(atom_ids=np.asarray(ensemble.source_metadata.get("sample_ids", np.arange(len(ensemble.r_m))), int),
                fringe_coefficients=np.linalg.lstsq(design, result["per_atom_P2"], rcond=None)[0].T,
                detection_weights=result["detection"]["detection_weights"],
                vz_m_s=ensemble.v_m_s[:, 2],
                target_population=ensemble.ground_density[:, G_CLOCK, G_CLOCK].real,
                scattered_photons=np.asarray(photons))


def ballistic_response(config, source):
    screen = ballistic_screen(config, source)
    model = _model(config)
    times = flight_time(source, model.end_z)
    r, v = trajectory(source, times, config.raman)
    out = Ensemble(r, v, source.flux_per_s, copy.deepcopy(source.source_metadata),
        ground_density=_density(screen["ground_populations"]),
        reference_plane_z_m=model.end_z, reference_time_s=times)
    return response_record(config, out, np.full(len(r), screen["mean_scattered_photons"]))


def evaluate_screen(config, source, baseline, settings):
    from .joint_statistics import case_statistics
    data, margins = {}, []
    for scenario in settings["scenarios"]:
        c, e = apply_scenario(config, source, scenario)
        margins.append(engineering_margins(c, baseline, settings["constraints"]))
        data[scenario["name"]] = [ballistic_response(c, e)]
    summary, _ = case_statistics(data, len(source.r_m), source.flux_per_s,
                                settings["constraints"], bootstrap=32, seed=70087)
    margin = metric_margins(summary, settings["constraints"]) + np.min(margins, axis=0).tolist()
    return summary, margin


def _source_digest(e):
    h = hashlib.sha256()
    for array in (e.r_m, e.v_m_s, e.ground_density,
                  np.asarray(e.source_metadata.get("sample_ids", np.arange(len(e.r_m))))):
        h.update(np.ascontiguousarray(array).tobytes())
    h.update(str(e.reference_plane_z_m).encode())
    if e.reference_time_s is not None:
        h.update(e.reference_time_s.tobytes())
    return h.hexdigest()


def assert_independent_sources(training, validation):
    """Reject overlapping trajectories even when two files are different subsets."""
    def fingerprints(e):
        return {hashlib.sha256(np.ascontiguousarray(np.r_[r, v, rho.real.ravel(), rho.imag.ravel()]).tobytes()).hexdigest()
                for r, v, rho in zip(e.r_m, e.v_m_s, e.ground_density)}
    if fingerprints(training) & fingerprints(validation):
        raise ValueError("Training and validation share source atom trajectories; use independent source samples")


def validation_subset(source, incident_samples):
    available = source.source_metadata.get("sampled", len(source.r_m))
    if incident_samples > available:
        raise ValueError("Validation requests more launches than were simulated; extend the MOT source first")
    return source_prefix(source, incident_samples)


def search_identity(baseline, training, settings):
    keys = ("bounds", "scenarios", "search_budget", "search_seed", "training_atoms",
            "screen_raman_steps", "incident_flux_per_s", "constraints")
    identity = dict(baseline=baseline.to_dict(), settings={key: settings[key] for key in keys},
                    source_digest=_source_digest(training))
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(), identity


def native_cases(cases, source, baseline, directory, settings, incident_samples,
                 replicas, workers, seed, bootstrap, progress=print):
    """One shared worker pool; hash-checked per-atom recoil caches across Raman designs."""
    from .joint_statistics import case_statistics
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Extending a source prefix may change its full digest; reuse only atom files
    # whose physical entry-state digest AND pump/seed/model identities agree.
    compatible = {}
    for manifest in (directory / "pump_cache").glob("*/manifest.json"):
        existing = json.loads(manifest.read_text(encoding="utf-8"))
        key = json.dumps({k: v for k, v in existing.items() if k != "source_digest"}, sort_keys=True)
        compatible.setdefault(key, []).append(manifest.parent)
    groups, pending, checked_groups = [], {}, set()
    for case, config in cases.items():
        for scenario in settings["scenarios"]:
            c, e = apply_scenario(config, source, scenario)
            c.raman.integration_steps = settings["native_raman_steps"]
            c.raman.scan_points = 5  # exact harmonic coefficients, verified against 41-point replay
            model = _model(c)
            entry = flight_time(e, model.start_z)
            r, v = trajectory(e, entry, c.raman)
            pop = np.diagonal(e.ground_density, axis1=1, axis2=2).real
            identity = dict(pumping=asdict(c.pumping), algorithm=PUMP_ALGORITHM_VERSION,
                source_digest=_source_digest(e), acceleration=c.raman.acceleration_m_s2, seed=seed)
            identity = json.loads(json.dumps(identity, sort_keys=True))
            common_key = json.dumps({k: v for k, v in identity.items() if k != "source_digest"}, sort_keys=True)
            signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            folder = directory / "pump_cache" / signature
            folder.mkdir(parents=True, exist_ok=True)
            manifest = folder / "manifest.json"
            if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != identity:
                raise ValueError("Native joint cache signature mismatch")
            write_json(manifest, identity)
            ids = np.asarray(e.source_metadata.get("sample_ids", np.arange(len(r))), int)
            for replica in range(replicas):
                group_key = (str(folder), replica)
                if group_key in checked_groups:
                    continue
                (folder / ("replica_%d" % replica)).mkdir(exist_ok=True)
                for i, atom_id in enumerate(ids):
                    path = folder / ("replica_%d" % replica) / ("atom_%06d.npz" % atom_id)
                    digest = pump_input_digest(r[i], v[i], pop[i], entry[i])
                    if not path.exists():
                        for old_folder in compatible.get(common_key, []):
                            old_path = old_folder / ("replica_%d" % replica) / path.name
                            if old_path.exists():
                                with np.load(old_path, allow_pickle=False) as old:
                                    same_input = str(old["input_digest"]) == digest
                                if same_input:
                                    temporary = path.with_suffix(".copy.npz")
                                    shutil.copy2(old_path, temporary)
                                    temporary.replace(path)
                                    break
                    if path.exists():
                        with np.load(path, allow_pickle=False) as saved:
                            if str(saved["input_digest"]) != digest:
                                raise ValueError("Cached native atom input changed")
                    else:
                        pending[str(path)] = (c.to_dict(), int(atom_id), replica, seed, r[i], v[i], pop[i], entry[i])
                checked_groups.add(group_key)
            groups.append((case, scenario["name"], c, e, model.end_z, folder, ids))
    if progress:
        progress("Native joint validation: %d missing cached atom/recoil jobs" % len(pending))
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_pump_one, job): path for path, job in pending.items()}
            for done, future in enumerate(as_completed(futures), 1):
                record = future.result()
                path = Path(futures[future])
                temporary = path.with_suffix(".tmp.npz")
                np.savez_compressed(temporary, **record)
                temporary.replace(path)
                if progress and (done % 128 == 0 or done == len(futures)):
                    progress("Native joint recoil jobs: %d/%d" % (done, len(futures)))
    data = {case: {} for case in cases}
    prepared_arrays = {}
    for case, scenario, c, e, end_z, folder, ids in groups:
        records = []
        for replica in range(replicas):
            group_key = (str(folder), replica)
            if group_key not in prepared_arrays:
                atoms = []
                for atom_id in ids:
                    with np.load(folder / ("replica_%d" % replica) / ("atom_%06d.npz" % atom_id), allow_pickle=False) as a:
                        atoms.append({key: a[key] for key in a.files})
                prepared_arrays[group_key] = dict(
                    r=np.array([a["r_m"] for a in atoms]), v=np.array([a["v_m_s"] for a in atoms]),
                    ground=np.array([a["ground"] for a in atoms]),
                    time=np.array([a["reference_time_s"] for a in atoms]),
                    photons=np.array([a["photons"] for a in atoms]))
            arrays = prepared_arrays[group_key]
            ground, photons = arrays["ground"], arrays["photons"]
            metadata = copy.deepcopy(e.source_metadata)
            metadata["optical_pumping"] = dict(preparation_efficiency=float(ground[:, G_CLOCK].mean()),
                mean_scattered_photons=float(photons.mean()), config=asdict(c.pumping), seed=seed, replica=replica)
            output = Ensemble(arrays["r"], arrays["v"],
                e.flux_per_s, metadata, ground_density=_density(ground), reference_plane_z_m=end_z,
                reference_time_s=arrays["time"])
            output.validate()
            target = directory / case / scenario
            target.mkdir(parents=True, exist_ok=True)
            save_stateful_ensemble(output, target / ("prepared_%d.h5" % replica))
            record = response_record(c, output, photons)
            np.savez_compressed(target / ("response_%d.npz" % replica), **record)
            records.append(record)
        data[case][scenario] = records
    summaries, draws = {}, {}
    for case in cases:
        summaries[case], draws[case] = case_statistics(data[case], incident_samples,
            settings["incident_flux_per_s"], settings["constraints"], bootstrap=bootstrap, seed=83087)
        margins = [engineering_margins(apply_scenario(cases[case], source, s)[0], baseline,
                                       settings["constraints"]) for s in settings["scenarios"]]
        summaries[case]["robust"]["engineering_margins"] = np.min(margins, axis=0).tolist()
        summaries[case]["robust"]["feasible"] &= bool(np.min(margins) >= 0)
        write_json(directory / case / "statistics.json", summaries[case])
        if progress:
            progress("Native %s: worst score=%.5f; worst C=%s; feasible=%s" % (
                case, summaries[case]["robust"]["worst_score"],
                summaries[case]["robust"]["worst_contrast"], summaries[case]["robust"]["feasible"]))
    return summaries, draws


def refine_native_raman(cases, initial, source, baseline, directory, settings, replicas, progress=print):
    """Bayesian correction on native prepared trajectories, before seeing validation.

    Pump families come from the joint search; all Raman proposals use the same
    native per-atom recoil realizations within a family. No constraint is relaxed.
    """
    from .joint_statistics import case_statistics
    directory = Path(directory)
    identity = dict(configs={key: c.to_dict() for key, c in cases.items()},
                    source_digest=_source_digest(source), scenarios=settings["scenarios"],
                    replicas=replicas, budget=settings["native_evaluations"], constraints=settings["constraints"],
                    native_raman_steps=settings["native_raman_steps"])
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    manifest = directory / ("native_search_R%d_manifest.json" % replicas)
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8"))["signature"] != signature:
        raise ValueError("Native refinement inputs changed; use a new study directory")
    write_json(manifest, dict(signature=signature, **identity))
    families, seen = {}, set()
    for case, c in cases.items():
        key = json.dumps(asdict(c.pumping), sort_keys=True)
        if key not in seen:
            families[case] = c
            seen.add(key)
    prepared = {}
    for family in families:
        prepared[family] = {}
        for s in settings["scenarios"]:
            name = s["name"]
            prepared[family][name] = []
            for replica in range(replicas):
                folder = directory / family / name
                e = read_stateful_ensemble(folder / ("prepared_%d.h5" % replica))
                with np.load(folder / ("response_%d.npz" % replica), allow_pickle=False) as rec:
                    photons = rec["scattered_photons"].copy()
                prepared[family][name].append((e, photons))
    candidates, structured = [], []
    for family, c in families.items():
        point = encode(parameters(c, baseline))
        for width, power, detuning in ((18., .85, 0.), (25., .85, 0.), (35., 1., 0.),
                                       (25., 1.4, 0.), (18., 1.1, -25.), (18., 1.1, 25.)):
            v = dict(parameters(c, baseline), raman_waist_um=width,
                     raman_relative_power=power, raman_detuning_kHz=detuning)
            structured.append((family, encode(v)))
        random = qmc.LatinHypercube(3, seed=67187).random(192)
        for p in random:
            full = point.copy()
            full[4:] = p
            candidates.append((family, full))
    pool_records = structured + candidates
    pool = np.array([row[1] for row in pool_records])
    history_path = directory / ("native_search_R%d_history.json" % replicas)
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
    else:
        history = [dict(case=name, point=encode(parameters(c, baseline)).tolist(),
            score=initial[name]["robust"]["worst_score"], metrics=initial[name],
            margins=metric_margins(initial[name], settings["constraints"]) +
                initial[name]["robust"]["engineering_margins"],
            feasible=initial[name]["robust"]["feasible"], fidelity="native recoil/motion training")
            for name, c in cases.items()]
    initial_count = len(cases)
    for i in range(len(history) - initial_count, settings["native_evaluations"]):
        X = np.array([row["point"] for row in history])
        if i < len(structured) and np.min(np.linalg.norm(X - pool[i], axis=1)) > 1e-10:
            index, acquisition = i, None
        else:
            index, acquisition = constrained_expected_improvement(X,
                np.array([row["score"] for row in history]),
                np.array([row["margins"] for row in history]), pool, seed=67187)
        family, point = pool_records[index]
        c = decode(point, baseline)
        c.raman.integration_steps, c.raman.scan_points = settings["native_raman_steps"], 5
        data, budget_margins = {}, []
        for scenario in settings["scenarios"]:
            varied, _ = apply_scenario(c, source, scenario)
            budget_margins.append(engineering_margins(varied, baseline, settings["constraints"]))
            data[scenario["name"]] = [response_record(varied, e, photons)
                for e, photons in prepared[family][scenario["name"]]]
        summary, _ = case_statistics(data, len(source.r_m), settings["incident_flux_per_s"],
                                     settings["constraints"], bootstrap=500, seed=83087)
        margins = metric_margins(summary, settings["constraints"]) + np.min(budget_margins, axis=0).tolist()
        name = "native_%03d" % i
        cases[name] = c
        initial[name] = summary
        history.append(dict(case=name, point=point.tolist(), family=family, score=summary["robust"]["worst_score"],
            metrics=summary, margins=margins, feasible=bool(min(margins) >= 0),
            acquisition=acquisition, fidelity="native recoil/motion training; reused prepared trajectories"))
        write_json(history_path, history)
        if progress:
            progress("Native-response BO %d/%d: worst score=%.5f C=%.4f feasible=%s" % (
                i + 1, settings["native_evaluations"], history[-1]["score"], summary["robust"]["worst_contrast"],
                history[-1]["feasible"]))
    # Reconstruct cached native proposals when resuming.
    for row in history[initial_count:]:
        cases[row["case"]] = decode(row["point"], baseline)
        initial[row["case"]] = row["metrics"]
        initial[row["case"]]["robust"]["feasible"] = row["feasible"]
    return cases, initial, history


def search(baseline, training, output, settings, progress=print):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    signature, identity = search_identity(baseline, training, settings)
    manifest = output / "search_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8"))["signature"] != signature:
        raise ValueError("Search parameters or source changed; use a new output directory")
    write_json(manifest, dict(signature=signature, **identity))
    save_stateful_ensemble(training, output / "training_source.h5")
    low = copy.deepcopy(baseline)
    low.raman.integration_steps, low.raman.scan_points = settings["screen_raman_steps"], 5
    baseline_values = parameters(low, baseline)
    structured = [encode(baseline_values)]
    for width, power in ((18., 1.), (18., 1.15), (18., 1.35), (22., 1.1), (30., 1.05)):
        structured.append(encode(dict(baseline_values, raman_waist_um=width, raman_relative_power=power)))
    random_pool = qmc.LatinHypercube(len(BOUNDS), seed=settings["search_seed"]).random(256)
    pool = np.vstack((structured, random_pool))
    history_path = output / "search_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
    for iteration in range(len(history), settings["search_budget"]):
        diagnostics = None
        if iteration < len(structured):
            index = iteration
        else:
            X = np.asarray([row["point"] for row in history])
            y = np.asarray([row["score"] for row in history])
            margins = np.asarray([row["margins"] for row in history])
            index, diagnostics = constrained_expected_improvement(X, y, margins, pool, seed=settings["search_seed"])
        point = pool[index]
        c = decode(point, low)
        summary, margins = evaluate_screen(c, training, baseline, settings)
        row = dict(case="joint_%03d" % iteration, point=point.tolist(),
            parameters=parameters(c, baseline), score=summary["robust"]["worst_score"],
            feasible=bool(min(margins) >= 0), margins=margins, metrics=summary,
            acquisition=diagnostics, fidelity="24-state ballistic populations; no mean-force/recoil feedback")
        history.append(row)
        write_json(history_path, history)
        if progress:
            progress("Joint BO %d/%d: score=%.5f feasible=%s (%s)" % (
                iteration + 1, settings["search_budget"], row["score"], row["feasible"], row["case"]))
    return history


def validate(baseline, training, validation_path, output, settings, replicas, workers, progress=print,
             native_training=None):
    from .joint_statistics import paired_comparison
    output = Path(output)
    signature, _ = search_identity(baseline, training, settings)
    if json.loads((output / "search_manifest.json").read_text(encoding="utf-8"))["signature"] != signature:
        raise ValueError("Validation baseline, training source or search settings differ from the frozen search")
    history = json.loads((output / "search_history.json").read_text(encoding="utf-8"))
    candidates = sorted((r for r in history if r["feasible"]), key=lambda r: r["score"], reverse=True)[:2]
    cases = {"baseline": baseline}
    for row in candidates:
        cases[row["case"]] = decode(row["point"], baseline)
    native_training = native_training if native_training is not None else training
    training_replicas = settings["training_replicas"]
    native, _ = native_cases(cases, native_training, baseline, output / "native_training", settings,
        len(native_training.r_m), training_replicas, workers, 170087, bootstrap=500, progress=progress)
    cases, native, native_history = refine_native_raman(cases, native, native_training, baseline,
        output / "native_training", settings, training_replicas, progress=progress)
    feasible = [name for name in cases if native[name]["robust"]["feasible"]]
    if not feasible:
        raise RuntimeError("No candidate, including baseline, satisfies declared native-training constraints")
    selected = max(feasible, key=lambda name: native[name]["robust"]["worst_score_95ci"][0])
    selected_config = cases[selected]
    # This choice is frozen BEFORE the independent source is read.
    selected_config.save(output / "selected_config.json")
    write_json(output / "selection.json", dict(selected_candidate=selected,
        selection_rule="Feasible native-training candidate with largest lower 95% source-cluster worst-score bound",
        independent_source_used_for_selection=False, native_training=native))
    source = read_stateful_ensemble(validation_path)
    held = validation_subset(source, settings["validation_atoms"])
    assert_independent_sources(native_training, held)
    summaries, draws = native_cases({"before": baseline, "after": selected_config}, held, baseline,
        output / "validation", settings, settings["validation_atoms"], replicas, workers,
        270087, bootstrap=1200, progress=progress)
    paired = paired_comparison(summaries["before"], draws["before"], summaries["after"], draws["after"])
    supported = bool(summaries["after"]["robust"]["feasible"] and paired["worst_score_improvement_95ci"][0] > 0)
    if supported:
        selected_config.save(output / "recommended_config.json")
    elif (output / "recommended_config.json").exists():
        # Derived output from an earlier validation must not imply current approval.
        (output / "recommended_config.json").unlink()
    study = dict(settings=settings, baseline_config=baseline.to_dict(), selected_config=selected_config.to_dict(),
        selected_candidate=selected, search_history=history, native_search_history=native_history,
        validation=dict(before=summaries["before"], after=summaries["after"], paired=paired),
        source_split=dict(training_digest=_source_digest(training), native_training_digest=_source_digest(native_training),
            validation_digest=_source_digest(held),
            training_path=settings.get("training_source_path"),
            validation_path=str(Path(validation_path).resolve()),
            training_source_seed=training.source_metadata.get("source_seed"),
            validation_source_seed=held.source_metadata.get("source_seed"),
            note="Existing independent source pool not used for this joint search; not newly generated blind data"),
        decision=dict(supported_improvement=supported,
            constraints_passed=summaries["after"]["robust"]["feasible"],
            note=("声明场景的约束点估计通过，且源原子配对区间支持联合响应目标提高；名义对比度是否提高需单独看其配对区间。"
                  if supported else "当前结果未同时通过约束点估计和联合目标配对改进区间，尚不生成确认的推荐配置。")))
    write_json(output / "study.json", study)
    from .joint_report import export_joint_report
    export_joint_report(study, output)
    return study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "search", "validate", "report"), default="all")
    parser.add_argument("--config", type=Path, default=Path("beam_configs/optical_pumping_optimized.json"))
    parser.add_argument("--training-source", type=Path, default=Path("beam_outputs/pylcp_multilevel_32/source_ensemble.h5"))
    parser.add_argument("--source", type=Path, default=Path("beam_outputs/pumping_optimization/source/source_ensemble.h5"))
    parser.add_argument("--output", type=Path, default=Path("beam_outputs/joint_optimization"))
    parser.add_argument("--evaluations", type=int, default=24)
    parser.add_argument("--native-evaluations", type=int, default=18)
    parser.add_argument("--training-atoms", type=int, default=16)
    parser.add_argument("--validation-atoms", type=int, default=256)
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument("--training-replicas", type=int, default=6)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if min(args.training_atoms, args.validation_atoms, args.replicas, args.training_replicas, args.workers, args.native_evaluations) < 1 or not 6 <= args.evaluations <= 262:
        parser.error("Positive sample/worker counts and 6..262 evaluations required")
    if args.stage == "report":
        from .joint_report import export_joint_report
        export_joint_report(json.loads((args.output / "study.json").read_text(encoding="utf-8")), args.output)
        return
    baseline = load_config(args.config)
    if baseline.raman.envelope != "gaussian" or baseline.raman.state_handling != "inherit" or baseline.raman.selection_sigma_vx_m_s != 0:
        parser.error("Joint study currently requires Gaussian Raman envelopes, inherit, and no artificial velocity selection")
    baseline.pumping.enabled, baseline.detection.noise = True, False
    baseline.pumping.rate_backend = "fixed_axis"
    baseline.pumping.diffusion_quadrature_order, baseline.pumping.recoil_chunk_s = 3, 25e-6
    baseline.raman.peak_rabi_Hz = first_peak_Hz(baseline.raman)
    baseline.validate()
    training_pool = read_stateful_ensemble(args.training_source)
    if args.training_atoms > len(training_pool.r_m):
        parser.error("Not enough independent training atoms")
    training = subset_ensemble(training_pool, np.linspace(0, len(training_pool.r_m) - 1, args.training_atoms, dtype=int))
    training.source_metadata["original_sample_ids"] = training.source_metadata.get("sample_ids", list(range(len(training.r_m))))
    training.source_metadata["sample_ids"] = list(range(len(training.r_m)))
    settings = dict(bounds=BOUNDS, scenarios=SCENARIOS, search_budget=args.evaluations, search_seed=51087,
        training_atoms=args.training_atoms, native_training_atoms=len(training_pool.r_m), validation_atoms=args.validation_atoms,
        native_evaluations=args.native_evaluations,
        training_replicas=args.training_replicas,
        recoil_replicas=args.replicas, native_raman_steps=192, screen_raman_steps=64,
        incident_flux_per_s=baseline.source.incident_flux_per_s,
        training_source_path=str(args.training_source.resolve()),
        constraints=dict(min_efficiency=.97, min_effective_target_flux_per_s=.95 * baseline.source.incident_flux_per_s,
            max_scattered_photons=6., min_fringe_contrast=.45, max_pump_power_W=160e-6,
            max_raman_peak_Hz=500e3, max_relative_raman_power=1.5, min_rayleigh_m=.001),
        objective="min_declared_scenarios 2*fringe_amplitude*sqrt(detected_flux/incident_flux)",
        uncertainty_note="Engineering drift scenarios are declared assumptions, not measured instrument noise",
        model_note="24-level pumping populations and recoil; effective Raman clock pair; fixed T and aperture; no asynchronous lab-arrival averaging")
    if args.stage in ("all", "search"):
        search(baseline, training, args.output, settings, progress=lambda message: print(message, flush=True))
    if args.stage in ("all", "validate"):
        training_pool.source_metadata["sample_ids"] = list(range(len(training_pool.r_m)))
        study = validate(baseline, training, args.source, args.output, settings, args.replicas, args.workers,
                         progress=lambda message: print(message, flush=True), native_training=training_pool)
        print("Report:", (args.output / "report.html").resolve())
        print("Validated improvement:", study["decision"]["supported_improvement"])


if __name__ == "__main__":
    main()
