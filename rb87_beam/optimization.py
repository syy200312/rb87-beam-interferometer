"""Constrained pump scan and held-out Monte Carlo convergence study.

Screening integrates all 24 populations on prescribed ballistic trajectories.
Final comparisons always use native PyLCP motion with recoil diffusion.
"""

import copy
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from .batch import atom_rng, subset_ensemble, source_prefix
from .config import SimulationConfig
from .io import read_stateful_ensemble, save_stateful_ensemble
from .physics import Ensemble, GAMMA, flight_time, trajectory, run_simulation
from .pumping import OpticalPumping, PUMP_ALGORITHM_VERSION
from .states import G_CLOCK

_MODELS = {}
_SCREEN_CONFIG = None
_SCREEN_SOURCE = None


def _model(config):
    key = json.dumps([asdict(config.pumping), asdict(config.raman)], sort_keys=True)
    if key not in _MODELS:
        if len(_MODELS) > 20:
            _MODELS.clear()
        _MODELS[key] = OpticalPumping(config.pumping, config.raman)
    return _MODELS[key]


def ballistic_screen(config, ensemble):
    """Fast, exact 24-population rate integration on *prescribed* ballistic motion.

    Constant bias, shared beam envelopes and transverse velocity allow one native
    PyLCP rate construction per atom. Recoil and mean-force feedback are omitted
    only at this screening stage, and explicitly restored for shortlisted cases.
    """
    model = _model(config)
    p = config.pumping
    if config.raman.acceleration_m_s2[0] != 0 or np.linalg.norm(config.raman.rotation_rad_s) != 0:
        raise ValueError("Ballistic screen requires constant vx and no rotation")
    times = flight_time(ensemble, model.start_z)
    r, v = trajectory(ensemble, times, config.raman)
    initial = np.real(np.diagonal(ensemble.ground_density, axis1=1, axis2=2))
    ground, scattered = [], []
    center = np.array([0., 0., p.center_z_m]) / model.x0
    decay = model.eqn.Rev_decay.copy()
    for ri, vi, pop in zip(r, v, initial):
        q, _ = model.eqn.construct_evolution_matrix(center, vi / model.v0)
        light = q.copy() - decay
        duration = (model.end_z - model.start_z) / vi[2]

        def generator(t):
            yy = ri[1] + vi[1] * t + .5 * config.raman.acceleration_m_s2[1] * t * t
            zz = ri[2] + vi[2] * t
            profile = np.exp(-2 * (((zz - p.center_z_m) / p.waist_z_m) ** 2 + (yy / p.waist_y_m) ** 2))
            matrix = np.zeros((25, 25))
            matrix[:24, :24] = GAMMA * (decay + profile * light)
            matrix[24, 8:24] = GAMMA
            return matrix

        sol = solve_ivp(lambda t, y: generator(t) @ y, (0, duration),
            np.r_[pop, np.zeros(17)], method="BDF", jac=lambda t, y: generator(t),
            rtol=2e-5, atol=1e-9, max_step=duration / 20)
        if not sol.success:
            raise RuntimeError("Ballistic population screen failed: " + sol.message)
        final = model.checked(sol.y[:24, -1])
        ground.append(final[:8] + model.branching @ final[8:])
        scattered.append(float(sol.y[24, -1] + final[8:].sum()))
    ground = np.array(ground)
    return {"preparation_efficiency": float(ground[:, G_CLOCK].mean()),
            "minimum_atom_efficiency": float(ground[:, G_CLOCK].min()),
            "mean_scattered_photons": float(np.mean(scattered)),
            "ground_populations": ground}


def scan_candidates(config):
    """Declared grid; both detunings are relative to zero-field hyperfine lines."""
    rows = []
    detunings = [(-3e6, -3e6), (-1.5e6, -1.5e6), (0., 0.),
                 (1.5e6, 1.5e6), (3e6, 3e6), (-1.5e6, 0.), (0., -1.5e6)]
    for total_uW in (20., 40., 80., 160., 320., 640., 1280.):
        for transfer_fraction in (.25, .5, .75):
            for length_m in (.004, .008, .016, .025):
                for dt, dd in detunings:
                    options = replace(config.pumping, enabled=True,
                        transfer_power_W=total_uW * 1e-6 * transfer_fraction,
                        dark_power_W=total_uW * 1e-6 * (1 - transfer_fraction),
                        waist_z_m=length_m / (2 * config.pumping.half_widths),
                        transfer_detuning_Hz=dt, dark_detuning_Hz=dd)
                    rows.append({"case": "scan_%04d" % len(rows), "pumping": asdict(options)})
    return rows


def _screen_init(config_data, source_path):
    global _SCREEN_CONFIG, _SCREEN_SOURCE
    _SCREEN_CONFIG = SimulationConfig.from_dict(config_data)
    _SCREEN_SOURCE = read_stateful_ensemble(source_path)


def _screen_one(candidate):
    config = copy.deepcopy(_SCREEN_CONFIG)
    config.pumping = type(config.pumping)(**candidate["pumping"])
    config.validate()
    result = ballistic_screen(config, _SCREEN_SOURCE)
    result.pop("ground_populations")
    return dict(candidate, **result)


def screen_grid(config, ensemble, directory, workers=6, target_efficiency=.97, progress=print):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Fixed training subset. Large fresh MOT pool is never used to choose a candidate.
    training = subset_ensemble(ensemble, np.arange(0, len(ensemble.r_m), 2))
    path = directory / "screen_training.h5"
    save_stateful_ensemble(training, path)
    rows = scan_candidates(config)
    completed = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_screen_init,
            initargs=(config.to_dict(), str(path.resolve()))) as pool:
        futures = [pool.submit(_screen_one, row) for row in rows]
        for future in as_completed(futures):
            completed.append(future.result())
            if progress and len(completed) % 42 == 0:
                progress("24-level ballistic scan: %d/%d" % (len(completed), len(rows)))
    completed.sort(key=lambda row: row["case"])
    (directory / "screen.json").write_text(json.dumps({
        "method": "24-level PyLCP populations on prescribed ballistic trajectories; no recoil or force feedback",
        "minimum_training_mean_efficiency": target_efficiency,
        "training_atoms": len(training.r_m), "cases": completed,
    }, indent=2), encoding="utf-8")
    feasible = sorted((row for row in completed if row["preparation_efficiency"] >= target_efficiency),
                      key=lambda row: row["mean_scattered_photons"])
    if not feasible:
        raise RuntimeError("No feasible scan candidate; inspect scan rather than reducing efficiency silently")
    # Shortlist independent detuning pairs to avoid validating only near-duplicate minima.
    selected, pairs = [], set()
    for row in feasible:
        pair = (row["pumping"]["transfer_detuning_Hz"], row["pumping"]["dark_detuning_Hz"])
        if pair not in pairs:
            selected.append(row); pairs.add(pair)
    (directory / "shortlist.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    return selected


def _pump_one(job):
    data, atom_id, replica, seed, r, v, ground, start_time = job
    cfg = SimulationConfig.from_dict(data)
    model = _model(cfg)
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(20, replica, int(atom_id))))
    rr, vv, pop, dt, photons, _ = model.evolve_atom(r, v, ground, rng)
    return dict(atom_id=int(atom_id), r_m=rr, v_m_s=vv, ground=pop,
                reference_time_s=start_time + dt, photons=photons, interaction_time_s=dt,
                input_digest=pump_input_digest(r, v, ground, start_time))


def pump_input_digest(r, v, ground, start_time):
    return hashlib.sha256(np.r_[r, v, ground, start_time].astype(np.float64).tobytes()).hexdigest()


def prepare_native_batch(config, ensemble, directory, replicas=3, workers=6,
                         seed=61087, progress=print, executor=None):
    config.validate()
    if config.pumping.recoil_model not in ("diffusion", "compound_poisson"):
        raise ValueError("Optimization validation must include stochastic recoil")
    if np.linalg.norm(config.raman.rotation_rad_s) > 0:
        raise ValueError("Pumping validation currently requires zero rotation")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    identity = dict(pumping=asdict(config.pumping), algorithm=PUMP_ALGORITHM_VERSION,
                    acceleration=config.raman.acceleration_m_s2, seed=seed,
                    source_digest=ensemble.source_metadata.get("source_signature") or hashlib.sha256(
                        ensemble.r_m.tobytes() + ensemble.v_m_s.tobytes() + ensemble.ground_density.tobytes()).hexdigest())
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    manifest = directory / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text())["signature"] != signature:
        raise ValueError("Pumping cache settings differ; choose a new output directory")
    manifest.write_text(json.dumps(dict(signature=signature, **identity), indent=2), encoding="utf-8")
    config.save(directory / "config_used.json")
    model = _model(config)
    entry_time = flight_time(ensemble, model.start_z)
    r, v = trajectory(ensemble, entry_time, config.raman)
    ground = np.real(np.diagonal(ensemble.ground_density, axis1=1, axis2=2))
    ids = np.asarray(ensemble.source_metadata.get("sample_ids", np.arange(len(r))), dtype=int)
    jobs = []
    for replica in range(replicas):
        folder = directory / ("replica_%d" % replica)
        folder.mkdir(exist_ok=True)
        for i, atom_id in enumerate(ids):
            path = folder / ("atom_%06d.npz" % atom_id)
            if not path.exists():
                jobs.append((config.to_dict(), int(atom_id), replica, seed, r[i], v[i], ground[i], entry_time[i]))
            elif "source_signature" in ensemble.source_metadata:
                with np.load(path) as handle:
                    if ("input_digest" not in handle or str(handle["input_digest"])
                            != pump_input_digest(r[i], v[i], ground[i], entry_time[i])):
                        raise ValueError("Cached pumping atom input changed: " + str(path))
    done, total = replicas * len(r) - len(jobs), replicas * len(r)
    if progress:
        progress("Native pumping %s: %d cached, %d remaining" % (directory.name, done, len(jobs)))
    # A scan may share one Windows process pool across pump configurations.
    # The caller owns supplied executors; do not shut them down here.
    with (ProcessPoolExecutor(max_workers=workers) if executor is None else nullcontext(executor)) as pool:
        futures = {pool.submit(_pump_one, job): job[2] for job in jobs}
        for future in as_completed(futures):
            record = future.result()
            path = directory / ("replica_%d" % futures[future]) / ("atom_%06d.npz" % record["atom_id"])
            temporary = path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, **record)
            temporary.replace(path)
            done += 1
            if progress and (done % 64 == 0 or done == total):
                progress("Native pumping %s: %d/%d" % (directory.name, done, total))
    results = []
    for replica in range(replicas):
        records = []
        for atom_id in ids:
            with np.load(directory / ("replica_%d" % replica) / ("atom_%06d.npz" % atom_id)) as handle:
                records.append({key: handle[key] for key in handle.files})
        pop = np.array([record["ground"] for record in records])
        density = np.zeros((len(pop), 8, 8), complex)
        density[:, np.arange(8), np.arange(8)] = pop
        photons = np.array([record["photons"] for record in records])
        metadata = copy.deepcopy(ensemble.source_metadata)
        metadata["optical_pumping"] = dict(mean_scattered_photons=float(photons.mean()),
            preparation_efficiency=float(pop[:, G_CLOCK].mean()), config=asdict(config.pumping),
            replica=replica, seed=seed, seed_scheme="per-source-atom and recoil replica")
        output = Ensemble(np.array([record["r_m"] for record in records]),
            np.array([record["v_m_s"] for record in records]), ensemble.flux_per_s, metadata,
            ground_density=density, reference_plane_z_m=model.end_z,
            reference_time_s=np.array([record["reference_time_s"] for record in records]))
        save_stateful_ensemble(output, directory / ("prepared_%d.h5" % replica))
        cfg = copy.deepcopy(config)
        cfg.pumping.enabled = False
        result = run_simulation(cfg, output)
        results.append((result, photons))
    return results


def clustered_statistics(results, incident_samples, incident_flux_per_s, bootstrap=500, seed=90087):
    """Bootstrap independent source IDs; recoil replicas are averaged within ID.

    Linear per-atom fringe coefficients avoid fitting hundreds of whole scans.
    Failed source launches contribute zero flux, not fabricated detected atoms.
    """
    num = np.zeros((incident_samples, 3))
    target = np.zeros(incident_samples)
    useful = np.zeros(incident_samples)
    photons = np.zeros(incident_samples)
    extracted = np.zeros(incident_samples)
    for result, scatter in results:
        e, d = result["ensemble"], result["detection"]
        ids = np.asarray(e.source_metadata.get("sample_ids", np.arange(len(e.r_m))), dtype=int)
        keep = ids < incident_samples
        ids = ids[keep]
        phase = result["phase_rad"]
        design = np.column_stack((np.ones(len(phase)), np.cos(phase), np.sin(phase)))
        coefficients = np.linalg.lstsq(design, result["per_atom_P2"][:, keep], rcond=None)[0].T
        weight = d["detection_weights"][keep]
        num[ids] += coefficients * (weight / e.v_m_s[keep, 2])[:, None]
        pp = e.ground_density[keep, G_CLOCK, G_CLOCK].real
        target[ids] += pp
        useful[ids] += weight * pp
        photons[ids] += scatter[keep]
        extracted[ids] += 1
    for array in (num, target, useful, photons, extracted):
        array /= len(results)

    def estimate(counts):
        coefficients = counts @ num
        exits = counts @ extracted
        return np.column_stack((
            (counts @ target) / exits,
            (counts @ useful) / incident_samples * incident_flux_per_s,
            np.hypot(coefficients[:, 1], coefficients[:, 2]) / coefficients[:, 0],
            (counts @ photons) / exits))

    rng = np.random.default_rng(seed)
    counts = rng.multinomial(incident_samples, np.full(incident_samples, 1 / incident_samples), size=bootstrap)
    values = estimate(counts)
    point = estimate(np.ones((1, incident_samples)))[0]
    ci = np.quantile(values, [.025, .975], axis=0)
    names = ("preparation_efficiency", "effective_target_flux_per_s", "fringe_contrast", "mean_scattered_photons")
    summary = {name: float(value) for name, value in zip(names, point)}
    summary.update({name + "_95ci": ci[:, j].tolist() for j, name in enumerate(names)})
    summary.update(incident_samples=incident_samples, extracted_samples=int(extracted.sum()),
                   recoil_replicas=len(results), bootstrap_clusters="independent incident source atoms")
    return summary, values


def validate_shortlist(config, ensemble, directory, workers=6, replicas=3, progress=print):
    directory = Path(directory)
    selected = json.loads((directory / "shortlist.json").read_text(encoding="utf-8"))
    cases = [{"case": "baseline_4mW", "pumping": asdict(config.pumping)}] + selected
    rows, draws = [], {}
    for case in cases:
        c = copy.deepcopy(config)
        c.pumping = type(c.pumping)(**case["pumping"])
        c.pumping.diffusion_quadrature_order = 3
        c.pumping.recoil_chunk_s = 25e-6
        c.pumping.rate_backend = "fixed_axis"
        c.pumping.enabled = True
        results = prepare_native_batch(c, ensemble, directory / "training_fixed_axis_q3" / case["case"],
            workers=workers, replicas=replicas, progress=progress)
        metrics, draws[case["case"]] = clustered_statistics(results, len(ensemble.r_m), ensemble.flux_per_s,
                                                             bootstrap=1500)
        rows.append(dict(case, pumping=asdict(c.pumping), **metrics))
        if progress:
            progress("Validated %s: eta=%.3f%%, C=%.3f%%, photons=%.2f" % (
                case["case"], 100 * metrics["preparation_efficiency"],
                100 * metrics["fringe_contrast"], metrics["mean_scattered_photons"]))
    baseline = rows[0]
    feasible = [row for row in rows[1:] if row["preparation_efficiency"] >= .95
                and row["effective_target_flux_per_s"] >= .95 * baseline["effective_target_flux_per_s"]
                and row["mean_scattered_photons"] <= .5 * baseline["mean_scattered_photons"]]
    if not feasible:
        raise RuntimeError("Native validation has no feasible improvement; inspect training results")
    best = max(feasible, key=lambda row: row["fringe_contrast_95ci"][0])
    delta = draws[best["case"]][:, 2] - draws["baseline_4mW"][:, 2]
    report = dict(cases=rows, selected_case=best["case"],
        selection_rule="eta>=95%, useful flux>=95% of baseline, photons<=50% of baseline; maximize lower cluster-bootstrap contrast bound",
        selection_data="Original 32 MOT atoms only; fresh large source pool is held out",
        training_contrast_improvement_95ci=np.quantile(delta, [.025, .975]).tolist())
    (directory / "training_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    best_config = copy.deepcopy(config)
    best_config.pumping = type(best_config.pumping)(**best["pumping"])
    best_config.pumping.enabled = True
    best_config.save(directory / "optimized_config.json")
    return best_config


def convergence_study(config, source, directory, workers=6, replicas=3, progress=print):
    directory = Path(directory)
    selected = SimulationConfig.from_dict(json.loads((directory / "optimized_config.json").read_text(encoding="utf-8")))
    # All selection is finished before exposing the held-out source trajectories.
    config.raman = copy.deepcopy(selected.raman)
    config.raman.integration_steps = selected.raman.integration_steps = 192
    config.raman.scan_points = selected.raman.scan_points = 41
    config.detection.noise = selected.detection.noise = False
    config.pumping.enabled = selected.pumping.enabled = True
    config.pumping.recoil_chunk_s = 25e-6
    config.pumping.diffusion_quadrature_order = 3
    config.pumping.rate_backend = selected.pumping.rate_backend = "fixed_axis"
    config.save(directory / "baseline_validation_config.json")
    selected.save(directory / "optimized_validation_config.json")
    cases = {}
    for name, cfg in (("baseline", config), ("optimized", selected)):
        cases[name] = prepare_native_batch(cfg, source, directory / "held_out" / name,
            replicas=replicas, workers=workers, progress=progress)
    total = source.source_metadata["sampled"]
    counts = [n for n in (32, 64, 128, 256, 512, 1024, 2048, 4096) if n <= total]
    if total not in counts:
        counts.append(total)
    rows, contrasts = [], []
    for count in counts:
        summaries, draws = {}, {}
        for name, results in cases.items():
            summary, values = clustered_statistics(results, count, config.source.incident_flux_per_s,
                                                    bootstrap=2000)
            rows.append(dict(case=name, **summary))
            summaries[name], draws[name] = summary, values
        contrast_delta = draws["optimized"][:, 2] - draws["baseline"][:, 2]
        eta_delta = draws["optimized"][:, 0] - draws["baseline"][:, 0]
        flux_ratio = draws["optimized"][:, 1] / draws["baseline"][:, 1]
        photon_ratio = draws["optimized"][:, 3] / draws["baseline"][:, 3]
        contrasts.append(dict(incident_samples=count,
            contrast_improvement=summaries["optimized"]["fringe_contrast"] - summaries["baseline"]["fringe_contrast"],
            contrast_improvement_95ci=np.quantile(contrast_delta, [.025, .975]).tolist(),
            efficiency_change_95ci=np.quantile(eta_delta, [.025, .975]).tolist(),
            useful_flux_ratio=summaries["optimized"]["effective_target_flux_per_s"] / summaries["baseline"]["effective_target_flux_per_s"],
            useful_flux_ratio_95ci=np.quantile(flux_ratio, [.025, .975]).tolist(),
            photon_ratio=summaries["optimized"]["mean_scattered_photons"] / summaries["baseline"]["mean_scattered_photons"],
            photon_ratio_95ci=np.quantile(photon_ratio, [.025, .975]).tolist()))
    report = dict(cases=rows, paired_differences=contrasts,
        source_seed=source.source_metadata["source_seed"],
        source_trajectories=total, actual_extracted=len(source.r_m), recoil_replicas=replicas,
        method="Nested fresh incident-source prefixes; cluster bootstrap over source ID, average recoil replicas within ID",
        scope="Convergence of modeled MOT source sampling and stochastic pumping, not experimental systematics")
    (directory / "convergence.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    from .study_report import write_report
    write_report(report, config, selected, directory, cases)
    return report


def load_prepared_results(config, directory, incident_samples, replicas):
    results = []
    cfg = copy.deepcopy(config)
    cfg.pumping.enabled = False
    for replica in range(replicas):
        e = source_prefix(read_stateful_ensemble(Path(directory) / ("prepared_%d.h5" % replica)), incident_samples)
        scatter = []
        for atom_id in e.source_metadata["sample_ids"]:
            with np.load(Path(directory) / ("replica_%d" % replica) / ("atom_%06d.npz" % atom_id)) as handle:
                scatter.append(float(handle["photons"]))
        results.append((run_simulation(cfg, e), np.array(scatter)))
    return results


def numerical_controls(source, directory, workers=6, replicas=3, progress=print):
    directory = Path(directory)
    controls = []
    for name in ("baseline", "optimized"):
        config = SimulationConfig.from_dict(json.loads(
            (directory / (name + "_validation_config.json")).read_text(encoding="utf-8")))
        count = min(128, source.source_metadata["sampled"])
        existing = load_prepared_results(config, directory / "held_out" / name, count, replicas)
        original, draws = clustered_statistics(existing, count, config.source.incident_flux_per_s, bootstrap=1500)
        config.pumping.recoil_chunk_s /= 2
        finer = prepare_native_batch(config, source_prefix(source, count), directory / "controls" / (name + "_half_step"),
            replicas=replicas, workers=workers, progress=progress)
        refined, fine_draws = clustered_statistics(finer, count, config.source.incident_flux_per_s, bootstrap=1500)
        delta = fine_draws[:, 2] - draws[:, 2]
        controls.append(dict(test=name + "_25_vs_12p5us", original=original, modified=refined,
                             contrast_difference_95ci=np.quantile(delta, [.025, .975]).tolist()))
    config = SimulationConfig.from_dict(json.loads((directory / "optimized_validation_config.json").read_text(encoding="utf-8")))
    count = min(512, source.source_metadata["sampled"])
    gaussian = load_prepared_results(config, directory / "held_out" / "optimized", count, replicas)
    original, draws = clustered_statistics(gaussian, count, config.source.incident_flux_per_s, bootstrap=1500)
    config.pumping.recoil_model = "compound_poisson"
    discrete = prepare_native_batch(config, source_prefix(source, count), directory / "controls" / "optimized_compound_poisson",
        replicas=replicas, workers=workers, progress=progress)
    modified, discrete_draws = clustered_statistics(discrete, count, config.source.incident_flux_per_s, bootstrap=1500)
    controls.append(dict(test="optimized_Gaussian_vs_compound_Poisson", original=original, modified=modified,
        contrast_difference_95ci=np.quantile(discrete_draws[:, 2] - draws[:, 2], [.025, .975]).tolist()))
    result = dict(controls=controls,
        interpretation="Recoil chunk changes use different random paths: these compare weak statistics, not per-atom strong convergence",
        poisson_scope="Compound Poisson recoil on native mean populations; not a state-resolved quantum jump trajectory")
    (directory / "numerical_controls.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    from .study_report import append_controls
    append_controls(directory)
    return result


def training_robustness(source, directory, workers=6, replicas=3, progress=print):
    """Validate the independently selected 12-replica training candidate as well."""
    directory = Path(directory)
    cfg = SimulationConfig.from_dict(json.loads(
        (directory / "refined_training" / "optimized_config.json").read_text(encoding="utf-8")))
    cfg.raman.scan_points, cfg.raman.integration_steps = 41, 192
    cfg.detection.noise = False
    cfg.pumping.rate_backend = "fixed_axis"
    cfg.save(directory / "refined_validation_config.json")
    total = source.source_metadata["sampled"]
    results = {}
    for name in ("baseline", "optimized"):
        cc = SimulationConfig.from_dict(json.loads((directory / (name + "_validation_config.json")).read_text(encoding="utf-8")))
        results[name] = load_prepared_results(cc, directory / "held_out" / name, total, replicas)
    results["refined"] = prepare_native_batch(cfg, source, directory / "held_out" / "refined",
        replicas=replicas, workers=workers, progress=progress)
    sizes = [n for n in (32, 64, 128, 256, 512, 1024, 2048, 4096) if n <= total]
    if total not in sizes:
        sizes.append(total)
    rows, comparisons = [], []
    for count in sizes:
        stats, draws = {}, {}
        for name, result in results.items():
            stats[name], draws[name] = clustered_statistics(result, count,
                cfg.source.incident_flux_per_s, bootstrap=2500)
            rows.append(dict(case=name, **stats[name]))
        for name in ("optimized", "refined"):
            delta = draws[name][:, 2] - draws["baseline"][:, 2]
            comparisons.append(dict(incident_samples=count, case=name,
                contrast_improvement=stats[name]["fringe_contrast"] - stats["baseline"]["fringe_contrast"],
                contrast_improvement_95ci=np.quantile(delta, [.025, .975]).tolist(),
                # Two candidate comparisons: Bonferroni family-wise 95% intervals.
                contrast_improvement_family95ci=np.quantile(delta, [.0125, .9875]).tolist(),
                useful_flux_ratio_95ci=np.quantile(draws[name][:, 1] / draws["baseline"][:, 1], [.025, .975]).tolist(),
                photon_ratio_95ci=np.quantile(draws[name][:, 3] / draws["baseline"][:, 3], [.025, .975]).tolist()))
    final_rows = {row["case"]: row for row in rows if row["incident_samples"] == total}
    final_comparisons = {row["case"]: row for row in comparisons if row["incident_samples"] == total}
    def feasible(name):
        row, difference = final_rows[name], final_comparisons[name]
        return (row["preparation_efficiency_95ci"][0] >= .95
                and difference["useful_flux_ratio_95ci"][0] >= .95
                and difference["photon_ratio_95ci"][1] < .5
                and difference["contrast_improvement_family95ci"][0] > 0)
    # Prefer the refined training choice if independently confirmed. Do not
    # maximize validation point estimates across this candidate family.
    recommendation = "refined" if feasible("refined") else ("optimized" if feasible("optimized") else None)
    report = dict(cases=rows, paired_differences=comparisons, recommended_case=recommendation,
        source_trajectories=total, recoil_replicas=replicas,
        candidate_selection="Both candidates selected from the original 32 atoms, with 3 or 12 training recoil replicas",
        uncertainty="Source-cluster bootstrap; two baseline comparisons use Bonferroni family-wise 95% intervals")
    (directory / "training_robustness.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if recommendation:
        chosen = cfg if recommendation == "refined" else SimulationConfig.from_dict(json.loads(
            (directory / "optimized_validation_config.json").read_text(encoding="utf-8")))
        chosen.save(directory / "recommended_config.json")
    from .study_report import write_final_report
    write_final_report(report, directory)
    return report


def kinetic_diagnostics(source, directory, replicas=3):
    """Compare velocity variances at the same first-Raman plane, not different exits."""
    from .physics import KB, MASS
    directory = Path(directory)
    cfg = SimulationConfig.from_dict(json.loads((directory / "baseline_validation_config.json").read_text(encoding="utf-8")))
    total = source.source_metadata["sampled"]
    rng = np.random.default_rng(31087)
    counts = rng.multinomial(total, np.full(total, 1 / total), size=2000)
    cases = {"source_free_flight": [source]}
    for name in ("baseline", "optimized", "refined"):
        cases[name] = [read_stateful_ensemble(directory / "held_out" / name / ("prepared_%d.h5" % replica))
                       for replica in range(replicas)]
    rows, draws = {}, {}
    for name, ensembles in cases.items():
        mean, square, exits = np.zeros((total, 3)), np.zeros((total, 3)), np.zeros(total)
        for e in ensembles:
            ids = np.asarray(e.source_metadata["sample_ids"])
            _, v = trajectory(e, flight_time(e, cfg.raman.first_zone_z_m), cfg.raman)
            mean[ids] += v
            square[ids] += v ** 2
            exits[ids] += 1
        mean /= len(ensembles); square /= len(ensembles); exits /= len(ensembles)
        def estimate(ww):
            norm = (ww @ exits)[:, None]
            return MASS / KB * ((ww @ square) / norm - ((ww @ mean) / norm) ** 2)
        point = estimate(np.ones((1, total)))[0]
        draws[name] = estimate(counts)
        ci = np.quantile(draws[name], [.025, .975], axis=0)
        rows[name] = dict(temperature_xyz_K=point.tolist(), temperature_xyz_95ci_K=ci.T.tolist())
    differences = {name: np.quantile(draws[name] - draws["baseline"], [.025, .975], axis=0).T.tolist()
                   for name in ("optimized", "refined")}
    result = dict(cases=rows, candidate_minus_baseline_temperature_xyz_95ci_K=differences,
        plane_z_m=cfg.raman.first_zone_z_m,
        interpretation="Unconditional velocity-variance temperature at one common plane; includes mean optical friction as well as recoil")
    (directory / "kinetic_diagnostics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    from .study_report import write_final_report
    write_final_report(json.loads((directory / "training_robustness.json").read_text(encoding="utf-8")), directory)
    return result


def main():
    import argparse
    from .presets import pylcp_mot_preset
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("screen", "validate", "convergence", "controls", "robustness", "kinetic"), required=True)
    parser.add_argument("--training-source", type=Path,
        default=Path("beam_outputs/pylcp_multilevel_32/source_ensemble.h5"))
    parser.add_argument("--output", type=Path, default=Path("beam_outputs/pumping_optimization"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--replicas", type=int, default=3)
    parser.add_argument("--source", type=Path,
        default=Path("beam_outputs/pumping_optimization/source/source_ensemble.h5"))
    args = parser.parse_args()
    config = pylcp_mot_preset()
    config.pumping.enabled = True
    config.detection.noise = False
    # Eight phase points determine the sinusoid; verify finalists at 41 and finer time steps.
    config.raman.scan_points = 9
    progress = lambda message: print(message, flush=True)
    if args.stage == "screen":
        ensemble = read_stateful_ensemble(args.training_source)
        screen_grid(config, ensemble, args.output, workers=args.workers, progress=progress)
    elif args.stage == "validate":
        ensemble = read_stateful_ensemble(args.training_source)
        validate_shortlist(config, ensemble, args.output, workers=args.workers,
                           replicas=args.replicas, progress=progress)
    elif args.stage == "convergence":
        convergence_study(config, read_stateful_ensemble(args.source), args.output,
                          workers=args.workers, replicas=args.replicas, progress=progress)
    elif args.stage == "controls":
        numerical_controls(read_stateful_ensemble(args.source), args.output,
                           workers=args.workers, replicas=args.replicas, progress=progress)
    elif args.stage == "robustness":
        training_robustness(read_stateful_ensemble(args.source), args.output,
                            workers=args.workers, replicas=args.replicas, progress=progress)
    else:
        kinetic_diagnostics(read_stateful_ensemble(args.source), args.output, replicas=args.replicas)


if __name__ == "__main__":
    main()
