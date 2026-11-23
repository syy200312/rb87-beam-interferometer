"""Source-cluster statistics for constrained pump/Raman co-design.

The resampling unit is an incident MOT source ID. Recoil replicas are averaged
within that ID before any bootstrap; the same counts are used for all declared
perturbation scenarios. Intervals describe finite source Monte Carlo sampling,
conditional on the scenarios and model, not laboratory noise or GP uncertainty.
"""

import hashlib

import numpy as np


_FIELDS = ("atom_ids", "fringe_coefficients", "detection_weights", "vz_m_s",
           "target_population", "scattered_photons")
_LIMITS = ("min_efficiency", "min_effective_target_flux_per_s", "max_scattered_photons")


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(name + " must be a positive integer")
    return int(value)


def _finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _interval(values):
    # Never silently condition an interval on only the finite bootstrap draws.
    # E.g. resampling a small extracted population may leave zero detected atoms.
    values = np.asarray(values)
    if not np.all(np.isfinite(values)):
        return None
    return np.quantile(values, [.025, .975]).tolist()


def _ratio(numerator, denominator):
    out = np.full(np.shape(numerator), np.nan, dtype=float)
    return np.divide(numerator, denominator, out=out, where=np.asarray(denominator) > 1e-15)


def _cluster_contributions(replicas, incident_samples):
    if not isinstance(replicas, (tuple, list)) or not replicas:
        raise ValueError("Each scenario needs a nonempty list of recoil replicas")
    # Columns: weighted A,B,D; photon weight; target population; accepted target;
    # scattered photons; extracted indicator; detector weight. Absent IDs are 0.
    data = np.zeros((incident_samples, 10), dtype=float)
    for replica in replicas:
        if not isinstance(replica, dict) or set(_FIELDS) - set(replica):
            raise ValueError("A replica must contain " + ", ".join(_FIELDS))
        raw_ids = np.asarray(replica["atom_ids"])
        if raw_ids.ndim != 1 or raw_ids.dtype.kind not in "iu" and raw_ids.size:
            raise ValueError("atom_ids must be a one-dimensional integer array")
        ids = raw_ids.astype(np.int64)
        if np.any(ids < 0) or np.any(ids >= incident_samples) or len(np.unique(ids)) != len(ids):
            raise ValueError("Source IDs must be unique in a replica and inside the incident sample range")
        n = len(ids)
        coefficients = np.asarray(replica["fringe_coefficients"], dtype=float)
        if coefficients.shape != (n, 3) or not np.all(np.isfinite(coefficients)):
            raise ValueError("fringe_coefficients must have finite shape (N,3)")
        arrays = {}
        for name in _FIELDS[2:]:
            value = np.asarray(replica[name], dtype=float)
            if value.shape != (n,) or not np.all(np.isfinite(value)):
                raise ValueError(name + " must have finite shape (N,)")
            arrays[name] = value
        weight, speed = arrays["detection_weights"], arrays["vz_m_s"]
        target, scattered = arrays["target_population"], arrays["scattered_photons"]
        if np.any(speed <= 0):
            raise ValueError("Longitudinal speeds must be positive")
        if np.any(weight < 0) or np.any(weight > 1 + 1e-9):
            raise ValueError("Detection weights must lie in [0,1]")
        if np.any(target < 0) or np.any(target > 1 + 1e-9) or np.any(scattered < 0):
            raise ValueError("Target populations must lie in [0,1] and photon counts must be nonnegative")
        amplitude = np.hypot(coefficients[:, 1], coefficients[:, 2])
        if np.any(coefficients[:, 0] - amplitude < -1e-7) or np.any(coefficients[:, 0] + amplitude > 1 + 1e-7):
            raise ValueError("Fringe coefficients must describe probabilities in [0,1]")
        photon_weight = weight / speed
        data[ids, :3] += coefficients * photon_weight[:, None]
        data[ids, 3] += photon_weight
        data[ids, 4] += target
        data[ids, 5] += weight * target
        data[ids, 6] += scattered
        data[ids, 7] += 1
        data[ids, 8] += weight
        # Keep an unweighted prepared target contribution for the source exit.
        data[ids, 9] += target
    return data / len(replicas)


def _estimates(counts, contribution, incident_samples, incident_flux_per_s):
    sums = counts @ contribution
    a, b, d = (_ratio(sums[:, i], sums[:, 3]) for i in range(3))
    amplitude = np.hypot(b, d)
    flux_scale = incident_flux_per_s / incident_samples
    exit_flux = sums[:, 7] * flux_scale
    detector_flux = sums[:, 8] * flux_scale
    # A zero-detection bootstrap sample has zero response, not infinite contrast.
    score = np.zeros(len(sums))
    modulated_flux = np.zeros(len(sums))
    active = sums[:, 3] > 1e-15
    score[active] = 2 * amplitude[active] * np.sqrt(detector_flux[active] / incident_flux_per_s)
    modulated_flux[active] = 2 * amplitude[active] * detector_flux[active]
    return {
        "preparation_efficiency": _ratio(sums[:, 4], sums[:, 7]),
        "exit_flux_per_s": exit_flux,
        "detected_flux_per_s": detector_flux,
        "effective_target_flux_per_s": sums[:, 5] * flux_scale,
        "exit_target_flux_per_s": sums[:, 9] * flux_scale,
        "extraction_fraction": sums[:, 7] / incident_samples,
        "offset": a, "cosine_coefficient": b, "sine_coefficient": d,
        "fringe_amplitude": amplitude, "fringe_contrast": _ratio(amplitude, a),
        "mean_scattered_photons": _ratio(sums[:, 6], sums[:, 7]),
        "modulated_flux_per_s": modulated_flux, "score": score,
    }


def _summary(point, draws):
    result = {}
    valid = {}
    for name, values in draws.items():
        result[name] = _finite_or_none(point[name][0])
        result[name + "_95ci"] = _interval(values)
        valid[name] = float(np.mean(np.isfinite(values)))
    result["valid_bootstrap_fraction"] = valid
    return result


def _robust_estimates(scenarios):
    # A minimum/max is evaluated inside each draw, preserving scenario correlation.
    def extremum(metric, function):
        return function(np.stack([scenario[metric] for scenario in scenarios.values()]), axis=0)

    return {
        "worst_score": extremum("score", np.min),
        "worst_contrast": extremum("fringe_contrast", np.min),
        "min_efficiency": extremum("preparation_efficiency", np.min),
        "min_effective_target_flux_per_s": extremum("effective_target_flux_per_s", np.min),
        "min_detected_flux_per_s": extremum("detected_flux_per_s", np.min),
        "max_scattered_photons": extremum("mean_scattered_photons", np.max),
    }


def _constraint_margins(robust, constraints):
    margins = {
        "efficiency": robust["min_efficiency"] - constraints["min_efficiency"],
        "effective_target_flux_per_s": (robust["min_effective_target_flux_per_s"]
                                         - constraints["min_effective_target_flux_per_s"]),
        "scattered_photons": constraints["max_scattered_photons"] - robust["max_scattered_photons"],
    }
    if "min_fringe_contrast" in constraints:
        margins["fringe_contrast"] = robust["worst_contrast"] - constraints["min_fringe_contrast"]
    return margins


def case_statistics(scenarios, incident_samples, incident_flux_per_s, constraints,
                    bootstrap=1000, seed=90087):
    """Compute point estimates and conditional 95% source-cluster intervals.

    ``scenarios`` maps names to lists of replica dictionaries. Each replica has
    incident ``atom_ids``, per-atom coefficients [A,B,D] of F2 probability,
    detection weights, longitudinal speeds, target populations and scattered
    photon numbers. A missing incident ID contributes zero exit/detected flux.

    Every scenario shares one multinomial count matrix. Repeated recoil paths
    are averaged by ID, including missing-path zero contributions, before these
    counts are applied. Matching before/after calls must refer to the same
    physical source IDs; the function can check counts, not source provenance.
    """
    incident_samples = _integer(incident_samples, "incident_samples")
    bootstrap = _integer(bootstrap, "bootstrap")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not np.isfinite(incident_flux_per_s) or incident_flux_per_s <= 0:
        raise ValueError("incident_flux_per_s must be positive and finite")
    if not isinstance(scenarios, dict) or not scenarios or any(not isinstance(k, str) or not k for k in scenarios):
        raise ValueError("scenarios must be a nonempty dictionary with nonempty string names")
    if not isinstance(constraints, dict) or set(_LIMITS) - set(constraints):
        raise ValueError("Constraints require " + ", ".join(_LIMITS))
    for key in _LIMITS:
        if not np.isfinite(constraints[key]) or constraints[key] < 0:
            raise ValueError(key + " must be nonnegative and finite")
    if constraints["min_efficiency"] > 1:
        raise ValueError("min_efficiency must lie in [0,1]")
    if "min_fringe_contrast" in constraints:
        if not np.isfinite(constraints["min_fringe_contrast"]) or not 0 <= constraints["min_fringe_contrast"] <= 1:
            raise ValueError("min_fringe_contrast must lie in [0,1]")
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(incident_samples, np.full(incident_samples, 1 / incident_samples), size=bootstrap)
    identity = {"incident_samples": incident_samples, "bootstrap": bootstrap, "seed": int(seed),
                "counts_sha256": hashlib.sha256(counts.astype("<i8").tobytes()).hexdigest()}
    point_counts = np.ones((1, incident_samples))
    all_point, all_draws, scenario_summary = {}, {}, {}
    for name, replicas in scenarios.items():
        contribution = _cluster_contributions(replicas, incident_samples)
        point = _estimates(point_counts, contribution, incident_samples, incident_flux_per_s)
        draws = _estimates(counts, contribution, incident_samples, incident_flux_per_s)
        all_point[name], all_draws[name] = point, draws
        scenario_summary[name] = _summary(point, draws)
        scenario_summary[name].update(recoil_replicas=len(replicas),
                                      mean_extracted_samples=float(contribution[:, 7].sum()))
    robust_point, robust_draws = _robust_estimates(all_point), _robust_estimates(all_draws)
    robust_summary = _summary(robust_point, robust_draws)
    margins_point = _constraint_margins(robust_point, constraints)
    margins_draws = _constraint_margins(robust_draws, constraints)
    robust_summary["constraint_margins"] = {key: {"value": _finite_or_none(value[0]),
        "95ci": _interval(margins_draws[key])} for key, value in margins_point.items()}
    robust_summary["feasible"] = bool(all(np.isfinite(value[0]) and value[0] >= 0
                                          for value in margins_point.values()))
    feasible_draws = np.logical_and.reduce([np.isfinite(value) & (value >= 0) for value in margins_draws.values()])
    robust_summary["bootstrap_constraint_feasible_fraction"] = float(feasible_draws.mean())
    summary = {"scenarios": scenario_summary, "robust": robust_summary,
               "incident_samples": incident_samples, "incident_flux_per_s": float(incident_flux_per_s),
               "constraints": {key: float(constraints[key]) for key in
                               _LIMITS + (("min_fringe_contrast",) if "min_fringe_contrast" in constraints else ())},
               "bootstrap_identity": identity,
               "bootstrap_clusters": "Independent incident source IDs; recoil replicas averaged within source ID",
               "score_definition": "2*fringe_amplitude*sqrt(detected_flux_per_s/incident_flux_per_s)",
               "interval_scope": "Source Monte Carlo sampling conditional on the declared scenarios and model; "
                                 "not a guarantee over a continuous error box or a GP posterior. "
                                 "Undefined bootstrap values produce a null interval, without conditioning them away."}
    draws = dict(robust_draws, scenarios=all_draws, constraint_margins=margins_draws, bootstrap_identity=identity)
    return summary, draws


def paired_comparison(before_summary, before_draws, after_summary, after_draws):
    """Paired source-cluster changes; shared underlying source provenance is required."""
    identities = [value.get("bootstrap_identity") for value in
                  (before_summary, before_draws, after_summary, after_draws)]
    if any(identity is None or identity != identities[0] for identity in identities):
        raise ValueError("Paired comparison requires identical source count, bootstrap seed and resampling counts")
    if before_summary.get("incident_flux_per_s") != after_summary.get("incident_flux_per_s"):
        raise ValueError("Paired cases must use the same incident-flux normalization")
    if set(before_draws["scenarios"]) != set(after_draws["scenarios"]):
        raise ValueError("Paired cases must use the same declared scenarios")
    if "nominal" not in before_draws["scenarios"]:
        raise ValueError("Paired comparison requires a nominal scenario")
    result = {"bootstrap_identity": identities[0],
              "pairing": "Same incident source IDs and bootstrap count matrix; caller guarantees source provenance"}
    metrics = (
        ("worst_score_improvement", before_summary["robust"]["worst_score"], after_summary["robust"]["worst_score"],
         before_draws["worst_score"], after_draws["worst_score"]),
        ("nominal_contrast_improvement", before_summary["scenarios"]["nominal"]["fringe_contrast"],
         after_summary["scenarios"]["nominal"]["fringe_contrast"],
         before_draws["scenarios"]["nominal"]["fringe_contrast"],
         after_draws["scenarios"]["nominal"]["fringe_contrast"]),
        ("nominal_effective_target_flux_improvement_per_s",
         before_summary["scenarios"]["nominal"]["effective_target_flux_per_s"],
         after_summary["scenarios"]["nominal"]["effective_target_flux_per_s"],
         before_draws["scenarios"]["nominal"]["effective_target_flux_per_s"],
         after_draws["scenarios"]["nominal"]["effective_target_flux_per_s"]),
    )
    for name, before, after, before_values, after_values in metrics:
        before_values, after_values = np.asarray(before_values), np.asarray(after_values)
        if before_values.shape != after_values.shape or before_values.shape != (identities[0]["bootstrap"],):
            raise ValueError("Paired bootstrap draw arrays must have matching declared length")
        delta = after_values - before_values
        interval = _interval(delta)
        result[name] = (float(after - before) if before is not None and after is not None else None)
        result[name + "_95ci"] = interval
        result[name + "_supported"] = bool(interval is not None and interval[0] > 0)
        result[name + "_valid_bootstrap_fraction"] = float(np.mean(np.isfinite(delta)))
    return result
