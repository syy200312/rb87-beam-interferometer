"""Compare the three-pulse chirp fringe before and after technical noise."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
from dataclasses import asdict, replace
from functools import partial
import json
from pathlib import Path

import numpy as np

from .chirp_noise_propagation import chirp_probabilities
from .config import SimulationConfig
from .io import read_stateful_ensemble
from .physics import flight_time
from .raman_noise_metrics import draw_fluorescence, make_detector
from .raman_noise_study import (DEFAULT_CONFIG, DEFAULT_NOISE, DEFAULT_SOURCE,
                                NoiseSettings, _write_table, sample_disturbances)


def _one_chirp(task, *, ensemble, raman, settings, draws, detector):
    index, slope, verify = task
    # A common set of technical realizations gives a smooth estimate of the
    # expectation versus chirp. Observed example gates use different records.
    check = ()
    if verify:
        largest_power = np.abs(draws["intensity"] - 1).max(axis=1).argmax()
        largest_field = np.abs(draws["magnetic"]).max(axis=1).argmax()
        check = tuple({0, int(largest_power), int(largest_field)})
    nominal, noisy, error = chirp_probabilities(ensemble, raman, slope, draws,
                                               settings.bias_field_G, check_indices=check)
    for prob in (nominal, noisy):
        if not np.isfinite(prob).all() or np.min(prob) < -1e-8 or np.max(prob) > 1 + 1e-8:
            raise ValueError("Invalid propagated F2 probability")
    nominal, noisy = np.clip(nominal, 0, 1), np.clip(noisy, 0, 1)
    photon_weights = detector.atom_counts * detector.q2
    photon_weights /= photon_weights.sum()
    before = float(np.dot(nominal, photon_weights))
    after = np.einsum("ra,a->r", noisy, photon_weights)
    row = dict(chirp_Hz_s=float(slope), baseline_mean_P2=before, combined_mean_P2=float(after.mean()),
               combined_mean_se_P2=float(after.std(ddof=1) / np.sqrt(len(after))))
    for stream, (label, prob) in enumerate((("baseline", np.broadcast_to(nominal, noisy.shape)),
                                            ("combined", noisy))):
        seed = np.random.SeedSequence(settings.seed, spawn_key=(17, index, stream))
        rng = np.random.default_rng(seed)
        observed = draw_fluorescence(prob, detector, rng)
        # One uniformly selected gate at each grid point makes a single sweep.
        example = int(rng.integers(len(observed)))
        finite = np.isfinite(observed)
        quantiles = np.quantile(observed[finite], [.025, .975]) if finite.any() else (np.nan, np.nan)
        row[label + "_example_P2"] = float(observed[example])
        row[label + "_q025"], row[label + "_q975"] = map(float, quantiles)
        row[label + "_valid_gates"] = int(finite.sum())
    row["combined_minus_baseline_P2"] = row["combined_mean_P2"] - before
    row["exact_check_max_abs_P2_error"] = float(error) if verify else None
    return row, after


def _curve_summary(alpha, mean):
    i, j = int(np.argmin(mean)), int(np.argmax(mean))
    lo, hi = float(mean[i]), float(mean[j])
    return dict(min_P2=lo, min_at_Hz_s=float(alpha[i]), max_P2=hi, max_at_Hz_s=float(alpha[j]),
                visibility=(hi-lo)/(hi+lo) if hi+lo > 0 else None)


def _visibility_se(records):
    """Delete-one-realization standard error of extrema on the same chirp grid."""
    count = len(records)
    mean = np.mean(records, axis=0)
    jack = mean + (mean - records)/(count-1)
    lo, hi = jack.min(axis=1), jack.max(axis=1)
    if np.any(hi+lo <= 0):
        return None
    values = (hi-lo)/(hi+lo)
    return float(np.sqrt((count-1) * np.mean((values-values.mean())**2)))


def run(config, ensemble, settings, output, *, workers=1, plot=True, progress=print, inputs=None):
    config = copy.deepcopy(config)
    config.pumping.enabled = False
    config.detection.gate_s = settings.gate_s
    config.detection.noise = True
    config.validate()
    settings.validate()
    ensemble.validate()
    if workers < 1:
        raise ValueError("workers must be positive")
    r, scan = config.raman, config.chirp_scan
    span = r.waist_z_m*r.envelope_half_widths if r.envelope == "gaussian" else r.rectangular_width_m/2
    first_edge = r.first_zone_z_m - span
    if first_edge <= ensemble.reference_plane_z_m:
        raise ValueError("The input ensemble must precede the first Raman pulse")
    transit = np.max(flight_time(ensemble, config.detection.z_m)-flight_time(ensemble, first_edge))
    if settings.record_period_s + 1e-12 < settings.gate_s + transit:
        raise ValueError("Independent records must cover the gate and the slowest Raman-to-detector flight")
    alpha = np.linspace(scan.center_Hz_s-scan.half_range_Hz_s, scan.center_Hz_s+scan.half_range_Hz_s, scan.points)
    reference_T = r.zone_spacing_m / r.reference_vz_m_s
    if (alpha[1]-alpha[0])*reference_T**2 >= .5:
        raise ValueError("Increase chirp_scan.points to resolve the reference fringe")
    detector = make_detector(ensemble, config)
    evaluate = partial(_one_chirp, ensemble=ensemble, raman=r, settings=settings,
                       draws=sample_disturbances(settings), detector=detector)
    checked = {0, len(alpha)//2, len(alpha)-1}
    tasks = [(i, a, i in checked) for i, a in enumerate(alpha)]
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    collected = []
    try:
        results = pool.map(evaluate, tasks) if pool else map(evaluate, tasks)
        for row, values in results:
            collected.append((row, values))
            if progress and (len(collected) == 1 or len(collected) % 10 == 0 or len(collected) == len(alpha)):
                progress(f"Chirp points {len(collected)}/{len(alpha)}")
    finally:
        if pool:
            pool.shutdown()
    rows = [pair[0] for pair in collected]
    by_record = np.stack([pair[1] for pair in collected], axis=1)
    base = _curve_summary(alpha, [row["baseline_mean_P2"] for row in rows])
    combined = _curve_summary(alpha, [row["combined_mean_P2"] for row in rows])
    report = dict(inputs=inputs, noise_settings=asdict(settings), beam_config=config.to_dict(),
                  trajectories=len(ensemble.r_m), chirp_points=len(alpha), reference_T_s=reference_T,
                  third_zone_phase_offset_rad=r.phase_offsets_rad[2], accepted_flux_per_s=detector.accepted_flux_per_s,
                  target_flux_per_s=detector.target_flux_per_s, baseline=base, combined=combined,
                  combined_visibility_mc_se=_visibility_se(by_record),
                  max_checked_abs_per_atom_P2_error=max(rows[i]["exact_check_max_abs_P2_error"] for i in checked),
                  sampled_visibility_definition="(max-min)/(max+min) of the ensemble mean on the sampled chirp grid.",
                  readout="Mean traces use expected normalized fluorescence. Example scans and 2.5/97.5 percentiles "
                          "draw atoms/photons/electronics in both arms of the comparison; percentiles are pointwise single gates.",
                  clock="Fixed third-pulse phase, chirp referenced to each trajectory's MOT exit. No laboratory-time "
                        "average over continuous asynchronous arrivals. Technical perturbations are frozen per gate.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    _write_table(output/"chirp_noise_fringes.csv", rows)
    (output/"chirp_noise_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
                                                  encoding="utf-8")
    if plot:
        from .chirp_noise_plot import plot_chirp_comparison
        plot_chirp_comparison(rows, output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(DEFAULT_CONFIG))
    parser.add_argument("--noise-config", type=Path, default=Path(DEFAULT_NOISE))
    parser.add_argument("--source-state-h5", type=Path, default=Path(DEFAULT_SOURCE))
    parser.add_argument("--output", type=Path, default=Path("beam_outputs/chirp_noise"))
    parser.add_argument("--records", type=int, help="Independent noise records per chirp")
    parser.add_argument("--points", type=int, help="Number of scanned chirps")
    parser.add_argument("--workers", type=int, default=4, help="Parallel chirp points (default 4)")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    def read(path):
        return json.loads(path.read_text(encoding="utf-8-sig"))
    config = SimulationConfig.from_dict(read(args.config))
    settings = NoiseSettings(**read(args.noise_config))
    if args.records is not None:
        settings = replace(settings, records=args.records)
    if args.points is not None:
        config.chirp_scan.points = args.points
    result = run(config, read_stateful_ensemble(args.source_state_h5), settings, args.output, workers=args.workers,
                 plot=not args.no_plot, inputs={k: str(getattr(args, k)) for k in ("config", "source_state_h5", "noise_config")})
    for case in ("baseline", "combined"):
        print(case, "sampled visibility:", result[case]["visibility"])
    print("Saved", args.output.resolve())


if __name__ == "__main__":
    main()
