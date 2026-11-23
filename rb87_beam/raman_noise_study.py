"""Paired technical-noise budget for the spatial three-pulse Raman gyro.

Run from the repository root: python -m rb87_beam.raman_noise_study --help
"""

import argparse
import copy
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from .config import SimulationConfig
from .io import read_stateful_ensemble
from .physics import flight_time
from .raman_noise import make_kernel
from .raman_noise_metrics import make_detector, summarize_noise


DEFAULT_CONFIG = "beam_outputs/raman_power_calibration/config_power_driven.json"
DEFAULT_SOURCE = "beam_outputs/endpoint_comparison/native_validation/baseline/nominal/prepared_0.h5"
DEFAULT_NOISE = "beam_configs/raman_noise_budget.json"
SCENARIOS = ("detection", "phase", "vibration", "magnetic", "intensity", "combined")


@dataclass(frozen=True)
class NoiseSettings:
    phase_rms_rad: float
    phase_zone_correlation: float
    mirror_displacement_rms_m: float
    mirror_zone_correlation: float
    bias_field_G: float
    field_rms_G: float
    field_zone_correlation: float
    fractional_power_rms: float
    log_power_zone_correlation: float
    gate_s: float
    record_period_s: float
    operating_phase_rad: float
    rotation_step_rad_s: float
    records: int
    scans: int
    phase_points: int
    seed: int

    def validate(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(name + " must be a finite number")
            if name.endswith("correlation") and not -1 <= value <= 1:
                raise ValueError(name + " must be between -1 and 1")
        for name in ("phase_rms_rad", "mirror_displacement_rms_m", "field_rms_G", "fractional_power_rms"):
            if getattr(self, name) < 0:
                raise ValueError(name + " must be nonnegative")
        for name, minimum in (("records", 4), ("scans", 2), ("phase_points", 4), ("seed", 0)):
            value = getattr(self, name)
            if not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not (self.record_period_s >= self.gate_s > 0 and self.rotation_step_rad_s > 0):
            raise ValueError("Period must cover a positive gate and rotation_step_rad_s must be positive")


def sample_disturbances(settings):
    """Independent records; covariance between zones i,j is rms² rho^abs(i-j)."""
    settings.validate()
    def zones(rho, stream):
        rng = np.random.default_rng(np.random.SeedSequence(settings.seed, spawn_key=(stream,)))
        z = rng.standard_normal((settings.records, 3))
        for j in (1, 2):
            z[:, j] = rho * z[:, j-1] + np.sqrt(1 - rho**2) * z[:, j]
        return z

    log_sigma = np.sqrt(np.log1p(settings.fractional_power_rms**2))
    return {
        "phase": settings.phase_rms_rad * zones(settings.phase_zone_correlation, 0),
        "vibration": settings.mirror_displacement_rms_m * zones(settings.mirror_zone_correlation, 1),
        "magnetic": settings.field_rms_G * zones(settings.field_zone_correlation, 2),
        "intensity": np.exp(log_sigma * zones(settings.log_power_zone_correlation, 3) - log_sigma**2/2),
    }


def _write_table(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(out, summaries, fringes):
    from matplotlib import pyplot as plt
    labels = ["Detector", "+ Phase", "+ Mirror", "+ Field", "+ Power", "Combined"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    y = np.arange(len(labels))
    for ax, key, err, scale, label in (
        (axes[0], "contrast", "contrast_mc_se", 100, "Mean fringe contrast (%)"),
        (axes[1], "allan_1s_rad_s", "allan_1s_mc_se_rad_s", 1, "Allan deviation at 1 s (rad/s)"),
    ):
        valid = [i for i, row in enumerate(summaries) if row[key] is not None]
        ax.errorbar([summaries[i][key]*scale for i in valid], y[valid],
                    xerr=[(summaries[i][err] or 0)*scale for i in valid], fmt="o", color="#147063", capsize=3)
        ax.set(yticks=y, yticklabels=labels, xlabel=label)
        ax.set_ylim(len(y)-0.5, -0.5)
        ax.grid(axis="x", alpha=.25)
    if any(row["allan_1s_rad_s"] is not None for row in summaries):
        axes[1].set_xscale("log")
    fig.savefig(out/"noise_budget.png", dpi=180)
    plt.close(fig)
    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(7.5, 6), constrained_layout=True)
    base = np.asarray(fringes[0]["mean_expected_P2"])
    for data, label in zip(fringes, labels):
        x = np.asarray(data["phase_rad"])
        p = np.asarray(data["mean_expected_P2"])
        if label in ("Detector", "Combined"):
            top.plot(x, p, label=label)
        if label != "Detector":
            bottom.plot(x, 1e3*(p-base), label=label)
    top.set_ylabel("Expected fluorescence fraction")
    top.legend()
    bottom.set(xlabel="Third-pulse phase (rad)", ylabel="Change from detector only (×10⁻³)")
    bottom.legend(ncol=3, fontsize=8)
    fig.savefig(out/"noise_fringes.png", dpi=180)
    plt.close(fig)


def run(config, ensemble, settings, out, plot=True, progress=print, inputs=None):
    settings.validate()
    config = copy.deepcopy(config)
    config.pumping.enabled = False  # input is a prepared checkpoint
    config.detection.gate_s = settings.gate_s
    config.detection.noise = True
    config.validate()
    ensemble.validate()
    r = config.raman
    halfwidth = r.envelope_half_widths*r.waist_z_m if r.envelope == "gaussian" else r.rectangular_width_m/2
    first_edge = r.first_zone_z_m - halfwidth
    if ensemble.reference_plane_z_m >= first_edge:
        raise ValueError("The prepared checkpoint must precede the first Raman envelope")
    # A record freezes each zone's local optics throughout the accepted atoms' flight.
    # Leave enough guard time between detector gates for different draws to be independent.
    transit_s = float(np.max(flight_time(ensemble, config.detection.z_m) - flight_time(ensemble, first_edge)))
    if settings.record_period_s + 1e-12 < settings.gate_s + transit_s:
        raise ValueError(f"Independent frozen records require record_period_s >= gate_s + {transit_s:.8g} s")
    draws = sample_disturbances(settings)
    phases = np.linspace(0, 2*np.pi, settings.phase_points, endpoint=False)
    detector = make_detector(ensemble, config)
    step = settings.rotation_step_rad_s
    center, plus, minus = [make_kernel(ensemble, r, rotation_delta_y=d) for d in (0., step, -step)]
    def shifted_detector(delta):
        shifted = copy.deepcopy(config)
        rotation = list(r.rotation_rad_s)
        rotation[1] += delta
        shifted.raman.rotation_rad_s = tuple(rotation)
        return make_detector(ensemble, shifted)
    dplus, dminus = shifted_detector(step), shifted_detector(-step)
    zero, one = np.zeros(3), np.ones(3)
    nominal = (zero, zero, zero, one, settings.bias_field_G)
    def slope_at(delta):
        values = []
        for sign in (1, -1):
            k = make_kernel(ensemble, r, rotation_delta_y=sign*delta)
            h = k.harmonics(*nominal)[None, :, :]
            values.append(float(np.ravel(shifted_detector(sign*delta).readout(h, settings.operating_phase_rad))[0]))
        return (values[0]-values[1])/(2*delta)
    halfstep_slope = slope_at(step/2)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    summaries, fringes = [], []
    for case in SCENARIOS:
        if progress:
            progress("Propagating " + case)
        active = set(draws) if case == "combined" else {case}
        args = [draws[name] if name in active else np.broadcast_to(default, (settings.records, 3))
                for name, default in zip(("phase", "vibration", "magnetic", "intensity"), (zero, zero, zero, one))]
        # For the detector-only row every optical realization is identical.
        count = 1 if case == "detection" else settings.records
        h = np.empty((count, 3, len(ensemble.r_m)))
        slopes = np.empty(count)
        exact_error = 0.
        check_ids = {0, count//2, count-1}
        for values, reference in ((args[2], 0.), (args[3], 1.)):
            deviation = np.max(abs(values[:count] - reference), axis=1)
            check_ids.add(int(np.argmax(deviation)))
        for i in range(count):
            perturb = tuple(array[i] for array in args) + (settings.bias_field_G,)
            h[i] = center.harmonics(*perturb)
            hp, hm = plus.harmonics(*perturb), minus.harmonics(*perturb)
            yp = dplus.readout(hp[None], settings.operating_phase_rad)
            ym = dminus.readout(hm[None], settings.operating_phase_rad)
            slopes[i] = (np.ravel(yp)[0]-np.ravel(ym)[0])/(2*step)
            if i in check_ids:
                exact_error = max(exact_error, float(np.max(abs(h[i] - center.harmonics(*perturb, exact=True)))))
        if case == "detection":
            h = np.broadcast_to(h, (settings.records, *h.shape[1:]))
            slopes = np.broadcast_to(slopes, (settings.records,))
        # Derivatives smaller than floating-point resolution at this step do not
        # provide a calibrated angular-rate estimate at the selected phase.
        slope_input = 0. if abs(np.mean(slopes)) < 8*np.finfo(float).eps/step else slopes
        metrics, curves = summarize_noise(h, detector, settings.operating_phase_rad, slope_input,
            phases=phases, scans=settings.scans, seed=settings.seed, cycle_s=settings.record_period_s)
        summaries.append(dict(scenario=case, **metrics, checked_max_abs_coeff_error=exact_error))
        fringes.append(curves)
    baseline = summaries[0]
    for row in summaries:
        row["allan_relative_to_detection"] = (row["allan_1s_rad_s"]/baseline["allan_1s_rad_s"]
            if row["allan_1s_rad_s"] is not None and baseline["allan_1s_rad_s"] else None)
        row["contrast_change_percentage_points"] = (100*(row["contrast"]-baseline["contrast"])
            if row["contrast"] is not None and baseline["contrast"] is not None else None)
    report = dict(settings=asdict(settings), inputs=inputs, simulation_config=config.to_dict(),
        prepared_trajectories=len(ensemble.r_m),
        source_flux_per_s=ensemble.flux_per_s, max_first_pulse_to_detector_time_s=transit_s,
        nominal_rotation_y_rad_s=r.rotation_rad_s[1], nominal_slope_half_step=halfstep_slope,
        nominal_slope_step_relative_difference=(abs(baseline["dP2_dOmega"]-halfstep_slope)/abs(halfstep_slope)
            if abs(halfstep_slope) > 8*np.finfo(float).eps/step else None),
        simulations=summaries, convention="Single forward beam. Quasistatic spatial perturbations per independent gate; "
        "local fluorescence rotation slope, conditional white Allan deviation at 1 s, including record dead time.")
    (out/"noise_budget.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    _write_table(out/"noise_budget.csv", summaries)
    rows = [dict(scenario=case, **dict(zip(curves, values))) for case, curves in zip(SCENARIOS, fringes)
            for values in zip(*(curves[key] for key in curves))]
    _write_table(out/"noise_fringes.csv", rows)
    if plot:
        _plot(out, summaries, fringes)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(DEFAULT_CONFIG), help="Existing beam/Raman simulation JSON")
    parser.add_argument("--source-state-h5", type=Path, default=Path(DEFAULT_SOURCE), help="Prepared ensemble checkpoint")
    parser.add_argument("--noise-config", type=Path, default=Path(DEFAULT_NOISE), help="Noise amplitudes, correlations and acquisition settings")
    for key in ("records", "scans", "seed"):
        parser.add_argument("--"+key, type=int)
    parser.add_argument("--output", type=Path, default=Path("beam_outputs/raman_noise"))
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    def read(path):
        return json.loads(path.read_text(encoding="utf-8-sig"))
    settings = read(args.noise_config)
    settings.update({key: getattr(args, key) for key in ("records", "scans", "seed") if getattr(args, key) is not None})
    report = run(SimulationConfig.from_dict(read(args.config)), read_stateful_ensemble(args.source_state_h5),
                 NoiseSettings(**settings), args.output, plot=not args.no_plot,
                 inputs={"simulation_config": args.config.as_posix(), "prepared_checkpoint": args.source_state_h5.as_posix(),
                         "noise_config": args.noise_config.as_posix()})
    for row in report["simulations"]:
        contrast = "undefined" if row["contrast"] is None else f"{100*row['contrast']:.5f}%"
        allan = "undefined" if row["allan_1s_rad_s"] is None else f"{row['allan_1s_rad_s']:.5g} rad/s"
        print(f"{row['scenario']:10}  C={contrast}   Allan(1 s)={allan}")
    print("Saved", args.output.resolve())


if __name__ == "__main__":
    main()
