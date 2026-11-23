"""Interactive atom flux vs three-pulse fringe study (run as a Python module)."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .config import SimulationConfig
from .io import read_stateful_ensemble
from .physics import (
    Ensemble,
    fit_fringe,
    flight_time,
    preparation_weights,
    reference_y,
    run_simulation,
    trajectory,
)
from .states import G_CLOCK


ARCHIVE = "beam_outputs/low_power_baseline/native_validation"
OUTPUT = "beam_outputs/flux_visibility"
RECIPES = (
    ("f11_P80_L16_D0", "1:1 · 40/40 µW · 16 mm · 0 MHz"),
    ("f31_P80_L4_D1p5", "3:1 · 60/20 µW · 4 mm · +1.5 MHz"),
)
DEFAULT_FLUX = (1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7, 3e7, 1e8, 3e8, 1e9, 3e9, 1e10)
RADII_MM = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 6.0)


def dump_json(path, obj):
    def serial(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError("Cannot serialize " + type(value).__name__)

    path.write_text(
        json.dumps(obj, default=serial, ensure_ascii=False, allow_nan=False, indent=2),
        encoding="utf-8",
    )


def write_csv(path, data):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_prepared(directory):
    """Two equal-weight realizations of the same source IDs, with ONE flux normalization."""
    config_file = directory / "config_used.json"
    cfg = SimulationConfig.from_dict(json.loads(config_file.read_text(encoding="utf-8")))
    paths = [directory / ("prepared_%d.h5" % i) for i in (0, 1)]
    arrays = [read_stateful_ensemble(p) for p in paths]
    base = arrays[0]
    meta = base.source_metadata
    sample_ids = meta["sample_ids"]
    n = meta["sampled"]
    extract = len(sample_ids) / n
    if (
        not 0 < extract <= 1
        or len(set(sample_ids)) != len(sample_ids)
        or min(sample_ids) < 0
        or max(sample_ids) >= n
        or "source_signature" not in meta
    ):
        raise ValueError("Bad incident source IDs in prepared checkpoint")
    for rep, e in enumerate(arrays):
        m = e.source_metadata
        if (
            m["sample_ids"] != sample_ids
            or m["source_signature"] != meta["source_signature"]
            or m["sampled"] != n
            or m["optical_pumping"]["replica"] != rep
            or not np.isclose(e.flux_per_s, extract * m["incident_flux_per_s"])
            or e.reference_plane_z_m != base.reference_plane_z_m
        ):
            raise ValueError(
                "Recoil replicas must describe the same incident source and pump plane"
            )
    if cfg.raman.state_handling != "inherit" or cfg.raman.selection_sigma_vx_m_s != 0:
        raise ValueError(
            "This archived comparison requires inherited states and no velocity preselection"
        )

    def cat(name):
        return np.concatenate([getattr(e, name) for e in arrays], axis=0)

    # The repeated rows are quadrature samples for recoil, not additional source flux.
    beam = Ensemble(
        cat("r_m"),
        cat("v_m_s"),
        base.flux_per_s,
        {"mode": "pooled_prepared_replicas"},
        ground_density=cat("ground_density"),
        reference_time_s=cat("reference_time_s"),
        reference_plane_z_m=base.reference_plane_z_m,
    )
    provenance = dict(
        incident_samples=n,
        extracted_per_replica=len(sample_ids),
        recoil_replicas=len(paths),
        source_signature=meta["source_signature"],
        source_ids=sample_ids,
        extraction_fraction=extract,
        nominal_incident_flux=float(meta["incident_flux_per_s"]),
        files={str(p): _sha(p) for p in [config_file] + paths},
    )
    return cfg, beam, provenance, np.tile(sample_ids, len(paths))


def aperture_rows(result, source_ids, frac, nominal_flux):
    """Same individual Raman responses; only the detection radius changes."""
    beam = result["ensemble"]
    cfg = result["config"]
    r, _ = trajectory(beam, flight_time(beam, cfg.detection.z_m), cfg.raman)
    rho_mm = 1e3 * np.hypot(r[:, 0], r[:, 1] - reference_y(cfg.detection.z_m, cfg.raman))
    clock = beam.ground_density[:, G_CLOCK, G_CLOCK].real
    rows = []
    for radius in RADII_MM:
        w = preparation_weights(beam, cfg.raman) * (rho_mm < radius)
        if not np.any(w):
            continue
        y = np.average(result["per_atom_P2"], axis=1, weights=w / beam.v_m_s[:, 2])
        rows.append(
            dict(
                radius_mm=radius,
                incident_flux_per_s=nominal_flux,
                detected_flux_per_s=float(nominal_flux * frac * np.mean(w)),
                effective_target_flux_per_s=float(nominal_flux * frac * np.mean(clock * w)),
                ideal_contrast=float(fit_fringe(result["phase_rad"], y)["contrast"]),
                accepted_source_ids=len(np.unique(source_ids[w > 0])),
            )
        )
    return rows


def run(args):
    from .flux_visibility import make_visibility_snapshot, scan_flux_visibility

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rates = sorted(map(float, args.fluxes))
    if (
        not rates
        or rates[0] <= 0
        or not np.isfinite(rates).all()
        or len(set(rates)) != len(rates)
        or not np.isfinite(args.gate_ms)
        or args.gate_ms <= 0
        or args.phase_points < 5
        or args.repeats < 2
        or args.seed < 0
    ):
        raise ValueError("Invalid flux, detection gate, repeats, seed or phase-point count")
    series = []
    provenance = []
    scans = []
    metrics = []
    traces = []
    apertures = []
    for ix, (folder, title) in enumerate(RECIPES):
        cfg, beam, prov, ids = load_prepared(Path(args.archive) / folder)
        if ix and any(
            prov[k] != provenance[0][k]
            for k in ("source_signature", "source_ids", "nominal_incident_flux")
        ):
            raise ValueError("The two recipes do not use the same MOT source")
        cfg.raman.scan_points = args.phase_points
        cfg.detection.gate_s = args.gate_ms / 1000
        # No additional pump traversal. Counting noise is added to complete scans below.
        cfg.pumping.enabled = False
        cfg.detection.noise = False
        cfg.validate()
        result = run_simulation(cfg, beam)
        print("Computed three Raman pulses:", folder, flush=True)
        det = result["detection"]
        snapshot = make_visibility_snapshot(
            result["phase_rad"],
            result["per_atom_P2"],
            det["detection_weights"],
            beam.v_m_s[:, 2],
            cfg.detection,
            extracted_fraction=prov["extraction_fraction"],
            target_population=beam.ground_density[:, G_CLOCK, G_CLOCK].real,
        )
        record = scan_flux_visibility(snapshot, rates, repeats=args.repeats, seed=args.seed + ix)
        if abs(record["ideal_contrast"] - result["expected_fit"]["contrast"]) > 1e-10:
            raise ArithmeticError(
                "Reweighted visibility differs from the native deterministic detector"
            )
        group = aperture_rows(
            result, ids, prov["extraction_fraction"], prov["nominal_incident_flux"]
        )
        provenance.append(dict(prov, recipe=folder))
        scans.append(dict(record, recipe=folder))
        rows = []
        for prof in record["profiles"]:
            row = dict(
                flux_per_s=prof["incident_flux_per_s"],
                ideal_contrast=record["ideal_contrast"],
                median_contrast=prof["fitted_median"],
                lo95=prof["fitted_q025"],
                hi95=prof["fitted_q975"],
                example_signal=prof["example_P2"],
                example_fit=prof["example_fit_P2"],
                **{
                    k: prof[k]
                    for k in (
                        "example_contrast",
                        "detected_flux_per_s",
                        "effective_target_flux_per_s",
                        "expected_atoms_per_gate",
                        "fit_success_rate",
                        "unphysical_fit_fraction",
                    )
                }
            )
            rows.append(row)
            metrics.append(
                dict(recipe=folder, **{k: v for k, v in row.items() if not k.startswith("example")})
            )
            for j, phi in enumerate(record["phase_rad"]):
                traces.append(
                    dict(
                        recipe=folder,
                        incident_flux_per_s=row["flux_per_s"],
                        phase_rad=phi,
                        ideal_P2=record["ideal_P2"][j],
                        observed_P2=row["example_signal"][j],
                        fitted_P2=row["example_fit"][j],
                        F1_raw_counts=prof["example_F1_counts"][j],
                        F2_raw_counts=prof["example_F2_counts"][j],
                    )
                )
        series.append(dict(name=title, ideal_signal=record["ideal_P2"], rows=rows, apertures=group))
        apertures.extend(dict(recipe=folder, **r) for r in group)
    meta = dict(
        gate_ms=args.gate_ms,
        phase_points=args.phase_points,
        repeats=args.repeats,
        seed=args.seed,
        source_samples=prov["incident_samples"],
        recoil_replicas=prov["recoil_replicas"],
        background_counts=cfg.detection.background_counts,
        electronic_noise_std_counts=cfg.detection.electronic_noise_std_counts,
        source_note="24态泵浦含反冲的制备出口；非独立 OBE 输出",
        nominal_incident_flux_per_s=prov["nominal_incident_flux"],
        flux_definition="captured-reservoir incident atom flux (atoms/s); fixed source phase-space and populations",
    )
    payload = dict(phase_rad=record["phase_rad"], recipes=series, metadata=meta)
    dump_json(out / "scan.json", dict(metadata=meta, preparation=provenance, recipes=scans))
    write_csv(out / "contrast.csv", metrics)
    write_csv(out / "example_fringes.csv", traces)
    write_csv(out / "apertures.csv", apertures)
    from .flux_visibility_plot import write_flux_plots

    write_flux_plots(out, payload)
    print(out / "flux_visibility.html", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", default=ARCHIVE)
    p.add_argument("--output", default=OUTPUT)
    p.add_argument("--fluxes", type=float, nargs="+", default=DEFAULT_FLUX, metavar="ATOMS/S")
    p.add_argument(
        "--gate-ms", type=float, default=10.0, help="Integration time for each phase point"
    )
    p.add_argument("--phase-points", type=int, default=41)
    p.add_argument("--repeats", type=int, default=600)
    p.add_argument("--seed", type=int, default=11087)
    run(p.parse_args())


if __name__ == "__main__":
    main()
