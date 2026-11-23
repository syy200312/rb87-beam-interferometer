"""Reproducible low-power pump comparator; run python -m rb87_beam.low_power_study."""
import argparse
import csv
import hashlib
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Limit BLAS threads in each Windows worker before importing NumPy.
for _var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np

from .batch import subset_ensemble
from .io import read_stateful_ensemble
from .joint_optimization import assert_independent_sources, load_config, validation_subset
from .low_power_evaluation import pump_config, screen_record, native_records
from .low_power_matched import select_matched, write_matched
from .low_power_report import paired_metrics, certify_family, write_report

POWERS = (40, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110, 120, 140, 160)
LENGTHS = (4, 8, 16, 25)
DETUNINGS = (-3, -1.5, 0, 1.5, 3)
VAL_POWERS = (60, 70, 80, 90, 100)
VAL_LENGTHS = (4, 8)
VAL_DETUNINGS = (-3, -1.5, 1.5)
ANCHOR = ("3:1", 80, 4, 1.5)
KEYS = ("family", "power_uW", "length_mm", "detuning_MHz")
TRAIN_SEED, VALIDATION_SEED = 71931, 61087
_BASE = _TRAIN = None


def coordinates(point):
    return dict(zip(KEYS, point))


def point_of(row):
    return tuple(row[k] for k in KEYS)


def config_at(base, point):
    family, power, length, delta = point
    return pump_config(base, power, length, delta, {"1:1": .5, "3:1": .75}[family])


def _init(config_path, train_path):
    global _BASE, _TRAIN
    _BASE = load_config(config_path)
    _TRAIN = read_stateful_ensemble(train_path)
    # Fixed 16 incident IDs out of 32; the archived source has no failed launches.
    _TRAIN = subset_ensemble(_TRAIN, np.arange(0, len(_TRAIN.r_m), 2))


def screen_one(point):
    record = screen_record(config_at(_BASE, point), _TRAIN)
    w, target = record["detection_weights"], record["target_population"]
    phi = _TRAIN.flux_per_s * np.mean(w * target)
    return dict(coordinates(point), flux_per_s=float(phi),
                photons=float(np.mean(record["scattered_photons"])), efficiency=float(np.mean(target)))


def write_json(path, obj):
    temp = path.with_suffix(".part")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _tag(point):
    return ("f%s_P%g_L%g_D%g" % (point[0].replace(":", ""), *point[1:])).replace(".", "p")


def _rank(row):
    return row["photons"], -row["flux_per_s"], row["length_mm"], row["power_uW"], row["detuning_MHz"]


def shortlist(rows, floor):
    """Independently retain the best feasible screen for each split and each detuning.

    Also retain the next feasible power above each of these finalists; the
    actual native training flux rather than the ballistic estimate decides.
    """
    points = {ANCHOR}
    for family, delta in itertools.product(("1:1", "3:1"), DETUNINGS):
        feasible = [r for r in rows if r["family"] == family and r["detuning_MHz"] == delta
                    and r["flux_per_s"] >= floor]
        if feasible:
            best = min(feasible, key=_rank)
            points.add(point_of(best))
            up = [r for r in feasible if r["length_mm"] == best["length_mm"]
                  and r["power_uW"] > best["power_uW"]]
            if up:
                points.add(point_of(min(up, key=lambda r: r["power_uW"])))
    return points


def _one_native(point, base, source, path, rep, workers, pool, seed):
    return native_records(config_at(base, point), source, path / _tag(point), rep, workers,
                          seed, executor=pool)


def _prefix(records, n):
    return [{k: v[r["atom_ids"] < n] for k, v in r.items()} for r in records]


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    cfg, tr, va = map(Path, (args.config, args.training_source, args.validation_source))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    base = load_config(cfg)
    training, full = read_stateful_ensemble(tr), read_stateful_ensemble(va)
    if training.source_metadata.get("sampled") != 32 or len(training.r_m) != 32:
        raise ValueError("Screening expects the archived complete 32-launch training pool")
    assert_independent_sources(training, full)
    flux = float(base.source.incident_flux_per_s)
    if not np.isclose(training.flux_per_s, flux) or full.source_metadata["incident_flux_per_s"] != flux:
        raise ValueError("Training/validation incident fluxes differ from config")
    if args.workers < 1 or args.replicas < 1 or args.atoms < 2 or not 0 < args.floor <= flux:
        raise ValueError("Invalid sample size, worker count, replicas or floor")
    val = validation_subset(full, args.atoms)
    grid = list(itertools.product(("1:1", "3:1"), POWERS, LENGTHS, DETUNINGS))
    # No validation result participates in ranking, matching or projection of a region.
    protocol = dict(version=2, config_sha256=_hash(cfg), training_sha256=_hash(tr), validation_sha256=_hash(va),
                    files=dict(config=str(cfg), training=str(tr), validation=str(va)),
                    powers_uW=POWERS, lengths_mm=LENGTHS, detunings_MHz=DETUNINGS,
                    validation_powers_uW=VAL_POWERS, validation_lengths_mm=VAL_LENGTHS,
                    validation_detunings_MHz=VAL_DETUNINGS, anchor=coordinates(ANCHOR),
                    validation_extra="held-out training winner of 3:1; 16/25 mm cuts at 80 uW,+1.5 MHz",
                    floor=args.floor, incident_flux_per_s=flux, n_train_screen=16, n_train_native=32,
                    train_replicas=2, train_feasible="point flux >= floor AND marginal bootstrap lower 95% >= floor",
                    selection="minimum mean emitted photons, then higher flux, shorter length, lower power, lower detuning",
                    shortlist="best above the floor per family/detuning and next higher feasible power at same length; plus anchor and a matched comparator",
                    training_recoil_seed=TRAIN_SEED, validation_recoil_seed=VALIDATION_SEED,
                    n_validation=args.atoms, validation_replicas=args.replicas,
                    bootstrap=2500)
    pfile = out / "protocol.json"
    if pfile.exists() and json.loads(pfile.read_text(encoding="utf-8")) != json.loads(json.dumps(protocol)):
        raise ValueError("Output protocol differs; choose a new output directory")
    write_json(pfile, protocol)
    train_native = out / ("native_training_seed_%d" % TRAIN_SEED)
    val_native = out / "native_validation"
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(str(cfg), str(tr))) as pool:
        sc = out / "screen.json"
        if sc.exists():
            rows = json.loads(sc.read_text(encoding="utf-8"))
        else:
            print("Screening %d low-power settings (both splits)" % len(grid), flush=True)
            rows = []
            for row in pool.map(screen_one, grid, chunksize=2):
                rows.append(row)
                if len(rows) % 80 == 0:
                    print("Screened %d/%d" % (len(rows), len(grid)), flush=True)
            write_json(sc, rows)
        if set(map(point_of, rows)) != set(grid) or len(rows) != len(grid):
            raise ValueError("Incomplete screening grid")
        write_csv(out / "training_screen.csv", rows)
        anchor = next(r for r in rows if point_of(r) == ANCHOR)
        match = select_matched(rows, anchor, args.floor)
        points = shortlist(rows, args.floor)
        if match is not None:
            points.add(point_of(match))
        train_rows = []
        for i, point in enumerate(sorted(points)):
            rec = _one_native(point, base, training, train_native, 2, args.workers, pool, TRAIN_SEED)
            m = paired_metrics(rec, rec, 32, flux, args.floor, bootstrap=1200)
            train_rows.append(dict(coordinates(point), flux_per_s=m["baseline_flux_per_s"],
                                   flux_low95=m["baseline_flux_per_s_lo95"], photons=m["baseline_photons"],
                                   efficiency=m["baseline_efficiency"], contrast=m["baseline_contrast"]))
            print("Native training %d/%d" % (i+1, len(points)), flush=True)
        write_csv(out / "training_native.csv", train_rows)
        winners = {}
        for family in ("1:1", "3:1"):
            eligible = [r for r in train_rows if r["family"] == family and
                        r["flux_per_s"] >= args.floor and r["flux_low95"] >= args.floor]
            if not eligible:
                raise RuntimeError("No certified training baseline for " + family + "; inspect training_native.csv")
            winners[family] = min(eligible, key=_rank)
        baseline = winners["1:1"]
        validate = set(itertools.product(("3:1",), VAL_POWERS, VAL_LENGTHS, VAL_DETUNINGS))
        validate.update([ANCHOR, point_of(winners["3:1"]), ("3:1",80,16,1.5), ("3:1",80,25,1.5)])
        frozen = dict(winners=winners, matched_via_ballistic_training=match, validation=list(map(coordinates, sorted(validate))))
        write_json(out / "selection.json", frozen)
        print("Frozen conventional:", point_of(baseline), "; %d independent validation nodes" % len(validate), flush=True)
        ref = _one_native(point_of(baseline), base, val, val_native, args.replicas, args.workers, pool, VALIDATION_SEED)
        valid_rows, convergence = [], []
        for i, point in enumerate(sorted(validate)):
            rec = _one_native(point, base, val, val_native, args.replicas, args.workers, pool, VALIDATION_SEED)
            m = paired_metrics(ref, rec, args.atoms, flux, args.floor, bootstrap=2500, retain_draws=True)
            valid_rows.append(dict(coordinates(point), **m))
            if point == ANCHOR:
                anchor_rec = rec
            if point in {ANCHOR, point_of(winners["3:1"])}:
                for n in sorted({v for v in (32, 64, args.atoms) if v <= args.atoms}):
                    vals = paired_metrics(_prefix(ref, n), _prefix(rec, n), n, flux, args.floor, bootstrap=2500)
                    convergence.append(dict(coordinates(point), **vals))
            print("Native validation %d/%d" % (i+1, len(validate)), flush=True)
        m = None
        if match is not None:
            mr = (_one_native(point_of(match), base, val, val_native, args.replicas, args.workers, pool, VALIDATION_SEED)
                  if point_of(match) != point_of(baseline) else ref)
            m = paired_metrics(mr, anchor_rec, args.atoms, flux, args.floor, bootstrap=2500)
        write_matched(out, match, anchor, m)
        write_csv(out / "convergence.csv", convergence)
    critical = certify_family(valid_rows, args.floor)
    metadata = dict(baseline=baseline, incident_samples=args.atoms, replicas=args.replicas,
                    min_flux_per_s=args.floor, incident_flux_per_s=flux,
                    selection_notes="两组都在同一个总功率、作用全长和共同失谐网格上筛选，共 %d 个设置。"
                    "随后在 32 个训练原子上按完整运动与反冲复核 %d 个设置，"
                    "按通量下界合格后每个出射原子的平均散射数最少，冻结 1:1 基线。"
                    "此处比较仅针对标称束速分布。另见[历史工作点的近等通量对照](matched_flux.md)。" %
                    (len(rows),len(points)))
    power_axis = sorted(set(VAL_POWERS) | {point[1] for point in validate})
    write_report(out, valid_rows, power_axis, metadata)
    write_json(out / "summary.json", dict(metadata=metadata, family_critical=critical,
               common_flux=sum(r["common_flux_supported"] for r in valid_rows), nodes=len(valid_rows),
               lower_scattering=sum(r["advantage"] for r in valid_rows),
               higher_scattering=sum(r["baseline_advantage"] for r in valid_rows),
               near_flux_lower_scattering=sum(r["advantage"] and r["matched_flux_1pct"] for r in valid_rows)))
    print("Report:", out / "report.md", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="beam_configs/optical_pumping_optimized.json")
    p.add_argument("--training-source", default="beam_outputs/pylcp_multilevel_32/source_ensemble.h5")
    p.add_argument("--validation-source", default="beam_outputs/pumping_optimization/source/source_ensemble.h5")
    p.add_argument("--output", default="beam_outputs/low_power_baseline")
    p.add_argument("--atoms", type=int, default=128)
    p.add_argument("--replicas", type=int, default=2)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--floor", type=float, default=9.5e9)
    run(p.parse_args())


if __name__ == "__main__":
    main()
