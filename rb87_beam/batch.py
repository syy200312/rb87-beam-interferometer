"""Resumable native PyLCP batches, with per-atom seeds independent of worker count.

These are fresh source trajectories, never resampled copies of an old checkpoint.
Only final source states are cached; the normal source solver remains unchanged.
"""

import copy
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import SimulationConfig
from .io import save_stateful_ensemble
from .physics import Ensemble, KB, MASS, disk_positions

_SOURCE_MODEL = None
_SOURCE_SEED = None


def atom_rng(seed, stage, atom_id):
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(stage, int(atom_id))))


def subset_ensemble(ensemble, indices, flux_per_s=None):
    indices = np.asarray(indices, dtype=int)
    extras = {key: getattr(ensemble, key)[indices] for key in (
        "ground_density", "exit_populations", "exit_time_s", "quantization_axes", "reference_time_s")
        if getattr(ensemble, key) is not None}
    metadata = copy.deepcopy(ensemble.source_metadata)
    if "sample_ids" in metadata:
        metadata["sample_ids"] = np.asarray(metadata["sample_ids"])[indices].tolist()
    result = Ensemble(ensemble.r_m[indices].copy(), ensemble.v_m_s[indices].copy(),
        ensemble.flux_per_s if flux_per_s is None else flux_per_s, metadata,
        reference_plane_z_m=ensemble.reference_plane_z_m, **extras)
    result.validate()
    return result


def source_prefix(ensemble, incident_samples):
    """Nested prefixes of independently launched atoms, including failed launches."""
    ids = np.asarray(ensemble.source_metadata["sample_ids"])
    selected = np.flatnonzero(ids < incident_samples)
    flux = ensemble.source_metadata["incident_flux_per_s"] * len(selected) / incident_samples
    result = subset_ensemble(ensemble, selected, flux)
    result.source_metadata.update(sampled=int(incident_samples), extracted=len(selected),
        extraction_fraction=len(selected) / incident_samples)
    return result


def _source_init(config_data, seed):
    from .pylcp_mot import MOT2DPlus
    global _SOURCE_MODEL, _SOURCE_SEED
    c = SimulationConfig.from_dict(config_data)
    _SOURCE_MODEL, _SOURCE_SEED = MOT2DPlus(c.source, c.pylcp), seed


def _source_atom(atom_id):
    model = _SOURCE_MODEL
    s = model.source
    rng = atom_rng(_SOURCE_SEED, 10, atom_id)
    r = disk_positions(rng, 1, s.initial_radius_m)[0]
    r[2] = model.options.initial_z_m
    v = rng.normal(0, np.sqrt(KB * s.initial_temperature_K / MASS), 3)
    v[2] = rng.normal(s.initial_vz_m_s, s.initial_sigma_vz_m_s)
    atom = model.evolve_atom(r, v, model.initial_populations(), rng)
    rho, axis = model.ground_after_decay(atom["populations"], atom["r_m"])
    return dict(atom_id=int(atom_id), r_m=atom["r_m"], v_m_s=atom["v_m_s"],
        populations=atom["populations"], time_s=atom["time_s"], status=atom["status"],
        ground_density=rho, axis=axis, initial_r_m=r, initial_v_m_s=v)


def generate_source_batch(config, count, directory, workers=8, seed=20087, progress=print):
    config.validate()
    if config.source.mode != "pylcp" or count < 1 or workers < 1:
        raise ValueError("Native source batch needs pylcp, positive count and worker count")
    directory = Path(directory)
    cache = directory / "source_atoms"
    cache.mkdir(parents=True, exist_ok=True)
    source_options = asdict(config.source)
    source_options.pop("n_atoms")
    signature = hashlib.sha256(json.dumps(dict(source=source_options, pylcp=asdict(config.pylcp),
        seed=seed, seed_scheme="SeedSequence(seed,spawn_key=(10,atom_id));v1"), sort_keys=True).encode()).hexdigest()
    manifest_path = directory / "source_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8"))["signature"] != signature:
            raise ValueError("Source cache settings differ; choose a separate output directory")
    else:
        manifest_path.write_text(json.dumps(dict(signature=signature, seed=seed,
            source=source_options, pylcp=asdict(config.pylcp)), indent=2), encoding="utf-8")
    missing = [i for i in range(count) if not (cache / ("atom_%06d.npz" % i)).exists()]
    completed = count - len(missing)
    if progress:
        progress("Fresh native MOT trajectories: %d cached, %d remaining" % (completed, len(missing)))
    with ProcessPoolExecutor(max_workers=workers, initializer=_source_init,
                             initargs=(config.to_dict(), seed)) as pool:
        futures = {pool.submit(_source_atom, i): i for i in missing}
        for future in as_completed(futures):
            atom = future.result()
            path = cache / ("atom_%06d.npz" % atom["atom_id"])
            temporary = path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, **atom)
            temporary.replace(path)
            completed += 1
            if progress and (completed % 16 == 0 or completed == count):
                progress("Native MOT launches completed: %d/%d" % (completed, count))
    records = []
    statuses = {}
    for i in range(count):
        with np.load(cache / ("atom_%06d.npz" % i), allow_pickle=False) as handle:
            atom = {key: handle[key] for key in handle.files}
        status = str(atom["status"])
        statuses[status] = statuses.get(status, 0) + 1
        if status == "extracted":
            records.append(atom)
    if not records:
        raise ValueError("No atoms extracted from the fresh source batch")
    r = np.array([a["r_m"] for a in records])
    r[:, 2] = 0
    ensemble = Ensemble(r, np.array([a["v_m_s"] for a in records]),
        config.source.incident_flux_per_s * len(records) / count,
        {"mode": "pylcp", "sampled": count, "extracted": len(records),
         "extraction_fraction": len(records) / count, "status_counts": statuses,
         "incident_flux_per_s": config.source.incident_flux_per_s,
         "absolute_flux_is_calibrated": False, "source_seed": seed,
         "sample_ids": [int(a["atom_id"]) for a in records],
         "source_signature": signature, "seed_scheme": "per-atom, independent of batch size and workers",
         "model": "native 24-level PyLCP MOT; fresh independent trajectories",
         "spin_transport": "adiabatic local magnetic axis; sudden switch to lab-z at exit",
         "recoil_model": config.pylcp.recoil_model},
        ground_density=np.array([a["ground_density"] for a in records]),
        exit_populations=np.array([a["populations"] for a in records]),
        exit_time_s=np.array([a["time_s"] for a in records]),
        quantization_axes=np.array([a["axis"] for a in records]))
    save_stateful_ensemble(ensemble, directory / "source_ensemble.h5")
    return ensemble


def main():
    import argparse
    from .presets import pylcp_mot_preset
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atoms", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20087)
    parser.add_argument("--output", type=Path, default=Path("beam_outputs/pumping_optimization/source"))
    args = parser.parse_args()
    generate_source_batch(pylcp_mot_preset(), args.atoms, args.output, args.workers,
                          args.seed, progress=lambda message: print(message, flush=True))


if __name__ == "__main__":
    main()
