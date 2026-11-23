"""Atom-resolved preparation and fluorescence for a matched low-power comparison.

The grid uses the existing full 24-state PyLCP rate equations on prescribed
trajectories. Native validation restores the mean light force and stochastic
recoil. Both routes return the input format of joint_statistics.case_statistics.
"""
import copy
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from .batch import subset_ensemble
from .joint_optimization import _density, response_record
from .optimization import ballistic_screen, prepare_native_batch
from .physics import Ensemble, flight_time, trajectory
from .states import G_CLOCK


def pump_config(base, power_uW, length_mm, detuning_MHz, transfer_fraction=None):
    """Copy base; set total pump power, full truncated length and common detuning.

    Both optical frequencies are detuned by the supplied ordinary frequency
    from their own zero-field line. 'Transfer fraction' is the share of total
    pump power in F=2 -> F'=1. The two propagation directions for each
    frequency, transverse waist, polarization, bias and all Raman optics are
    inherited. As in OpticalPumping, peak intensity changes with the waist.
    """
    c = copy.deepcopy(base)
    p = c.pumping
    if transfer_fraction is None:
        total = p.transfer_power_W + p.dark_power_W
        if total <= 0:
            raise ValueError("Set transfer_fraction for a zero-power template")
        transfer_fraction = p.transfer_power_W / total
    if (not np.all(np.isfinite([power_uW, length_mm, detuning_MHz, transfer_fraction]))
            or power_uW <= 0 or length_mm <= 0 or not 0 < transfer_fraction < 1):
        raise ValueError("Power and length must be positive, detuning finite, split strictly between 0 and 1")
    p.enabled = True
    p.transfer_power_W = float(power_uW * 1e-6 * transfer_fraction)
    p.dark_power_W = float(power_uW * 1e-6 * (1 - transfer_fraction))
    p.waist_z_m = float(length_mm * 1e-3 / (2 * p.half_widths))
    p.transfer_detuning_Hz = p.dark_detuning_Hz = float(detuning_MHz * 1e6)
    c.validate()
    return c


def _evaluation_config(config, source):
    c = copy.deepcopy(config)
    if (not c.pumping.enabled or c.raman.state_handling not in ("inherit", "auto")
            or source.ground_density is None or source.reference_plane_z_m != 0):
        raise ValueError("Comparison requires a stateful MOT exit, enabled pump and inherited Raman populations")
    if np.linalg.norm(c.raman.rotation_rad_s) or c.raman.acceleration_m_s2[0]:
        raise ValueError("Matched ballistic/native comparison requires zero rotation and x acceleration")
    c.detection.noise = False
    # Only the third-zone phase changes. Its first harmonic is exact for
    # the clock-pair unitary; 5 phases suffice to fit the 3 real coefficients.
    c.raman.scan_points = 5
    c.validate()
    source.validate()
    return c


def screen_record(config, source):
    """Return ballistic per-atom scatter, inherited-state fringe and aperture weights.

    Each one-atom call reuses the cached native rate matrix builder. Recording
    the actual per-atom scatter (instead of repeating the group mean) preserves
    correlations for matched-atom differences.
    """
    c = _evaluation_config(config, source)
    rows = [ballistic_screen(c, subset_ensemble(source, [j])) for j in range(len(source.r_m))]
    pop = np.concatenate([row["ground_populations"] for row in rows], axis=0)
    photons = np.array([row["mean_scattered_photons"] for row in rows])
    end = c.pumping.center_z_m + c.pumping.half_widths * c.pumping.waist_z_m
    times = flight_time(source, end)
    r, v = trajectory(source, times, c.raman)
    prepared = Ensemble(r, v, source.flux_per_s, copy.deepcopy(source.source_metadata),
                        ground_density=_density(pop), reference_plane_z_m=end, reference_time_s=times)
    return response_record(c, prepared, photons)


def native_records(config, source, directory, replicas=2, workers=4, seed=61087, progress=None,
                   executor=None):
    """Re-evaluate a point with native 24-level motion and per-source seeded recoil.

    Use a separate cache directory per point and input pool. Identical values
    of seed, source atom ID and replica give paired random-number streams in
    both arms, even though kicks differ with detuning and power.
    """
    if any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 1
           for n in (replicas, workers)):
        raise ValueError("replicas and workers must be positive integers")
    c = _evaluation_config(config, source)
    runs = prepare_native_batch(c, source, directory, replicas=replicas,
                                workers=workers, seed=seed, progress=progress, executor=executor)
    records = []
    for result, photons in runs:
        e = result["ensemble"]
        phase = result["phase_rad"]
        design = np.column_stack([np.ones_like(phase), np.cos(phase), np.sin(phase)])
        records.append(dict(atom_ids=np.asarray(e.source_metadata.get("sample_ids", np.arange(len(e.r_m))), int),
                            fringe_coefficients=np.linalg.lstsq(design, result["per_atom_P2"], rcond=None)[0].T,
                            detection_weights=result["detection"]["detection_weights"],
                            vz_m_s=e.v_m_s[:, 2],
                            target_population=e.ground_density[:, G_CLOCK, G_CLOCK].real,
                            scattered_photons=np.asarray(photons)))
    return records


def native_records_many(configs, source, directory, replicas=2, workers=4, seed=61087,
                        progress=None, executor=None):
    """Evaluate a mapping of case names to configs using one shared worker pool.

    Pump caches are kept under directory/<case name>. Distinct Raman configs
    are allowed across runs of an identical pump cache; Raman propagation is
    always recalculated from the cached positions, velocities and populations.
    """
    if not configs:
        return {}
    if any(not isinstance(name, str) or not name or name in (".", "..")
           or "/" in name or "\\" in name or ":" in name for name in configs):
        raise ValueError("Case names must be plain directory names")
    with (ProcessPoolExecutor(max_workers=workers) if executor is None else nullcontext(executor)) as pool:
        return {name: native_records(config, source, Path(directory) / name, replicas, workers, seed,
                                     progress, executor=pool) for name, config in configs.items()}
