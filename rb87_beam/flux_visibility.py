"""Flux-only sweeps of the normalized three-pulse fluorescence phase fringe.

The prepared phase-space mixture, Raman response and gains are frozen. The
experiment here is a FULL phase sweep at each repeat, with an independent gate
at every phase. These intervals are repeat-to-repeat detection variability
conditional on the simulated source, not uncertainty in that source.
"""

from dataclasses import dataclass
import struct

import numpy as np

from .physics import GAMMA, fit_fringe


@dataclass(frozen=True)
class VisibilitySnapshot:
    phase_rad: np.ndarray
    per_atom_P2: np.ndarray
    detection_weights: np.ndarray
    q1: np.ndarray
    q2: np.ndarray
    extracted_fraction: float
    target_population: object
    gain1: float
    gain2: float
    gate_s: float
    background_counts: float
    electronic_noise_std_counts: float


def make_visibility_snapshot(
    phases,
    per_atom_P2,
    detection_weights,
    vz_m_s,
    detection,
    extracted_fraction=1.0,
    target_population=None,
):
    """Freeze a phase-by-extracted-trajectory response and post-aperture weights.

    Input flux below denotes INCIDENT flux. extracted_fraction includes failed
    launches. Multiple equally represented recoil replicas can be concatenated
    along the atom axis: this represents a mixture of possible emitted atoms,
    NOT extra flux. For a target-state flux, target_population is the initial
    clock-state probability on top of the supplied detection weights (do not
    multiply by it again if the weights already select those atoms).
    """
    phase = np.asarray(phases, dtype=float)
    prob = np.asarray(per_atom_P2, dtype=float)
    w = np.asarray(detection_weights, dtype=float)
    v = np.asarray(vz_m_s, dtype=float)
    if phase.ndim != 1 or len(phase) < 4 or not np.all(np.isfinite(phase)):
        raise ValueError("At least four finite phases are required")
    design = np.column_stack([np.ones(len(phase)), np.cos(phase), np.sin(phase)])
    if np.linalg.matrix_rank(design) < 3:
        raise ValueError("Phases must resolve offset, cosine and sine")
    if (
        w.ndim != 1
        or len(w) < 1
        or v.shape != w.shape
        or prob.shape != (len(phase), len(w))
        or not np.all(np.isfinite(prob))
        or not np.all(np.isfinite(w))
        or not np.all(np.isfinite(v))
        or np.any(v <= 0)
        or np.any(w < 0)
        or np.any(w > 1 + 1e-10)
        or np.sum(w) <= 0
    ):
        raise ValueError("Invalid trajectory probabilities, weights or speeds")
    if np.min(prob) < -1e-10 or np.max(prob) > 1 + 1e-10:
        raise ValueError("Population outside [0,1]")
    if not 0 < extracted_fraction <= 1:
        raise ValueError("Require positive extraction fraction <=1")
    if target_population is not None:
        target_population = np.asarray(target_population, dtype=float)
        if (
            target_population.shape != w.shape
            or not np.isfinite(target_population).all()
            or target_population.min() < -1e-10
            or target_population.max() > 1 + 1e-10
        ):
            raise ValueError("Invalid target-state population")
        target_population = np.clip(target_population, 0, 1)
    d = detection if isinstance(detection, dict) else vars(detection)
    sat = np.array([d["saturation_F1"], d["saturation_F2"]], dtype=float)
    scalars = np.r_[
        sat,
        [
            d[k]
            for k in (
                "interaction_length_m",
                "collection_fraction",
                "quantum_efficiency",
                "gate_s",
                "background_counts",
                "electronic_noise_std_counts",
            )
        ],
    ]
    if (
        not np.isfinite(scalars).all()
        or np.any(scalars[:6] <= 0)
        or np.any(scalars[6:] < 0)
        or max(d["collection_fraction"], d["quantum_efficiency"]) > 1
    ):
        raise ValueError("Invalid detector gains, gate or noise")
    eff = d["collection_fraction"] * d["quantum_efficiency"]
    q1, q2 = (GAMMA * 0.5 * sat / (1 + sat) * d["interaction_length_m"] * eff)[:, None] / v[None, :]
    return VisibilitySnapshot(
        phase.copy(),
        np.clip(prob, 0, 1),
        np.clip(w, 0, 1),
        q1,
        q2,
        float(extracted_fraction),
        target_population,
        float(np.average(q1, weights=w)),
        float(np.average(q2, weights=w)),
        float(d["gate_s"]),
        float(d["background_counts"]),
        float(d["electronic_noise_std_counts"]),
    )


def from_simulation(result, extracted_fraction=1.0, target_population=None):
    """Wrap run_simulation(): use its effective post-aperture/selection weights."""
    return make_visibility_snapshot(
        result["phase_rad"],
        result["per_atom_P2"],
        result["detection"]["detection_weights"],
        result["ensemble"].v_m_s[:, 2],
        result["config"].detection,
        extracted_fraction,
        target_population,
    )


def _example(values):
    return [float(x) if np.isfinite(x) else None for x in values]


def scan_flux_visibility(snapshot, fluxes_per_s, repeats=400, seed=20261122, gate_s=None):
    """Exact compound-Poisson detector replicas, vectorized without grouping speeds.

    Invalid nonpositive total counts yield a missing phase. As in fit_fringe,
    fits need >=4 finite phase samples, rank 3 and a positive offset. Negative
    net channel signals, ratios outside [0,1] and contrasts above one are NOT
    clipped. Quantiles (2.5/50/97.5%) are conditional on successful fits;
    separate diagnostics describe missing/physically outlying fits.
    background_counts is the known mean PER channel PER gate in fluorescence_scan;
    electronic noise is per read. Changing gate_s alone leaves these fixed.
    """
    s = snapshot
    fluxes = np.asarray(fluxes_per_s, dtype=float)
    if (
        fluxes.ndim != 1
        or not len(fluxes)
        or not np.isfinite(fluxes).all()
        or np.any(fluxes <= 0)
        or len(np.unique(fluxes)) != len(fluxes)
    ):
        raise ValueError("Supply unique positive finite incident fluxes")
    if not isinstance(repeats, (int, np.integer)) or isinstance(repeats, bool) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    gate = s.gate_s if gate_s is None else float(gate_s)
    if not np.isfinite(gate) or gate <= 0:
        raise ValueError("gate_s must be finite and positive")
    nphase, natoms = s.per_atom_P2.shape
    # Mean counts per incident atom for each phase, including all losses.
    weights = s.detection_weights
    normalized_w = s.extracted_fraction / natoms * weights
    rate1 = (1 - s.per_atom_P2) * normalized_w[None, :]
    rate2 = s.per_atom_P2 * normalized_w[None, :]
    mu1 = np.einsum("pi,i->p", rate1, s.q1)
    mu2 = np.einsum("pi,i->p", rate2, s.q2)
    ideal = (mu2 / s.gain2) / (mu1 / s.gain1 + mu2 / s.gain2)
    ideal_fit = fit_fringe(s.phase_rad, ideal)
    design = np.column_stack((np.ones(nphase), np.cos(s.phase_rad), np.sin(s.phase_rad)))
    # Approx 1 million Poisson values per array, not repeats*phase*sample at once.
    chunk = max(1, min(64, 1_000_000 // (natoms * nphase)))
    results = []
    for raw_flux in fluxes:
        flux = float(raw_flux)
        # A point's random stream is independent of flux grid order and length.
        flux_bits = struct.unpack("<II", struct.pack("<d", flux))
        gen = np.random.default_rng(np.random.SeedSequence(int(seed), spawn_key=flux_bits))
        lam1 = rate1 * flux * gate
        lam2 = rate2 * flux * gate
        valid_points = 0
        complete = 0
        unphysical = 0
        above_one = 0
        Cs = []
        first = {}
        for start in range(0, repeats, chunk):
            m = min(chunk, repeats - start)

            def channel(lam, q):
                n = gen.poisson(lam, size=(m, nphase, natoms))
                mu = np.einsum("rpi,i->rp", n, q)
                return gen.poisson(mu + s.background_counts) + gen.normal(
                    0, s.electronic_noise_std_counts, size=(m, nphase)
                )

            raw1 = channel(lam1, s.q1)
            raw2 = channel(lam2, s.q2)
            a = (raw1 - s.background_counts) / s.gain1
            b = (raw2 - s.background_counts) / s.gain2
            denom = a + b
            y = np.divide(b, denom, out=np.full_like(denom, np.nan), where=denom > 0)
            for j, line in enumerate(y):
                count = int(np.isfinite(line).sum())
                valid_points += count
                complete += int(count == nphase)
                try:
                    f = fit_fringe(s.phase_rad, line)
                except ValueError:
                    f = None
                c = None if f is None else f["contrast"]
                ok = c is not None and np.isfinite(c)
                if ok:
                    Cs.append(c)
                    above_one += int(c > 1)
                    # Count fits with Pmax>1 or Pmin<0. Keep them in the quantiles.
                    unphysical += int(
                        f["offset"] + f["amplitude"] > 1 + 1e-10
                        or f["offset"] - f["amplitude"] < -1e-10
                    )
                if start == 0 and j == 0:
                    first = dict(
                        example_P2=_example(line),
                        example_F1_counts=raw1[j].tolist(),
                        example_F2_counts=raw2[j].tolist(),
                        example_contrast=c if ok else None,
                        example_fit_P2=(
                            _example(design @ f["coefficients"]) if ok else [None] * nphase
                        ),
                        example_valid_phase_points=count,
                    )
        nfit = len(Cs)
        lo, median, hi = (
            (float(x) for x in np.quantile(Cs, [0.025, 0.5, 0.975])) if nfit else (None, None, None)
        )
        results.append(
            dict(
                incident_flux_per_s=flux,
                extracted_flux_per_s=flux * s.extracted_fraction,
                detected_flux_per_s=float(flux * np.sum(normalized_w)),
                effective_target_flux_per_s=(
                    None
                    if s.target_population is None
                    else float(flux * np.dot(normalized_w, s.target_population))
                ),
                expected_atoms_per_gate=float(flux * gate * np.sum(normalized_w)),
                expected_F1_counts=(flux * gate * mu1 + s.background_counts).tolist(),
                expected_F2_counts=(flux * gate * mu2 + s.background_counts).tolist(),
                ideal_contrast=ideal_fit["contrast"],
                fitted_median=median,
                fitted_q025=lo,
                fitted_q975=hi,
                n_fits=nfit,
                fit_success_rate=nfit / repeats,
                unphysical_fit_fraction=unphysical / nfit if nfit else None,
                fitted_above_one_fraction=above_one / nfit if nfit else None,
                valid_phase_fraction=valid_points / (nphase * repeats),
                complete_scan_fraction=complete / repeats,
                **first,
            )
        )
    return dict(
        phase_rad=s.phase_rad.tolist(),
        ideal_P2=ideal.tolist(),
        ideal_contrast=ideal_fit["contrast"],
        ideal_fit=ideal_fit,
        gate_s=gate,
        repeats=int(repeats),
        seed=int(seed),
        background_counts_per_channel_gate=s.background_counts,
        electronic_noise_std_counts=s.electronic_noise_std_counts,
        gain_F1_counts_per_atom=s.gain1,
        gain_F2_counts_per_atom=s.gain2,
        profiles=results,
    )
