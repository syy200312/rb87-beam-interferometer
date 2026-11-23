"""Export finite-transit Rabi and chirp-rate scans without changing state weights."""

import csv
import json
from pathlib import Path

import numpy as np

from .io import save_stateful_ensemble
from .states import GROUND_LABELS, G_CLOCK


def _json_value(value):
    """Represent unavailable statistics as null, never emit nonstandard NaN JSON."""
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def _write_json(path, value):
    path.write_text(json.dumps(_json_value(value), indent=2, ensure_ascii=False,
                               allow_nan=False), encoding="utf-8")


def _write_csv(path, columns):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([name for name, values in columns])
        writer.writerows(zip(*(np.asarray(values) for name, values in columns)))


def _readout_curves(ax, x, detection, include_readout=True):
    ax.plot(x, detection["expected_atom_P2"], "--", color="#267b9c",
            label="Atom-weighted beam expectation")
    ax.plot(x, detection["expected_photon_P2"], color="#cf682a",
            label="Photon-weighted beam expectation")
    if include_readout:
        ax.scatter(x, detection["observed_P2"], s=12, color="#374151",
                   label="Simulated readout", zorder=3)
    ax.set_ylabel("Normalized total F=2 fraction")
    ax.grid(alpha=0.2)


def _fit_label(fit):
    contrast = fit.get("contrast")
    phase = fit.get("phase_rad")
    contrast_text = "undefined" if contrast is None else "%.2f%%" % (100 * contrast)
    phase_text = "undefined" if phase is None else "%.4f rad" % phase
    return "C_fit = %s; signed phase = %s" % (contrast_text, phase_text)


def save_scan_plot(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    experiment = result["experiment"]
    detection = result["detection"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    if experiment == "rabi":
        areas = np.asarray(result["reference_area_pi"])
        power_axis = "beam1_power_mW" in result and "beam2_power_mW" in result
        x = (np.asarray(result["beam1_power_mW"]) + np.asarray(result["beam2_power_mW"])
             if power_axis else areas)
        _readout_curves(axes[0], x, detection)
        axes[0].plot(x, np.sin(np.pi * areas / 2) ** 2, ":", color="#8b5cf6",
                     label="Ideal resonant pure clock state")
        axes[0].set(xlabel="Total power of both Raman colors at atoms (mW)" if power_axis else "Reference Raman pulse area / pi",
                    title="Single-zone Rabi oscillation: optical-power scan" if power_axis else "Single-zone Rabi oscillation: coupling-amplitude scan")
        transit = result["transit"]
        times = np.asarray(transit["reference_elapsed_s"]) * 1e6
        _readout_curves(axes[1], times, transit, include_readout=False)
        axes[1].set(xlabel="Reference elapsed transit time (us; position-aligned)",
                    title="Single-zone transit; reference pulse area = %.3g pi" %
                          transit["reference_area_pi"])
        fig.suptitle("87Rb beam Rabi scan: %d Monte Carlo atoms" % len(result["ensemble"].r_m))
    elif experiment == "chirp":
        x = np.asarray(result["chirp_Hz_s"]) * 1e-6
        _readout_curves(axes[0], x, detection)
        phase = np.asarray(result["phase_coordinate_rad"])
        fit = result.get("fit")
        if fit is not None and fit.get("coefficients") is not None:
            a, b, d = fit["coefficients"]
            order = np.argsort(x)
            dense_x = np.linspace(x.min(), x.max(), 701)
            dense_phase = np.interp(dense_x, x[order], phase[order])
            axes[0].plot(dense_x, a + b * np.cos(dense_phase) + d * np.sin(dense_phase),
                         ":", color="black", label="Nominal-period diagnostic fit")
        title = "Three-pulse interferometer: chirp-rate scan"
        visibility = result.get("sampled_expected_visibility")
        if visibility is not None:
            title += "\nSampled visibility = %.2f%%" % (100 * visibility)
        if fit is not None:
            title += "; " + _fit_label(fit) + "\nNominal-period fit RMS = %.4g (approximate)" % fit["residual_rms"]
        axes[0].set(xlabel="Raman frequency-difference chirp rate (MHz/s)", title=title)
        axes[1].plot(x, detection["F1_counts"], label="F=1 channel (ideal repump)")
        axes[1].plot(x, detection["F2_counts"], label="F=2 channel")
        axes[1].set(xlabel="Raman frequency-difference chirp rate (MHz/s)",
                    ylabel="Photoelectron counts / gate",
                    title="Fluorescence readout; third-pulse added phase held fixed")
        axes[1].grid(alpha=0.2)
        fig.suptitle("87Rb beam: %d atoms; T_ref = %.4f ms; alpha_ref = %.5g MHz/s" %
                     (len(result["ensemble"].r_m), result["reference_T_s"] * 1e3,
                      result["reference_chirp_Hz_s"] * 1e-6))
    else:
        plt.close(fig)
        raise ValueError("Unknown scan experiment: " + str(experiment))
    for ax in axes:
        ax.legend(fontsize=8, loc="best")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def export_scan_result(result, directory, plot=True):
    """Save SI arrays, state checkpoint, readout CSV, and explicitly labelled figures."""
    experiment = result["experiment"]
    if experiment not in ("rabi", "chirp"):
        raise ValueError("Scan export supports rabi or chirp")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config, ensemble, detection = result["config"], result["ensemble"], result["detection"]
    _write_json(directory / "config_used.json", config.to_dict())
    columns = [("scan_value", result["scan_values"])]
    arrays = {
        "scan_values": result["scan_values"], "r_m": ensemble.r_m,
        "v_m_s": ensemble.v_m_s, "flux_per_s": ensemble.flux_per_s,
        "reference_plane_z_m": ensemble.reference_plane_z_m,
        "source_trace": result.get("source_trace", np.empty((0, 5))),
        "per_atom_P2": result["per_atom_P2"],
        "detection_weights": detection["detection_weights"],
        "ground_basis_F_mF": GROUND_LABELS,
    }
    if experiment == "rabi":
        columns = [("reference_area_pi", result["reference_area_pi"]),
                   ("peak_rabi_Hz", result["peak_rabi_Hz"])]
        arrays.update(reference_area_pi=result["reference_area_pi"],
                      peak_rabi_Hz=result["peak_rabi_Hz"])
        for name in ("beam1_power_mW", "beam2_power_mW"):
            if name in result:
                columns.append((name, result[name]))
                arrays[name] = result[name]
    else:
        columns = [("chirp_Hz_s", result["chirp_Hz_s"]),
                   ("chirp_rad_s2", result["chirp_rad_s2"]),
                   ("phase_coordinate_rad", result["phase_coordinate_rad"])]
        arrays.update(chirp_Hz_s=result["chirp_Hz_s"],
                      chirp_rad_s2=result["chirp_rad_s2"],
                      phase_coordinate_rad=result["phase_coordinate_rad"])
    readout_names = ("F1_counts", "F2_counts", "observed_P2",
                     "expected_photon_P2", "expected_atom_P2")
    for name in readout_names:
        columns.append((name, detection[name]))
        arrays[name] = detection[name]
    _write_csv(directory / (experiment + ".csv"), columns)
    for name in ("ground_density", "exit_populations", "exit_time_s",
                 "quantization_axes", "population_trace", "reference_time_s"):
        value = getattr(ensemble, name)
        if value is not None:
            arrays[name] = value
    populations = result.get("per_atom_ground_populations")
    if populations is not None:
        arrays["raman_ground_populations"] = populations
    if experiment == "rabi":
        transit = result["transit"]
        names = ("reference_elapsed_s", "expected_photon_P2", "expected_atom_P2")
        _write_csv(directory / "rabi_transit.csv", [(name, transit[name]) for name in names])
        arrays.update({"transit_" + name: transit[name] for name in names})
        arrays["transit_reference_area_pi"] = transit["reference_area_pi"]
    np.savez_compressed(directory / "scan.npz", **arrays)
    save_stateful_ensemble(ensemble, directory / "source_ensemble.h5")
    if result.get("pumping") is not None:
        save_stateful_ensemble(result["source_ensemble"], directory / "mot_ensemble.h5")
        _write_json(directory / "pumping_summary.json", result["pumping"]["diagnostics"])
    weights = np.asarray(detection["detection_weights"])
    summary = {
        "experiment": experiment, "sample_count": len(ensemble.r_m),
        "scan_points": len(result["scan_values"]), "state_handling": result["state_handling"],
        "source": ensemble.source_metadata, "exit_flux_per_s": ensemble.flux_per_s,
        "detected_flux_per_s": detection["detected_flux_per_s"],
        "expected_atoms_per_gate": detection["expected_atoms_per_gate"],
        "effective_detected_MC_atoms": weights.sum() ** 2 / np.sum(weights ** 2),
        "max_state_norm_error": result["max_state_norm_error"],
        "state_checkpoint": "source_ensemble.h5", "scan_data": experiment + ".csv",
        "scan_arrays": "scan.npz", "reference_plane_z_m": ensemble.reference_plane_z_m,
        "scope": [
            "Finite Gaussian or rectangular Raman envelopes; effective two-state clock coupling",
            "Raman evolves the clock pair only; six spectator Zeeman levels retain their populations, not a full 24-level Raman OBE",
            "Native upstream MOT and optical pumping, when computed, use 24-level population rate equations",
            "Inherited Zeeman populations remain in total F2 readout; no ideal state reset in inherit mode",
            "Atom-weighted and residence-time photon-weighted readout are saved separately",
            "Absolute flux is a configured normalization unless independently calibrated",
        ],
        "not_modelled": ["Multilevel coherent Raman pathways and spontaneous Raman scattering",
                         "Separate wave-packet branches and imperfect spatial closure"],
    }
    if "clock_convention" in result:
        summary["clock_convention"] = result["clock_convention"]
    if "raman_power_calibration" in result:
        summary["raman_power_calibration"] = result["raman_power_calibration"]
        summary["raman_spin_transport"] = result["raman_spin_transport"]
        summary["scope"].append("Raman coupling and AC light shift computed from optical powers via coherent D2 hyperfine pathways")
        summary["not_modelled"][0] = "Full multilevel coherent Raman dynamics and spontaneous Raman scattering trajectories (rates are diagnostics only)"
    if ensemble.ground_density is not None:
        population = np.real(np.diagonal(ensemble.ground_density, axis1=1, axis2=2))
        summary["ground_basis_F_mF"] = GROUND_LABELS
        summary["input_ground_population_mean"] = population.mean(axis=0)
        summary["input_target_state_flux_per_s"] = ensemble.flux_per_s * population[:, G_CLOCK].mean()
    if populations is not None:
        summary["max_raman_ground_population_norm_error"] = np.max(abs(populations.sum(axis=-1) - 1))
    if experiment == "rabi":
        summary.update(
            scan_axis="Reference Raman pulse area / pi; peak coupling varied at fixed geometry",
            reference_transit_s=result["reference_transit_s"],
            reference_area_pi_range=[np.min(result["reference_area_pi"]), np.max(result["reference_area_pi"])],
            peak_rabi_Hz_range=[np.min(result["peak_rabi_Hz"]), np.max(result["peak_rabi_Hz"])],
            transit_reference_area_pi=result["transit"]["reference_area_pi"],
            transit_time_convention="Position-aligned transit history; time axis uses reference longitudinal speed, not common laboratory time")
    else:
        summary.update(
            scan_axis="Linear chirp rate of Raman frequency difference, Hz/s",
            chirp_conversion="chirp_rad_s2 = 2*pi*chirp_Hz_s",
            chirp_Hz_s_range=[np.min(result["chirp_Hz_s"]), np.max(result["chirp_Hz_s"])],
            reference_chirp_Hz_s=result["reference_chirp_Hz_s"],
            reference_T_s=result["reference_T_s"],
            fit=result.get("fit"), expected_fit=result.get("expected_fit"),
            sampled_expected_visibility=result.get("sampled_expected_visibility"),
            fit_definition="Fixed-reference-sensitivity sinusoid; C=amplitude/positive offset, phase is signed in rad",
            fit_limitation="Finite pulse duration, Doppler detuning and velocity spread can make a chirp scan non-sinusoidal; inspect residual_rms and raw data",
            phase_scan_convention="Third-pulse added phase is held fixed while all three pulses are propagated at each chirp rate")
        for name in ("chirp_time_origin", "chirp_time_origin_s", "phase_reference", "timing_convention"):
            if name in result:
                summary[name] = result[name]
    if plot:
        try:
            save_scan_plot(result, directory / (experiment + ".png"))
            summary["plot"] = experiment + ".png"
        except ModuleNotFoundError as error:
            if error.name != "matplotlib":
                raise
            summary["plot"] = "omitted: install matplotlib to render the scan"
    summary = _json_value(summary)
    _write_json(directory / "summary.json", summary)
    return summary
