"""Artifact export and explicit conversion from existing PyLCP HDF5 solutions."""

import csv
import json
from pathlib import Path

import numpy as np

from .physics import Ensemble, KB, MASS, K_EFF, HBAR, HYPERFINE_HZ
from .states import GROUND_LABELS, EXCITED_LABELS, G_CLOCK, E_CLOCK


def save_stateful_ensemble(ensemble, path):
    """Versioned SI source checkpoint; no state-order or unit inference on reload."""
    import h5py

    ensemble.validate()
    with h5py.File(path, "w") as handle:
        handle.attrs["schema"] = "rb87_beam.source.v2"
        handle.attrs["reference_plane_z_m"] = ensemble.reference_plane_z_m
        handle.attrs["position_unit"] = "m"
        handle.attrs["velocity_unit"] = "m/s"
        handle.attrs["ground_basis"] = "lab_z_F_mF"
        handle.attrs["exit_population_basis"] = "local_B_F_mF"
        handle.attrs["metadata_json"] = json.dumps(ensemble.source_metadata)
        handle.attrs["flux_per_s"] = ensemble.flux_per_s
        handle.create_dataset("r_m", data=ensemble.r_m)
        handle.create_dataset("v_m_s", data=ensemble.v_m_s)
        handle.create_dataset("ground_labels_F_mF", data=GROUND_LABELS)
        handle.create_dataset("excited_labels_F_mF", data=EXCITED_LABELS)
        for name in ("ground_density", "exit_populations", "exit_time_s", "quantization_axes", "population_trace", "reference_time_s"):
            value = getattr(ensemble, name)
            if value is not None:
                handle.create_dataset(name, data=value, compression="gzip")


def read_stateful_ensemble(path):
    import h5py

    with h5py.File(path, "r") as handle:
        if handle.attrs.get("schema") not in ("rb87_beam.source.v1", "rb87_beam.source.v2"):
            raise ValueError("Unsupported stateful ensemble schema")
        if handle.attrs["schema"] == "rb87_beam.source.v2" and "reference_plane_z_m" not in handle.attrs:
            raise ValueError("Version 2 checkpoints require an explicit reference plane")
        required = {"r_m", "v_m_s", "ground_labels_F_mF", "excited_labels_F_mF"}
        if not required.issubset(handle) or "flux_per_s" not in handle.attrs or "metadata_json" not in handle.attrs:
            raise ValueError("Incomplete source checkpoint")
        if (handle.attrs.get("position_unit") != "m" or handle.attrs.get("velocity_unit") != "m/s"
                or handle.attrs.get("ground_basis") != "lab_z_F_mF"
                or handle.attrs.get("exit_population_basis") != "local_B_F_mF"):
            raise ValueError("Source checkpoint must have SI units and explicit supported bases")
        if (not np.array_equal(handle["ground_labels_F_mF"][:], GROUND_LABELS)
                or not np.array_equal(handle["excited_labels_F_mF"][:], EXCITED_LABELS)):
            raise ValueError("Unexpected source basis order")
        extras = {name: np.asarray(handle[name]) for name in (
            "ground_density", "exit_populations", "exit_time_s", "quantization_axes", "population_trace", "reference_time_s") if name in handle}
        extras["reference_plane_z_m"] = float(handle.attrs.get("reference_plane_z_m", 0.0))
        ensemble = Ensemble(np.asarray(handle["r_m"]), np.asarray(handle["v_m_s"]),
            float(handle.attrs["flux_per_s"]), json.loads(handle.attrs["metadata_json"]), **extras)
    ensemble.validate()
    return ensemble


def read_legacy_h5(path, length_scale_m, time_scale_s, exit_z_m,
                   flux_per_s, axis_order="xyz"):
    """Take FINAL endpoints from sol_i/{r,v}; convert units, rotate, cross exit plane.

    axis_order='x,-z,y' maps old (x,y,z) -> new (x,-z,y), a proper rotation.
    Improper rotations are rejected. No internal-state indices
    are guessed; downstream state preparation is an explicit ideal reset.
    """
    import h5py

    axes = axis_order.split(",") if "," in axis_order else list(axis_order)
    if sorted(axis.lstrip("+-") for axis in axes) != list("xyz"):
        raise ValueError("axis_order must be a permutation of xyz")
    if not np.isfinite(length_scale_m) or not np.isfinite(time_scale_s) or min(length_scale_m, time_scale_s) <= 0:
        raise ValueError("Explicit positive length and time scales are required")
    order = ["xyz".index(axis.lstrip("+-")) for axis in axes]
    signs = np.array([-1 if axis.startswith("-") else 1 for axis in axes])
    if np.linalg.det(np.eye(3)[order] * signs[:, None]) < 0:
        raise ValueError("axis_order reverses handedness; use a proper rotation such as x,-z,y")
    positions, velocities = [], []

    def endpoint(dataset):
        value = np.asarray(dataset)
        if value.shape == (3,):
            return value
        if value.ndim == 2 and value.shape[0] == 3 and value.shape[1] > 0:
            return value[:, -1]
        raise ValueError("Expected r/v shape (3,) or (3,Nt), found " + str(value.shape))

    with h5py.File(path, "r") as handle:
        for name in sorted(handle, key=lambda item: int(item[4:]) if item.startswith("sol_") and item[4:].isdigit() else -1):
            group = handle[name]
            if not name.startswith("sol_") or not isinstance(group, h5py.Group):
                continue
            positions.append(endpoint(group["r"])[order] * signs * length_scale_m)
            velocities.append(endpoint(group["v"])[order] * signs * length_scale_m / time_scale_s)
    if not positions:
        raise ValueError("No sol_i/r,v groups found in input HDF5")
    r, v = np.asarray(positions), np.asarray(velocities)
    if not np.all(np.isfinite(r)) or not np.all(np.isfinite(v)) or not np.isfinite(exit_z_m):
        raise ValueError("Input endpoints and exit plane must be finite")
    if np.any(v[:, 2] <= 0) or np.any(r[:, 2] > exit_z_m):
        raise ValueError("All imported atoms must be upstream of exit_z_m with positive longitudinal velocity")
    t = (exit_z_m - r[:, 2]) / v[:, 2]
    r = r + v * t[:, None]
    r[:, 1] -= 0.5 * 9.80665 * t ** 2
    v[:, 1] -= 9.80665 * t
    r[:, 2] = 0
    ensemble = Ensemble(r, v, flux_per_s,
                        {"mode": "legacy_h5", "path": str(Path(path).resolve()),
                         "length_scale_m": length_scale_m, "time_scale_s": time_scale_s,
                         "axis_order": axis_order, "exit_z_m": exit_z_m,
                         "sampled": len(r), "extracted": len(r),
                         "absolute_flux_is_calibrated": False,
                         "internal_state": "discarded; ideal downstream preparation"})
    ensemble.validate()
    return ensemble


def save_plot(result, path):
    """Optional Matplotlib rendering. Numeric outputs are always available."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e, d = result["ensemble"], result["detection"]
    phase = result["phase_rad"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    ax = axes[0, 0]
    ax.hist(e.v_m_s[:, 2], bins=35, color="#267b9c")
    ax.set(xlabel="Longitudinal velocity (m/s)", ylabel="Monte Carlo atoms", title="Reference-plane distribution")
    ax = axes[0, 1]
    ax.scatter(e.v_m_s[:, 0], e.v_m_s[:, 1], s=5, alpha=0.4)
    ax.set(xlabel="vx (m/s)", ylabel="vy (m/s)", title="Transverse velocities before selection")
    ax = axes[1, 0]
    ax.plot(phase, d["expected_atom_P2"], "--", label="Atom-weighted expectation")
    ax.plot(phase, d["expected_photon_P2"], label="Photon-weighted expectation")
    ax.scatter(phase, d["observed_P2"], s=15, label="Simulated readout", color="#cf682a")
    dense = np.linspace(0, 2 * np.pi, 301)
    a, b, s = result["fit"]["coefficients"]
    ax.plot(dense, a + b * np.cos(dense) + s * np.sin(dense), ":", color="black", label="Fit")
    contrast, fitted_phase = result["fit"]["contrast"], result["fit"]["phase_rad"]
    contrast_text = "undefined" if contrast is None else "%.2f%%" % (100 * contrast)
    phase_text = "undefined" if fitted_phase is None else "%.4f rad" % fitted_phase
    ax.set(xlabel="Third-zone phase (rad)", ylabel="Normalized F=2 fraction",
           title="Spatial Raman MZ fringe\nContrast = %s; phase = %s" % (contrast_text, phase_text))
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    ax.plot(phase, d["F1_counts"], label="F=1 channel (ideal repump)")
    ax.plot(phase, d["F2_counts"], label="F=2 channel")
    ax.set(xlabel="Third-zone phase (rad)", ylabel="Photoelectron counts / gate", title="Fluorescence signals")
    ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.suptitle("87Rb continuous beam: " + e.source_metadata["mode"] + " source")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_source_plot(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trace = result["source_trace"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), constrained_layout=True)
    axes[0, 0].plot(trace[:, 0] * 1e3, trace[:, 3] * 1e6)
    if not np.any(np.isfinite(trace[:, 3])):
        axes[0, 0].text(0.5, 0.5, "Fewer than 20 active atoms:\ntemperature statistic omitted",
                         ha="center", va="center", transform=axes[0, 0].transAxes)
    axes[0, 0].set(ylabel="Transverse temperature (uK)", title="Remaining active population")
    axes[0, 1].plot(trace[:, 0] * 1e3, trace[:, 4])
    axes[0, 1].set(ylabel="Mean vz (m/s)", title="Cooling and pushing")
    axes[1, 0].plot(trace[:, 0] * 1e3, trace[:, 1])
    axes[1, 0].set(ylabel="Active Monte Carlo atoms", title="Atoms remaining in source")
    axes[1, 1].plot(trace[:, 0] * 1e3, trace[:, 2])
    axes[1, 1].set(ylabel="Extracted Monte Carlo atoms", title="Cumulative accepted first exits")
    for ax in axes.flat:
        ax.set_xlabel("Time (ms)")
        ax.grid(alpha=0.2)
    title = "PyLCP 24-state 2D+ MOT" if result["ensemble"].source_metadata["mode"] == "pylcp" else "Reduced 2D+ MOT"
    fig.suptitle(title + ": population changes as atoms leave")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_state_plot(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e = result["ensemble"]
    rho = e.ground_density
    ground = np.real(np.diagonal(rho, axis1=1, axis2=2))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    if e.exit_populations is not None:
        raw = e.exit_populations.mean(axis=0)
        grouped = [raw[:3].sum(), raw[3:8].sum()] + [raw[8:][EXCITED_LABELS[:, 0] == f].sum() for f in range(4)]
        axes[0, 0].bar(["g F1", "g F2", "e F0", "e F1", "e F2", "e F3"], grouped)
        axes[0, 0].set(title="At MOT exit: local magnetic basis", ylabel="Mean population")
    else:
        axes[0, 0].bar(["F1", "F2"], [ground[:, :3].sum(axis=1).mean(), ground[:, 3:].sum(axis=1).mean()])
        axes[0, 0].set(title="After preparation and free decay", ylabel="Mean population")
    labels = ["%d,%+d" % (f, m) for f, m in GROUND_LABELS]
    axes[0, 1].bar(labels, ground.mean(axis=0), color=["#cf682a" if i in (G_CLOCK, E_CLOCK) else "#267b9c" for i in range(8)])
    axes[0, 1].set(title="Reference plane: laboratory z basis", xlabel="F,mF (orange = Raman pair)", ylabel="Mean population")
    t = e.population_trace
    if t is not None:
        axes[1, 0].plot(t[:, 0] * 1e3, t[:, 2:5].sum(axis=1), label="Ground F=1")
        axes[1, 0].plot(t[:, 0] * 1e3, t[:, 5:10].sum(axis=1), label="Ground F=2")
        axes[1, 0].plot(t[:, 0] * 1e3, t[:, 10:].sum(axis=1), label="Excited")
    axes[1, 0].set(title="Active source population (changing sample)", xlabel="Time (ms)", ylabel="Mean population")
    if t is None and result.get("pumping") is not None:
        pump = result["pumping"]
        axes[1, 0].plot(pump["z_trace_m"] * 1e3, pump["population_trace"][:, G_CLOCK], label="F1,m=0")
        axes[1, 0].plot(pump["z_trace_m"] * 1e3, pump["population_trace"][:, 8:].sum(axis=1), label="Excited")
        axes[1, 0].set(title="Optical preparation", xlabel="z (mm)", ylabel="Mean population")
    if axes[1, 0].lines:
        axes[1, 0].legend(fontsize=8)
    else:
        axes[1, 0].text(.5, .5, "History is saved in the pumping run", ha="center", va="center", transform=axes[1, 0].transAxes)
    scan = result["per_atom_ground_populations"]
    if scan is not None:
        weight = result["detection"]["detection_weights"]
        mean = np.average(scan, axis=1, weights=weight)
        axes[1, 1].plot(result["phase_rad"], mean[:, E_CLOCK], label="Clock pair: F2,m=0")
        axes[1, 1].plot(result["phase_rad"], mean[:, GROUND_LABELS[:, 0] == 2].sum(axis=1) - mean[:, E_CLOCK], label="Spectator F2 states")
        axes[1, 1].plot(result["phase_rad"], mean[:, GROUND_LABELS[:, 0] == 2].sum(axis=1), label="Total F2")
        axes[1, 1].legend(fontsize=8)
    else:
        axes[1, 1].text(0.5, 0.5, "No complete state scan recorded\nfor this preparation mode",
                         ha="center", va="center", transform=axes[1, 1].transAxes)
    axes[1, 1].set(title="Raman output and retained state background", xlabel="Third-zone phase (rad)", ylabel="Atom-weighted population")
    for ax in axes.flat:
        ax.grid(alpha=0.2, axis="y")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def export_result(result, directory, plot=True):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    c, e, d = result["config"], result["ensemble"], result["detection"]
    c.save(directory / "config_used.json")
    with (directory / "fringe.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["phase_rad", "F1_counts", "F2_counts", "observed_P2",
                         "expected_photon_P2", "expected_atom_P2"])
        writer.writerows(zip(result["phase_rad"], d["F1_counts"], d["F2_counts"],
                             d["observed_P2"], d["expected_photon_P2"], d["expected_atom_P2"]))
    arrays = dict(r_m=e.r_m, v_m_s=e.v_m_s, flux_per_s=e.flux_per_s,
                  reference_plane_z_m=e.reference_plane_z_m,
                  detection_weights=d["detection_weights"], phase_rad=result["phase_rad"],
                  per_atom_P2=result["per_atom_P2"], source_trace=result["source_trace"])
    for name in ("ground_density", "exit_populations", "exit_time_s", "quantization_axes", "population_trace", "reference_time_s"):
        if getattr(e, name) is not None:
            arrays[name] = getattr(e, name)
    if result["per_atom_ground_populations"] is not None:
        arrays["raman_ground_populations"] = result["per_atom_ground_populations"]
    np.savez_compressed(directory / "ensemble.npz", **arrays)
    if result.get("pumping") is not None:
        pumping = result["pumping"]
        save_stateful_ensemble(result["source_ensemble"], directory / "mot_ensemble.h5")
        (directory / "pumping_summary.json").write_text(
            json.dumps(pumping["diagnostics"], indent=2), encoding="utf-8")
        np.savez_compressed(directory / "pumping.npz", **{key: value for key, value in pumping.items()
            if key not in ("ensemble", "diagnostics")})
    if e.ground_density is not None:
        save_stateful_ensemble(e, directory / "source_ensemble.h5")
        ground = np.real(np.diagonal(e.ground_density, axis1=1, axis2=2))
        with (directory / "source_states.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["atom"] + ["F%d_m%+d" % (f, m) for f, m in GROUND_LABELS])
            writer.writerows((i,) + tuple(row) for i, row in enumerate(ground))
    weights = d["detection_weights"]
    effective_n = weights.sum() ** 2 / np.sum(weights ** 2)
    temperatures = MASS * np.var(e.v_m_s, axis=0) / KB
    mean_vz = float(e.v_m_s[:, 2].mean())
    offsets = c.raman.phase_offsets_rad
    reference_phase = float(np.angle(np.exp(1j * (np.pi + offsets[0] - 2 * offsets[1] + offsets[2]))))
    summary = {
        "source": e.source_metadata,
        "exit_flux_per_s": e.flux_per_s,
        "detected_flux_per_s": d["detected_flux_per_s"],
        "expected_atoms_per_gate": d["expected_atoms_per_gate"],
        "source_exit_mean_vz_m_s": mean_vz,
        "source_exit_sigma_vz_m_s": float(e.v_m_s[:, 2].std()),
        "source_exit_temperature_xyz_K": temperatures.tolist(),
        "nominal_T_s": c.raman.zone_spacing_m / c.raman.reference_vz_m_s,
        "mean_atom_T_s": float(np.mean(c.raman.zone_spacing_m / e.v_m_s[:, 2])),
        "effective_detected_MC_atoms": float(effective_n),
        "fit": result["fit"], "expected_fit": result["expected_fit"],
        "max_state_norm_error": result["max_state_norm_error"],
        "state_handling": result["state_handling"],
        "ensemble_reference_plane_z_m": e.reference_plane_z_m,
        "mirror_displacement_per_2pi_m": 2 * np.pi / K_EFF,
        "reference_Raman_frequency_difference_Hz": (HYPERFINE_HZ
            + HBAR * K_EFF ** 2 / (4 * np.pi * MASS)
            + K_EFF * c.raman.k_direction[2] * c.raman.reference_vz_m_s / (2 * np.pi)
            + c.raman.two_photon_detuning_Hz),
        "contrast_definition": "hypot(cosine_coefficient, sine_coefficient) / positive_offset; dimensionless, separate from signed phase_rad",
        "phase_convention": "P2=offset+amplitude*cos(scan_phase+phase_rad); amplitude>=0; phase_rad in [-pi, pi]; reference includes configured phase offsets",
        "ideal_short_pulse_reference_phase_rad": reference_phase,
        "scope": ["effective cycling two-level Doppler MOT or empirical exit sampler",
                  "constant longitudinal velocity; constant initial-velocity transverse Coriolis force",
                  "finite Gaussian or rectangular Raman envelopes, two internal/momentum modes",
                  "configured removal or background of unprepared atoms; optional Gaussian velocity selection",
                  "stationary two-channel fluorescence; ideal F1 repump and state separation"],
        "not_modelled": ["thermal-vapor loading and absolute source flux prediction",
                         "multilevel dark states, polarization-gradient cooling, coherent sideband interference",
                         "separate wave-packet branches, imperfect closure, spontaneous Raman scattering",
                         "detector pumping dynamics, state cross-talk, bandwidth and feedback loops"],
    }
    if e.source_metadata["mode"] == "pylcp":
        summary["scope"][0] = "native 24-state PyLCP rate equations, linear Zeeman, adiabatic local quantization axis"
        summary["not_modelled"][1] = "optical coherences, coherent dark states, sub-Doppler cooling and nonadiabatic spin transport"
    if "optical_pumping" in e.source_metadata:
        summary["optical_pumping"] = e.source_metadata["optical_pumping"]
        summary["scope"].append("finite-flight 24-level population optical pumping, force and optional recoil diffusion")
    if "raman_power_calibration" in result:
        summary["raman_power_calibration"] = result["raman_power_calibration"]
        summary["raman_spin_transport"] = result["raman_spin_transport"]
        summary["scope"].append("Actual per-color Raman powers determine the coupling and differential light shift through coherent D2 pathways")
    if e.ground_density is not None:
        population = np.real(np.diagonal(e.ground_density, axis1=1, axis2=2))
        summary["ground_basis_F_mF"] = GROUND_LABELS.tolist()
        summary["exit_ground_population_mean_lab_z"] = population.mean(axis=0).tolist()
        summary["exit_clock_pair_fraction_mean"] = float(population[:, [G_CLOCK, E_CLOCK]].sum(axis=1).mean())
        summary["state_checkpoint"] = "source_ensemble.h5"
        if e.exit_populations is not None:
            summary["excited_fraction_at_exit_mean"] = float(e.exit_populations[:, 8:].sum(axis=1).mean())
        if result["per_atom_ground_populations"] is not None:
            summary["max_raman_ground_population_norm_error"] = float(np.max(np.abs(result["per_atom_ground_populations"].sum(axis=2) - 1)))
    plot_path = directory / "overview.png"
    if plot:
        try:
            save_plot(result, plot_path)
            summary["plot"] = plot_path.name
            if len(result["source_trace"]):
                save_source_plot(result, directory / "source_cooling.png")
                summary["source_plot"] = "source_cooling.png"
            if e.ground_density is not None:
                save_state_plot(result, directory / "state_populations.png")
                summary["state_plot"] = "state_populations.png"
        except ModuleNotFoundError as error:
            if error.name != "matplotlib":
                raise
            summary["plot"] = "omitted: install matplotlib to render the overview"
    (directory / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    # Markdown is convenient for inspecting the run without opening large arrays.
    phase = result["fit"]["phase_rad"]
    phase_text = "undefined (zero contrast)" if phase is None else f"{phase:.6f} rad"
    contrast = result["fit"]["contrast"]
    contrast_text = "undefined" if contrast is None else f"{contrast:.6f}（{100 * contrast:.2f}%）"
    text = ("# 87Rb 束流仿真结果\n\n"
            f"- 原子源：{e.source_metadata['mode']}；出射样本数：{len(e.r_m)}\n"
            f"- 平均纵向速度：{mean_vz:.5f} m/s\n"
            f"- 出射横向温度：Tx={temperatures[0]*1e6:.2f}、Ty={temperatures[1]*1e6:.2f} µK\n"
            f"- 参考速度对应 T：{summary['nominal_T_s']*1e3:.5f} ms\n"
            f"- 探测有效 Monte Carlo 样本数：{effective_n:.1f}\n"
            f"- 拉曼入射态处理：{result['state_handling']}\n"
            f"- 条纹对比度：{contrast_text}\n"
            f"- 条纹拟合相位：{phase_text}（配置相位对应的理想短脉冲参考值为 {reference_phase:.6f} rad，需参考运行差分）\n"
            f"- 最大态归一化误差：{result['max_state_norm_error']:.3e}\n\n"
            "对比度 C = sqrt(B² + D²) / A，由 P₂(φ) = A + B cosφ + D sinφ 拟合得到，无量纲。\n"
            "拟合相位取 [-π, π] 内的值，可以为负；它表示条纹的水平偏移，与条纹对比度是不同物理量。\n\n"
            "MOT 模式的通量由假定的已捕获原子注入通量归一化；calibrated 模式由配置给定，来源见 summary.json。\n"
            "拟合残差给出的相位误差仅反映本次读出散布，不包含 Monte Carlo 误差或系统误差。\n"
            "本结果是约化模型演示，不能直接视为博士论文实验复现。\n")
    if plot_path.exists() and plot:
        text += "\n![仿真概览](overview.png)\n"
    if e.ground_density is not None:
        text += ("\n参考平面处的实验室 z 轴 8 维基态密度矩阵保存在 source_ensemble.h5；含光泵浦时另存参考平面位置和自 MOT 出口起算的飞行时刻。\n"
                 "入射态使用 inherit 时，Raman 仅耦合 mF=0 态对，其他磁子能级保留为探测背景。\n")
        if result.get("pumping") is not None:
            text += "\n光泵浦前 MOT 样本保存在 mot_ensemble.h5，制备演化和散射统计见 pumping.npz、pumping_summary.json。\n"
        if plot:
            text += "\n![原子态分布](state_populations.png)\n"
    (directory / "report.md").write_text(text, encoding="utf-8")
    return summary
