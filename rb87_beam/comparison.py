"""Paired source-checkpoint comparison of preparation, throughput and fringes."""

import copy
import csv
import json
from pathlib import Path

import numpy as np

from .io import export_result, save_stateful_ensemble
from .physics import run_simulation, KB, MASS
from .states import G_CLOCK, GROUND_LABELS


LABELS = {
    "unprepared": "MOT 原始混态",
    "transfer_only": "仅 F2→F′1 转移光",
    "two_frequency": "双频 σ± 光泵浦",
    "two_frequency_no_diffusion": "双频，关闭反冲扩散",
    "pi_leak_1pct": "双频，1% π 偏振泄漏",
    "project_before": "原始混态 + 理想目标态筛选",
    "project_after": "双频光泵浦 + 理想目标态筛选",
}


def metrics(name, result):
    e, d = result["ensemble"], result["detection"]
    pop = np.real(np.diagonal(e.ground_density, axis1=1, axis2=2))
    # The selected run already includes the target fraction in its detection weight.
    target_weights = d["detection_weights"] * (1 if result["state_handling"] == "project_F1_m0" else pop[:, G_CLOCK])
    norm = e.flux_per_s / len(pop)
    return {
        "case": name,
        "preparation_efficiency": float(pop[:, G_CLOCK].mean()),
        "F2_fraction_before_Raman": float(pop[:, GROUND_LABELS[:, 0] == 2].sum(axis=1).mean()),
        "target_flux_before_selection_per_s": float(e.flux_per_s * pop[:, G_CLOCK].mean()),
        "detected_total_flux_per_s": d["detected_flux_per_s"],
        "effective_target_flux_per_s": float(norm * target_weights.sum()),
        "fringe_contrast": result["expected_fit"]["contrast"],
        "fringe_amplitude": result["expected_fit"]["amplitude"],
        "transverse_temperature_K": float(MASS * np.var(e.v_m_s[:, :2], axis=0).mean() / KB),
        "mean_scattered_photons": e.source_metadata.get("optical_pumping", {}).get("mean_scattered_photons", 0.0),
        "monte_carlo_atoms": len(pop),
    }


def compare_preparation(config, ensemble, directory, plot=True, progress=None):
    """Use identical input atoms. Disable readout noise and ideal eta losses."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if ensemble.ground_density is None:
        raise ValueError("Preparation comparison requires a stateful MOT ensemble")
    if ensemble.reference_plane_z_m != 0 or "optical_pumping" in ensemble.source_metadata:
        raise ValueError("Preparation comparison requires an unprepared MOT exit checkpoint")
    save_stateful_ensemble(ensemble, directory / "input_mot_ensemble.h5")
    base = copy.deepcopy(config)
    base.pumping.enabled = False
    base.raman.state_handling = "inherit"
    base.raman.preparation_efficiency = 1.0
    base.detection.noise = False
    base.save(directory / "comparison_config.json")
    results, rows = {}, []
    for name in LABELS:
        cfg = copy.deepcopy(base)
        atoms = ensemble
        if name in ("transfer_only", "two_frequency", "two_frequency_no_diffusion", "pi_leak_1pct"):
            cfg.pumping.enabled = True
            if name == "transfer_only":
                cfg.pumping.dark_power_W = 0
            if name == "pi_leak_1pct":
                cfg.pumping.pi_power_fraction = .01
            if name == "two_frequency_no_diffusion":
                cfg.pumping.recoil_model = "none"
        if name.startswith("project_"):
            cfg.raman.state_handling = "project_F1_m0"
            if name == "project_after":
                atoms = results["two_frequency"]["ensemble"]
        if progress:
            print("Preparation case:", name, flush=True)
        result = run_simulation(cfg, atoms, progress)
        export_result(result, directory / name, plot=plot)
        results[name] = result
        rows.append(metrics(name, result))
    with (directory / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (directory / "comparison.json").write_text(json.dumps({
        "cases": rows,
        "definitions": {
            "preparation_efficiency": "Unconditional mean F1,m0 population after free decay, before any selection",
            "effective_target_flux_per_s": "Input-normalized F1,m0 flux surviving configured velocity selection and detector aperture, before Raman redistribution",
            "detected_total_flux_per_s": "All internal states surviving selection and aperture",
            "fringe_contrast": "Amplitude / offset of expected photon-weighted F2/(F1+F2) fringe",
            "projection": "Ideal removal of all other states, unit transmission for target; not a simulated physical cleaner",
        },
        "noise": False,
        "velocity_selection_sigma_m_s": config.raman.selection_sigma_vx_m_s,
        "source_flux_note": "Inherited source normalization; not a vapor-loading/absolute-flux prediction",
    }, indent=2), encoding="utf-8")
    lines = ["# 多能级光泵浦与干涉比较", "",
        "相同 MOT 出射样本、相同拉曼及探测参数；关闭读取噪声。未按论文效率拟合参数。", "",
        "| 方案 | F1,m0 人口 | 探测总通量 /s | 有效目标态通量 /s | 条纹对比度 | 条纹振幅 | 平均散射光子 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append("| %s | %.3f%% | %.5g | %.5g | %.3f%% | %.5f | %.2f |" % (
            LABELS[row["case"]], 100 * row["preparation_efficiency"], row["detected_total_flux_per_s"],
            row["effective_target_flux_per_s"], 100 * row["fringe_contrast"], row["fringe_amplitude"], row["mean_scattered_photons"]))
    lines += ["", "效率为筛选前无条件目标态人口。有效通量为通过速度选择和探测孔径的初始 F1,m0 原子通量，尚未计入拉曼转移概率。",
        "理想筛选只用于给出对照，不表示已实现物理清除脉冲。筛选前后效率栏相同，损失已计入探测总通量。",
        "F1 旁观态会降低绝对条纹振幅，但在 F2 背景足够小的情况下不一定降低此处定义的相对对比度，因此应同时看目标态通量和振幅。", "",
        "关闭反冲扩散的对照仍保留平均光学力，可用于辨认随机加热的影响；它不是无光或关闭全部运动效应。",
        "模型使用完整 D2 跃迁强度、自发分支、塞曼失谐、非共振泄漏、有限飞行时间、平均散射力和可选反冲扩散。",
        "采用偏置场下的布居速率方程，初始基态相干作世俗平均；不含相干暗态、激光相干干涉、非绝热自旋输运和多波包拉曼路径。",
        "原子通量继承源的归一化；少量蒙特卡洛样本仅供功能和趋势验证，不能替代收敛统计或实验标定。",
        "论文 §2.2.2、§4.1.4 给出双频 σ 跃迁方案、4 mW 总功率及 12.5 mm 直径；功率平分、光区位置、反向双束几何和理想偏振为本程序假设。",
        "论文实测 84.84% 来自拉曼谱峰统计；此处直接计算全部八个基态的人口，不将两种口径等同。", "",
        "`input_mot_ensemble.h5` 可重放全部对照；各方案目录保留配置、条纹、八态分布和带飞行时刻的出口检查点。"]
    (directory / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if plot:
        save_comparison_plot(results, rows, directory / "comparison.png")
    return rows


def save_comparison_plot(results, rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#4c6273", "#d68a2b", "#21867a", "#78a845", "#a64c60", "#7295ba", "#4b5db3"]
    display = {"two_frequency_no_diffusion": "two_freq_no_diffusion"}
    names = [r["case"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    x = np.arange(len(rows))
    axes[0, 0].bar(x, [100 * r["preparation_efficiency"] for r in rows], color=colors)
    axes[0, 0].set(ylabel="F=1,m=0 population (%)", title="Preparation before selection")
    axes[0, 1].bar(x - .18, [r["detected_total_flux_per_s"] / 1e9 for r in rows], .36, label="Total", color="#b4c1ca")
    axes[0, 1].bar(x + .18, [r["effective_target_flux_per_s"] / 1e9 for r in rows], .36, label="Useful target", color=colors)
    axes[0, 1].set(ylabel="Flux (10^9 atoms/s)", title="Detector aperture and velocity acceptance")
    axes[0, 1].legend()
    for ax in axes[0]:
        ax.set_xticks(x); ax.set_xticklabels([display.get(n, n) for n in names], rotation=30, ha="right", fontsize=8)
    for name, color in zip(names, colors):
        result = results[name]
        axes[1, 0].plot(result["phase_rad"], result["detection"]["expected_photon_P2"], label=display.get(name, name), color=color)
    axes[1, 0].set(xlabel="Third Raman phase (rad)", ylabel="F2 / (F1+F2)", title="Same finite three-pulse interferometer")
    axes[1, 0].legend(fontsize=8)
    for name, color in zip(names[:5], colors[:5]):
        pump = results[name].get("pumping")
        if pump is not None:
            axes[1, 1].plot(pump["z_trace_m"] * 1e3, pump["population_trace"][:, G_CLOCK], label=display.get(name, name), color=color)
    axes[1, 1].set(xlabel="z from MOT exit (mm)", ylabel="F=1,m=0 population", title="Population evolves through the pumping light")
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle("87Rb 24-level optical pumping: paired source sample")
    fig.savefig(path, dpi=160); plt.close(fig)
