"""Export optimization evidence and uncertainty without implying experimental calibration."""

import csv
import json
from pathlib import Path

import numpy as np


def write_report(report, baseline, optimized, directory, cases):
    directory = Path(directory)
    rows = report["cases"]
    with (directory / "convergence.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    scan = json.loads((directory / "screen.json").read_text(encoding="utf-8"))
    with (directory / "screen.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["case", "total_power_uW", "transfer_power_uW", "dark_power_uW",
                  "transfer_detuning_MHz", "dark_detuning_MHz", "length_mm",
                  "preparation_efficiency", "mean_scattered_photons"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in scan["cases"]:
            p = row["pumping"]
            writer.writerow(dict(case=row["case"], total_power_uW=1e6 * (p["transfer_power_W"] + p["dark_power_W"]),
                transfer_power_uW=1e6 * p["transfer_power_W"], dark_power_uW=1e6 * p["dark_power_W"],
                transfer_detuning_MHz=p["transfer_detuning_Hz"] / 1e6, dark_detuning_MHz=p["dark_detuning_Hz"] / 1e6,
                length_mm=2e3 * p["half_widths"] * p["waist_z_m"],
                preparation_efficiency=row["preparation_efficiency"], mean_scattered_photons=row["mean_scattered_photons"]))
    maximum = report["source_trajectories"]
    final = {row["case"]: row for row in rows if row["incident_samples"] == maximum}
    paired = report["paired_differences"][-1]
    efficiency_ok = final["optimized"]["preparation_efficiency_95ci"][0] >= .95
    flux_ok = paired["useful_flux_ratio_95ci"][0] >= .95
    heating_ok = paired["photon_ratio_95ci"][1] < .5
    contrast_resolved = paired["contrast_improvement_95ci"][0] > 0
    final_ci = final["optimized"]["fringe_contrast_95ci"]
    practical_precision = (final_ci[1] - final_ci[0]) / 2 <= .03
    previous = [row for row in rows if row["case"] == "optimized" and row["incident_samples"] < maximum]
    last_change = (final["optimized"]["fringe_contrast"] - previous[-1]["fringe_contrast"]) if previous else None
    decision = dict(efficiency_lower_bound_at_least_95pct=efficiency_ok,
        useful_flux_retention_lower_bound_at_least_95pct=flux_ok,
        photon_ratio_upper_bound_below_half=heating_ok,
        paired_contrast_improvement_lower_bound_positive=contrast_resolved,
        contrast_95ci_halfwidth_at_most_3_percentage_points=practical_precision,
        contrast_change_from_previous_prefix=last_change)
    (directory / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    lines = ["# 光泵浦优化与原子样本收敛检查", "",
        "参数只用原有 32 原子训练样本选择；新的原生 MOT 轨迹用于独立验证。未按任何实验制备效率调参。", "",
        "## 参数", "", "| 参数 | 原方案 | 优化候选 |", "|---|---:|---:|"]
    for label, key, scale in (("F2→F′1 功率 /µW", "transfer_power_W", 1e6),
            ("F1→F′0 功率 /µW", "dark_power_W", 1e6),
            ("F2 转移光失谐 /MHz", "transfer_detuning_Hz", 1e-6),
            ("F1 暗态光失谐 /MHz", "dark_detuning_Hz", 1e-6)):
        lines.append("| %s | %.4g | %.4g |" % (label, getattr(baseline.pumping, key) * scale,
                                                         getattr(optimized.pumping, key) * scale))
    lines += ["| 截断光区全长 /mm | %.4g | %.4g |" % (
        2e3 * baseline.pumping.half_widths * baseline.pumping.waist_z_m,
        2e3 * optimized.pumping.half_widths * optimized.pumping.waist_z_m), "",
        "固定条件：光区中心 14 mm、y 向强度束腰 6.25 mm、偏置场 2.05 G、理想 σ± 偏振；失谐相对零场超精细跃迁。",
        "作用长度指中心 ±2 个 z 束腰的全长；改变长度同时改变峰值强度，程序按固定总功率重新计算，未假定长度与功率完全独立。", "",
        "## 最大样本结果", "",
        "新模拟 %d 条独立 MOT 入射轨迹，%d 个原子出射；每个出射原子使用 %d 组反冲随机数。" % (
            maximum, report["actual_extracted"], report["recoil_replicas"]), "",
        "| 方案 | 目标态人口 | 有效目标态通量 /s | 条纹对比度（95%区间） | 平均散射光子 |",
        "|---|---:|---:|---:|---:|"]
    for name, label in (("baseline", "原 4 mW 方案"), ("optimized", "优化候选")):
        row = final[name]
        ci = row["fringe_contrast_95ci"]
        lines.append("| %s | %.3f%% | %.5g | %.3f%% [%.3f%%, %.3f%%] | %.3f |" % (
            label, 100 * row["preparation_efficiency"], row["effective_target_flux_per_s"],
            100 * row["fringe_contrast"], 100 * ci[0], 100 * ci[1], row["mean_scattered_photons"]))
    lines += ["", "配对对比度变化：%.3f 个百分点，95%% 区间 [%.3f, %.3f] 个百分点。" % (
        100 * paired["contrast_improvement"], 100 * paired["contrast_improvement_95ci"][0],
        100 * paired["contrast_improvement_95ci"][1]),
        "目标态有效通量保留比例 %.5f，平均散射光子比例 %.5f。" % (paired["useful_flux_ratio"], paired["photon_ratio"]), "",
        ("独立样本的配对区间支持对比度提高。" if contrast_resolved else
         "配对区间跨过零：当前统计量尚不能确认对比度提高，不能把训练样本上的上升视为已验证改进。"),
        ("目标态人口、有效通量保留及散射降低均满足预设约束。" if efficiency_ok and flux_ok and heating_ok else
         "至少一项预设效率/通量/散射约束尚未由独立样本的区间确认，详见 decision.json。"), "",
        "## 样本数检查", "",
        "各级为同一独立样本池的前缀，彼此相关。区间以入射源原子编号为重采样单元；反冲重复在每个编号内平均，没有把重复数乘入独立原子数。",
        "源中未出射的轨迹保留为零通量；样本增加时仍使用原先的原子随机流。", "",
        "| 入射轨迹数 | 原方案 C | 优化候选 C | 候选 C 的95%区间 |", "|---:|---:|---:|---:|"]
    sizes = sorted({row["incident_samples"] for row in rows})
    for n in sizes:
        rr = {row["case"]: row for row in rows if row["incident_samples"] == n}
        ci = rr["optimized"]["fringe_contrast_95ci"]
        lines.append("| %d | %.3f%% | %.3f%% | [%.3f%%, %.3f%%] |" % (
            n, 100 * rr["baseline"]["fringe_contrast"], 100 * rr["optimized"]["fringe_contrast"], 100 * ci[0], 100 * ci[1]))
    if last_change is not None:
        lines += ["", "最大两级样本的候选对比度变化为 %.3f 个百分点。" % (100 * last_change)]
    lines += [("最终区间半宽已小于 3 个百分点。" if practical_precision else
               "最终区间半宽仍超过 3 个百分点：结果已有大样本误差范围，尚不宜宣称对比度达到该精度的统计收敛。"), "",
        "## 方法和边界", "",
        "1. 588 组网格扫描联合改变总功率、两频功率比、两路失谐及光区长度；全 24 态人口沿给定弹道演化作预筛选。",
        "2. 七个失谐组合各取低散射可行解，使用原生运动求解与反冲重算；约束目标态人口≥95%、有效通量≥原方案95%、散射≤原方案50%，再按训练集对比度下界选择。",
        "3. 独立验证保留全部 24 态、平均光力、重力、三点反冲积分及 25 µs 分裂步长；Raman 使用 192 时间片、41 相位点。固定场系数缓存与原生 PyLCP 生成矩阵/光力逐点对照，未消去激发态。",
        "4. 有效目标态通量指通过速度选择与探测孔径的 Raman 入射前 F1,m0 通量。绝对值沿用假定的已捕获原子源入射通量，不是热蒸气加载预测。",
        "5. 布居速率方程不描述相干暗态和非绝热自旋输运；反冲扩散需结合低光子数对照和步长检查判断。平均散射光子可衡量反冲负担，但光学摩擦也会影响净速度分布。",
        "6. 网格与训练集给出该搜索范围内的候选，不能称为全局最优。原方案的大样本对比度与旧 32 原子单次反冲读数不同，应以同池配对对照评价。", "",
        "![优化与收敛](optimization.png)", "",
        "原始数据：screen.csv/json、training_validation.json、convergence.csv/json、decision.json；可复用配置：optimized_validation_config.json；完整源和各组制备后检查点分别在 source/、held_out/。"]
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_plot(report, scan, cases, directory / "optimization.png")
    if (directory / "numerical_controls.json").exists():
        append_controls(directory)


def append_controls(directory):
    directory = Path(directory)
    control = json.loads((directory / "numerical_controls.json").read_text(encoding="utf-8"))
    path = directory / "report.md"
    text = path.read_text(encoding="utf-8").split("\n## 数值与低光子数对照")[0]
    lines = ["", "## 数值与低光子数对照", "",
        "| 对照 | 入射原子数 | 原 C | 修改后 C | 配对变化95%区间（百分点） |",
        "|---|---:|---:|---:|---:|"]
    for row in control["controls"]:
        ci = row["contrast_difference_95ci"]
        lines.append("| %s | %d | %.3f%% | %.3f%% | [%.3f, %.3f] |" % (
            row["test"], row["original"]["incident_samples"], 100 * row["original"]["fringe_contrast"],
            100 * row["modified"]["fringe_contrast"], 100 * ci[0], 100 * ci[1]))
    lines += ["", "改变反冲分裂步长会改变随机路径，此处检查分布统计量；不能把逐原子差异解释为纯积分误差。",
        "复合泊松对照将反冲改为离散吸收/受激发射计数及随机自发辐射方向，同时保留速率方程人口；它不等于内态与运动完全关联的量子跳跃求解。",
        "区间跨零表示当前精度下未分辨该对照的系统变化，不等于证明两个模型完全相同。", ""]
    path.write_text(text + "\n".join(lines), encoding="utf-8")


def write_final_report(report, directory):
    directory = Path(directory)
    total = report["source_trajectories"]
    final = {row["case"]: row for row in report["cases"] if row["incident_samples"] == total}
    comparisons = {row["case"]: row for row in report["paired_differences"] if row["incident_samples"] == total}
    labels = {"baseline": "原 4 mW 方案", "optimized": "3 次训练反冲选择的候选", "refined": "12 次训练反冲选择的候选"}
    lines = ["# 光泵浦优化：最终验证与推荐", "",
        "本研究完成588组功率/失谐/长度预筛选，并在新生成的%d条原生MOT轨迹上验证。每个源原子采用%d个反冲重复，统计以独立源原子为单位。" % (
            total, report["recoil_replicas"]), "",
        "原有32个原子负责参数选择。将训练阶段的反冲重复由3次增至12次后，候选排名发生变化，因此两组都在同一新样本池验证，分别报告，不将训练排序差异隐藏为确定的最优解。", "",
        "| 方案 | F1,m0 人口 | 有效目标态通量 /s | 对比度及95%区间 | 平均散射光子 |",
        "|---|---:|---:|---:|---:|"]
    for name in ("baseline", "optimized", "refined"):
        r = final[name]; ci = r["fringe_contrast_95ci"]
        lines.append("| %s | %.3f%% | %.5g | %.3f%% [%.3f%%, %.3f%%] | %.3f |" % (
            labels[name], 100 * r["preparation_efficiency"], r["effective_target_flux_per_s"],
            100 * r["fringe_contrast"], 100 * ci[0], 100 * ci[1], r["mean_scattered_photons"]))
    lines += ["", "| 候选相对原方案 | 对比度变化（百分点） | 两组比较校正后的95%区间 |", "|---|---:|---:|"]
    for name in ("optimized", "refined"):
        r = comparisons[name]; ci = r["contrast_improvement_family95ci"]
        lines.append("| %s | %+.3f | [%+.3f, %+.3f] |" % (labels[name], 100 * r["contrast_improvement"], 100 * ci[0], 100 * ci[1]))
    name = report["recommended_case"]
    if name:
        cfg = json.loads((directory / "recommended_config.json").read_text(encoding="utf-8"))
        p = cfg["pumping"]
        lines += ["", "推荐配置：`recommended_config.json`（%s）。" % labels[name], "",
            "- F2→F′1 / F1→F′0 功率：%.4g / %.4g µW，总功率 %.4g µW。" % (
                p["transfer_power_W"] * 1e6, p["dark_power_W"] * 1e6,
                (p["transfer_power_W"] + p["dark_power_W"]) * 1e6),
            "- 两路失谐：%+.4g / %+.4g MHz，相对零场超精细跃迁。" % (p["transfer_detuning_Hz"] / 1e6, p["dark_detuning_Hz"] / 1e6),
            "- 截断光区全长：%.4g mm；中心：%.4g mm；y 向光强束腰：%.4g mm。" % (
                2e3 * p["half_widths"] * p["waist_z_m"], 1e3 * p["center_z_m"], 1e3 * p["waist_y_m"]),
            "- 平均散射光子较原方案减少 %.2f%%，有效目标态通量保留 %.3f%%。" % (
                100 * (1 - final[name]["mean_scattered_photons"] / final["baseline"]["mean_scattered_photons"]),
                100 * final[name]["effective_target_flux_per_s"] / final["baseline"]["effective_target_flux_per_s"]), "",
            "推荐优先遵循增加训练反冲次数后的选择，前提是独立验证的效率、通量、散射约束与校正后对比度变化区间均通过；未按验证集最高对比度点估计重排候选。"]
        if name == "optimized":
            lines += ["加密训练选择的红失谐候选同样降低散射和速度展宽，但其对比度提高区间仍跨零，因此保留已通过检验的初选蓝失谐候选。"]
    else:
        lines += ["", "当前没有候选同时通过全部约束和校正后的对比度提高检查，因此不生成已确认的推荐配置。"]
    lines += ["", "## 样本数与收敛", "",
        "| 独立MOT轨迹数 | 原方案 C | 初选候选 C | 加密训练候选 C |", "|---:|---:|---:|---:|"]
    sizes = sorted({r["incident_samples"] for r in report["cases"]})
    for n in sizes:
        rr = {r["case"]: r for r in report["cases"] if r["incident_samples"] == n}
        lines.append("| %d | %.3f%% | %.3f%% | %.3f%% |" % (
            n, 100 * rr["baseline"]["fringe_contrast"], 100 * rr["optimized"]["fringe_contrast"], 100 * rr["refined"]["fringe_contrast"]))
    if name:
        ci = final[name]["fringe_contrast_95ci"]
        previous = [r for r in report["cases"] if r["case"] == name and r["incident_samples"] < total][-1]
        lines += ["", "推荐候选的最终对比度区间半宽约 %.3f 个百分点，最大两级样本的均值变化为 %+.3f 个百分点。" % (
            50 * (ci[1] - ci[0]), 100 * (final[name]["fringe_contrast"] - previous["fringe_contrast"])),
            "效率和通量已较稳定；对比度仍须按上述误差范围使用，不能把1024个样本称为亚百分点精度的严格收敛。"]
    lines += ["", "误差区间对独立入射源原子做配对簇重采样；每个原子内平均反冲重复。两组候选相对原方案的区间作 Bonferroni 同时覆盖校正。两级样本前缀相互相关，反冲重复也不算作新增独立源样本。",
        "区间只反映相空间及反冲的Monte Carlo抽样误差；这里关闭了荧光读出噪声，也没有加入独立的量子投影抽样，不能把它当作实验测量误差。", "",
        "## 使用与模型条件", "",
        "绝对通量按假定的已捕获原子入射通量1e10/s归一化；有效目标态通量是通过探测孔径与配置速度选择的Raman入射前F1,m0通量。",
        "完整24态布居速率方程、平均光力、重力、三点反冲积分及25 µs分裂步长保留；Raman为192时间片、41相位点。固定场缓存与原生PyLCP生成矩阵及光力已核对。",
        "这是约化模型中的可行方案，未声称连续参数空间的全局最优，也未将全部误差等同于上述Monte Carlo区间。相干暗态、非绝热自旋和实验光场误差仍需其他模型或测量。", "",
        "低光子数与步长对照见 [初选方案的数值对照](report.md#数值与低光子数对照) 和 `numerical_controls.json`。",
        "原始数据和配置：`training_robustness.json`、`recommended_config.json`、`refined_training/training_validation.json`；实际制备出口在 `held_out/optimized/` 和 `held_out/refined/`。", "",
        "![参数优化与样本数检查](final_convergence.png)", ""]
    if (directory / "kinetic_diagnostics.json").exists():
        kinetic = json.loads((directory / "kinetic_diagnostics.json").read_text(encoding="utf-8"))
        lines += ["## 同一 Raman 入射平面的速度温度", "",
            "| 方案 | Tx /µK | Ty /µK |", "|---|---:|---:|"]
        for key in ("source_free_flight", "baseline", "optimized", "refined"):
            t = kinetic["cases"][key]["temperature_xyz_K"]
            lines.append("| %s | %.3f | %.3f |" % (labels.get(key, "源自由飞行对照"), 1e6 * t[0], 1e6 * t[1]))
        if name:
            ci = kinetic["candidate_minus_baseline_temperature_xyz_95ci_K"][name][0]
            lines.append("\n推荐候选相对原方案的 Tx 变化95%%区间为 [%.3f, %.3f] µK。" % (1e6 * ci[0], 1e6 * ci[1]))
        lines += ["\n这里比较相同第一束 Raman 平面的无条件速度方差，包含平均光力与反冲的净作用，不把光力冷却混同为纯反冲项。", ""]
    (directory / "final_report.md").write_text("\n".join(lines), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    fields = [(axes[0, 0], "fringe_contrast", 100, "Fringe contrast (%)"),
              (axes[0, 1], "preparation_efficiency", 100, "F1,m0 population (%)"),
              (axes[1, 0], "mean_scattered_photons", 1, "Scattered photons"),
              (axes[1, 1], "effective_target_flux_per_s", 1e-9, "Useful target flux (10^9/s)")]
    for key, label, color in (("baseline", "4 mW baseline", "#a86943"),
            ("optimized", "Initial candidate", "#3b779d"), ("refined", "Refined training candidate", "#128577")):
        rr = [r for r in report["cases"] if r["case"] == key]
        for ax, field, scale, ylabel in fields:
            y = np.array([r[field] for r in rr]) * scale
            ci = np.array([r[field + "_95ci"] for r in rr]).T * scale
            ax.errorbar([r["incident_samples"] for r in rr], y,
                yerr=np.maximum([y - ci[0], ci[1] - y], 0), fmt="o-", color=color, label=label, capsize=3, ms=4)
            ax.set(xscale="log", xlabel="Independent incident source atoms", ylabel=ylabel)
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].set(yscale="log")
    fig.suptitle("87Rb pumping optimization: independent-source convergence with 95% intervals")
    fig.savefig(directory / "final_convergence.png", dpi=160); plt.close(fig)


def save_plot(report, scan, cases, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)
    data = scan["cases"]
    axes[0, 0].scatter([r["mean_scattered_photons"] for r in data],
        [100 * r["preparation_efficiency"] for r in data], s=8, alpha=.5,
        c=[4e3 * r["pumping"]["waist_z_m"] for r in data], cmap="viridis")
    axes[0, 0].axhline(97, color="black", ls=":", lw=1)
    axes[0, 0].set(xscale="log", xlabel="Mean photons (ballistic screen)", ylabel="F1,m0 (%)", title="588 power / detuning / length candidates")
    columns = [(axes[0, 1], "fringe_contrast", 100, "Fringe contrast (%)"),
               (axes[1, 0], "preparation_efficiency", 100, "F1,m0 (%)"),
               (axes[1, 1], "effective_target_flux_per_s", 1e-9, "Useful target flux (10^9/s)"),
               (axes[2, 0], "mean_scattered_photons", 1, "Mean scattered photons")]
    for name, color in (("baseline", "#ae6740"), ("optimized", "#168278")):
        rows = [r for r in report["cases"] if r["case"] == name]
        for ax, field, scale, ylabel in columns:
            x = [r["incident_samples"] for r in rows]
            y = np.array([r[field] for r in rows]) * scale
            ci = np.array([r[field + "_95ci"] for r in rows]).T * scale
            ax.errorbar(x, y, yerr=np.maximum(np.array([y - ci[0], ci[1] - y]), 0),
                        fmt="o-", capsize=3, label=name, color=color, ms=4)
            ax.set(xscale="log", xlabel="Independent incident source atoms", ylabel=ylabel)
            ax.legend()
        numerator = denominator = None
        for result, _ in cases[name]:
            e, d = result["ensemble"], result["detection"]
            weight = d["detection_weights"] / e.v_m_s[:, 2]
            nn = result["per_atom_P2"] @ weight
            numerator = nn if numerator is None else numerator + nn
            denominator = weight.sum() if denominator is None else denominator + weight.sum()
        axes[2, 1].plot(result["phase_rad"], numerator / denominator, label=name, color=color)
    axes[0, 1].set(title="Nested prefixes; source-cluster 95% intervals")
    axes[2, 1].set(xlabel="Third Raman phase (rad)", ylabel="F2/(F1+F2)", title="Pooled recoil replicas at largest N")
    axes[2, 1].legend()
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle("87Rb optical pumping: optimization and held-out convergence")
    fig.savefig(path, dpi=160); plt.close(fig)
