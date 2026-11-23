"""Traceable CSV, HTML and scientific figures for joint robust optimization."""

import csv
import html
import json
from pathlib import Path

import numpy as np


METRICS = (
    ("preparation_efficiency", "目标态制备效率", "%", 100.0),
    ("exit_flux_per_s", "出射总通量", "原子/s", 1.0),
    ("detected_flux_per_s", "探测总通量", "原子/s", 1.0),
    ("effective_target_flux_per_s", "探测孔径内目标态通量", "原子/s", 1.0),
    ("fringe_contrast", "条纹对比度 C", "%", 100.0),
    ("fringe_amplitude", "归一化 F2 条纹振幅 A", "1", 1.0),
    ("modulated_flux_per_s", "Raman 调制通量 2AΦdet", "原子/s", 1.0),
    ("mean_scattered_photons", "泵浦平均散射光子", "光子/原子", 1.0),
    ("score", "联合目标 2A√(Φdet/Φin)", "1", 1.0),
)


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _finite(value):
    return value is not None and np.ndim(value) == 0 and np.isfinite(value)


def _metric(row, name):
    if name == "fringe_amplitude":
        return row.get(name, row.get("amplitude"))
    if name == "modulated_flux_per_s":
        value = row.get(name, row.get("modulated_flux"))
        amplitude, flux = _metric(row, "fringe_amplitude"), row.get("detected_flux_per_s")
        if value is None and _finite(amplitude) and _finite(flux):
            return 2 * amplitude * flux
        return value
    return row.get(name)


def _ci(row, name):
    value = row.get(name + "_95ci")
    if value is None and name == "fringe_amplitude":
        value = row.get("amplitude_95ci")
    if value is None and name == "modulated_flux_per_s":
        value = row.get("modulated_flux_95ci")
    if value is None or len(value) != 2 or not all(_finite(item) for item in value):
        return None
    return value


def _format(value, scale=1.0):
    if not _finite(value):
        return "未定义"
    value = float(value) * scale
    return "%.5g" % value


def _estimate_text(row, name, scale=1.0):
    text = _format(_metric(row, name), scale)
    interval = _ci(row, name)
    if interval is not None:
        text += " [%s, %s]" % (_format(interval[0], scale), _format(interval[1], scale))
    return text


def _peak_rabi_Hz(config):
    r = config["raman"]
    explicit = r.get("peak_rabi_Hz", 0.0)
    if explicit and explicit > 0:
        return float(explicit)
    integrated_width = (r["waist_z_m"] * np.sqrt(np.pi / 2)
                        if r["envelope"] == "gaussian" else r["rectangular_width_m"])
    return float(r["reference_vz_m_s"] / (4 * integrated_width))


def parameter_rows(before, after):
    """Human units, while complete unmodified SI configs remain in study.json."""
    reference = _peak_rabi_Hz(before) * before["raman"]["waist_z_m"]
    rows = []

    def add(key, label, unit, old, new):
        rows.append(dict(parameter=key, label=label, unit=unit,
                         before=float(old), after=float(new), change=float(new - old)))

    for key, label in (("transfer_power_W", "F2 转移光功率"), ("dark_power_W", "F1 暗态制备光功率")):
        add(key, label, "µW", before["pumping"][key] * 1e6, after["pumping"][key] * 1e6)
    add("pump_total_power", "泵浦总功率", "µW",
        (before["pumping"]["transfer_power_W"] + before["pumping"]["dark_power_W"]) * 1e6,
        (after["pumping"]["transfer_power_W"] + after["pumping"]["dark_power_W"]) * 1e6)
    for key, label in (("transfer_detuning_Hz", "F2 转移光失谐"), ("dark_detuning_Hz", "F1 暗态制备光失谐")):
        add(key, label, "MHz", before["pumping"][key] * 1e-6, after["pumping"][key] * 1e-6)
    add("pump_interaction_length", "泵浦截断光区全长", "mm",
        2 * before["pumping"]["half_widths"] * before["pumping"]["waist_z_m"] * 1e3,
        2 * after["pumping"]["half_widths"] * after["pumping"]["waist_z_m"] * 1e3)
    add("raman_waist_z", "Raman z 向光强束腰", "µm",
        before["raman"]["waist_z_m"] * 1e6, after["raman"]["waist_z_m"] * 1e6)
    add("raman_peak_rabi", "第一/三光区峰值 Ω/(2π)", "kHz",
        _peak_rabi_Hz(before) * 1e-3, _peak_rabi_Hz(after) * 1e-3)
    add("raman_mirror_peak_rabi", "中间光区峰值 Ω/(2π)", "kHz",
        2 * _peak_rabi_Hz(before) * 1e-3, 2 * _peak_rabi_Hz(after) * 1e-3)
    add("raman_relative_power_proxy", "相对 Raman 光功率代理 Ωwz/(Ωwz)基线", "1", 1.0,
        _peak_rabi_Hz(after) * after["raman"]["waist_z_m"] / reference)
    add("two_photon_detuning", "Raman 双光子失谐", "kHz",
        before["raman"]["two_photon_detuning_Hz"] * 1e-3,
        after["raman"]["two_photon_detuning_Hz"] * 1e-3)
    return rows


def _comparison_rows(validation):
    rows = []
    for case in ("before", "after"):
        for name, scenario in validation[case]["scenarios"].items():
            row = dict(case=case, scenario=name)
            for metric, label, unit, scale in METRICS:
                row[metric] = _metric(scenario, metric)
                interval = _ci(scenario, metric)
                row[metric + "_95ci_low"] = interval[0] if interval is not None else None
                row[metric + "_95ci_high"] = interval[1] if interval is not None else None
            rows.append(row)
    return rows


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fringe_data(study, directory):
    """Regenerate from this study; a previous run's CSV is not a valid cache."""
    path = directory / "nominal_fringe.csv"
    phase = np.linspace(0, 2 * np.pi, 129)
    curves = []
    for name in ("before", "after"):
        nominal = study["validation"][name]["scenarios"].get("nominal", {})
        coefficient = [nominal.get(key) for key in ("offset", "cosine_coefficient", "sine_coefficient")]
        if not all(_finite(value) for value in coefficient):
            return None
        a, b, d = coefficient
        curves.append(a + b * np.cos(phase) + d * np.sin(phase))
    _write_csv(path, [dict(phase_rad=x, before_P2=old, after_P2=new)
                      for x, old, new in zip(phase, *curves)])
    return phase, curves[0], curves[1]


def save_joint_plot(study, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    validation = study["validation"]
    names = list(validation["before"]["scenarios"])
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    fringe = _fringe_data(study, directory)
    if fringe is not None:
        phase, before, after = fringe
        axes[0, 0].plot(phase, before, label="Before")
        axes[0, 0].plot(phase, after, label="After")
    else:
        axes[0, 0].text(.5, .5, "Nominal fringe coefficients unavailable", ha="center",
                          transform=axes[0, 0].transAxes)
    axes[0, 0].set(title="Nominal phase fringe: aggregated recoil replicas",
                   xlabel="Third-pulse phase (rad)", ylabel="Expected normalized F=2 fraction")
    index = np.arange(len(names))
    for case, shift, color, label in (("before", -.15, "#267b9c", "Before"),
                                       ("after", .15, "#cf682a", "After")):
        metrics = validation[case]["scenarios"]
        contrasts, errors, modulation = [], [[], []], []
        for name in names:
            row = metrics[name]
            value = _metric(row, "fringe_contrast")
            contrasts.append(np.nan if value is None else 100 * value)
            interval = _ci(row, "fringe_contrast")
            errors[0].append(max(0, value - interval[0]) * 100 if interval is not None and _finite(value) else 0)
            errors[1].append(max(0, interval[1] - value) * 100 if interval is not None and _finite(value) else 0)
            mod = _metric(row, "modulated_flux_per_s")
            modulation.append(np.nan if mod is None else mod * 1e-9)
        axes[0, 1].errorbar(index + shift, contrasts, yerr=errors, fmt="o", capsize=3,
                           color=color, label=label)
        axes[1, 0].plot(index, modulation, "o-", color=color, label=label)
    axes[0, 1].set(title="Scenario contrast: source-cluster 95% intervals", ylabel="Contrast (%)")
    axes[1, 0].set(title="Usable Raman modulation, including flux", ylabel="2A Phi_detected (1e9 atoms/s)")
    for ax in (axes[0, 1], axes[1, 0]):
        ax.set_xticks(index)
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    history = study.get("search_history", [])
    scores = np.array([row.get("score", np.nan) for row in history], float)
    feasible = np.array([row.get("feasible", False) for row in history], bool)
    steps = np.arange(1, len(history) + 1)
    if len(history):
        axes[1, 1].scatter(steps[feasible], scores[feasible], color="#267b9c", label="Feasible screening")
        axes[1, 1].scatter(steps[~feasible], scores[~feasible], marker="x", color="#999999", label="Constraint failed")
        best = np.maximum.accumulate(np.where(feasible & np.isfinite(scores), scores, -np.inf))
        best[~np.isfinite(best)] = np.nan
        axes[1, 1].plot(steps, best, color="#cf682a", label="Best feasible screening score")
    axes[1, 1].set(title="Bayesian screening history (not held-out evidence)",
                   xlabel="Evaluation", ylabel="Worst-scenario 2A sqrt(Phi_det/Phi_in)")
    for ax in axes.flat:
        ax.grid(alpha=.2)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)
    fig.suptitle("87Rb joint state-preparation / Raman robust optimization")
    fig.savefig(directory / "optimization.png", dpi=160)
    plt.close(fig)


def _table(headers, rows):
    head = "".join("<th>%s</th>" % html.escape(str(value)) for value in headers)
    body = "".join("<tr>" + "".join("<td>%s</td>" % html.escape(str(value)) for value in row) + "</tr>"
                   for row in rows)
    return "<div class='table'><table><thead><tr>%s</tr></thead><tbody>%s</tbody></table></div>" % (head, body)


def _method_html(study):
    settings = study.get("settings", {})
    screening = study.get("search_history", [])
    native = study.get("native_search_history", [])
    proposals = [row for row in native if "family" in row]
    families = sorted({str(row["family"]) for row in proposals})
    rows = [
        ("低保真联合候选", str(len(screening)), str(settings.get("training_atoms", "未声明")),
         "24态人口沿给定弹道，暂不反馈平均光力与随机反冲；含结构化初始点和GP建议"),
        ("原生态制备复核", "去重泵浦组" + ("：" + str(len(families)) if families else ""),
         str(settings.get("native_training_atoms", settings.get("training_atoms", "未声明"))),
         "实际反冲/运动轨迹，每个源原子%s次反冲重复" % settings.get("training_replicas", "未声明")),
    ]
    if native:
        rows.append(("原生反馈Raman校正", "%d个新增评价；%d个已有初始评价" % (len(proposals), len(native) - len(proposals)),
                     str(settings.get("native_training_atoms", "未声明")),
                     "复用各泵浦组的真实prepared轨迹，搜索Raman束腰、相对光功率代理和双光子失谐；物理约束保持不变"))
    rows.append(("独立源验证", "冻结后两方案配对比较", str(settings.get("validation_atoms", "未声明")),
                 "每个源原子%s次反冲重复；不使用验证结果重新选择参数" % settings.get("recoil_replicas", "未声明")))
    parts = ["<h2>多保真联合搜索与原生反馈</h2>",
             _table(("阶段", "评价记录", "独立源原子数", "计算口径"), rows),
             "<p>低保真候选排名用于缩小搜索范围，不能代替包含反冲与运动反馈的原生响应。原生反馈阶段在固定泵浦组上重新计算Raman响应，从而修正仅凭低保真联合候选可能遗漏的Raman参数；原生训练的评价记录也不属于独立验证证据。</p>"]
    if native:
        cache_path = "native_training/native_search_R%s_history.json" % settings.get("training_replicas", "unknown")
        parts += ["<p>声明的原生新增评价预算：%s；实际记录新增评价：%d。原生反冲重复与低保真训练、独立验证的样本数分别列出，不能把本次计算概括成只评价了低保真候选。</p>" %
                  (html.escape(str(settings.get("native_evaluations", "未声明"))), len(proposals)),
                  "<p>原生搜索记录：<a href='%s'>%s</a>；完整记录也保存在study.json的native_search_history中。</p>" %
                  (html.escape(cache_path, quote=True), html.escape(cache_path)),
                  _table(("新增候选", "泵浦组", "最差场景训练目标 [95%区间]", "训练约束可行"),
                         [(row.get("case", ""), row.get("family", ""),
                           _estimate_text(row.get("metrics", {}).get("robust", {}), "worst_score"),
                           row.get("feasible", False)) for row in proposals])]
    parts += ["<p>约束贝叶斯搜索的方法依据：<a href='https://proceedings.mlr.press/v32/gardner14.html'>Gardner等（2014），Bayesian Optimization with Inequality Constraints</a>。本实现采用高斯过程代理和约束期望改进；代理可行概率是搜索近似，不是经实测噪声校准的实验保证。</p>"]
    return "\n".join(parts)


def _save_html(study, directory, parameters, comparison, has_plot):
    validation, settings = study["validation"], study.get("settings", {})
    decision = study.get("decision", {})
    supported = bool(decision.get("supported_improvement", False))
    note = decision.get("note", "独立样本的配对区间支持联合目标提高。" if supported else
                        "当前验证尚不足以确认联合目标提高；请结合物理约束与配对区间判断。")
    nominal_rows = []
    for metric, label, unit, scale in METRICS:
        before = validation["before"]["scenarios"].get("nominal", {})
        after = validation["after"]["scenarios"].get("nominal", {})
        nominal_rows.append((label, unit, _estimate_text(before, metric, scale), _estimate_text(after, metric, scale)))
    robust_rows = []
    for key, label, scale in (("worst_score", "最差场景联合目标", 1),
                              ("worst_contrast", "最差场景对比度 / %", 100),
                              ("min_efficiency", "最低目标态效率 / %", 100),
                              ("min_effective_target_flux_per_s", "最低有效目标态通量 /s", 1),
                              ("max_scattered_photons", "最高平均散射光子", 1)):
        robust_rows.append((label, _estimate_text(validation["before"].get("robust", {}), key, scale),
                             _estimate_text(validation["after"].get("robust", {}), key, scale)))
    robust_rows.append(("所有声明场景约束通过", str(validation["before"].get("robust", {}).get("feasible")),
                        str(validation["after"].get("robust", {}).get("feasible"))))
    baseline_pump = study["baseline_config"]["pumping"]
    baseline_power = (baseline_pump["transfer_power_W"] + baseline_pump["dark_power_W"]) * 1e6
    html_parts = [
        "<!doctype html><html lang='zh-CN'><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>Rb87 态制备与束流 Raman 联合鲁棒优化</title>",
        "<style>body{font-family:'Microsoft YaHei',system-ui,sans-serif;max-width:1180px;margin:32px auto;padding:0 20px;color:#162333;line-height:1.65}h1{font-size:26px}h2{margin-top:30px;font-size:21px}.table{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px 12px;border-bottom:1px solid #d7dfe7;text-align:left}th{background:#eef3f8}pre{background:#f4f6f8;padding:14px;white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}img{width:100%;height:auto}.decision{border-left:4px solid #267b9c;padding:8px 16px;background:#eef6f9}.scope{background:#fff8e8;padding:12px 18px}a{color:#175d8e}</style>",
        "<h1>Rb87 态制备与束流 Raman 联合鲁棒优化</h1>",
        "<p class='decision'>%s</p>" % html.escape(str(note)),
        "<p>基线为本次输入配置，泵浦总功率%s µW；本报告比较的是在此基础上的联合搜索。默认输入为已有的蓝失谐80 µW配置，不能与早期4 mW泵浦比较混为同一组实验。候选：<code>%s</code>。</p>" % (_format(baseline_power), html.escape(str(study.get("selected_candidate", "未给出")))),
        "<p>联合目标为声明扰动场景中的最小 <b>2A√(Φdet/Φin)</b>，A是归一化F2条纹振幅。报告同时保留对比度、振幅、目标态通量和散射统计，避免仅用高对比度奖励很少原子参与的窄接受度方案。</p>",
        _method_html(study),
        "<h2>优化前后参数</h2>",
        _table(("参数", "单位", "优化前", "优化后", "变化"),
               [(row["label"], row["unit"], _format(row["before"]), _format(row["after"]), _format(row["change"])) for row in parameters]),
        "<p>Raman峰值指第一、第三光区的Ω/(2π)，中间光区为其2倍。旧配置峰值为0时按参考速度和空间光区面积计算基线等效峰值。Ωwz仅为匹配光束条件下的相对光功率代理，尚未标定为真实 Raman 激光功率mW。</p>",
        "<h2>独立源样本：标称场景</h2>",
        _table(("指标", "单位", "优化前 [95%区间]", "优化后 [95%区间]"), nominal_rows),
        "<h2>所声明场景的稳健指标</h2>",
        _table(("指标", "优化前", "优化后"), robust_rows),
        "<h2>配对变化</h2><pre>%s</pre>" % html.escape(json.dumps(_json_value(validation.get("paired", {})), ensure_ascii=False, indent=2)),
    ]
    if has_plot:
        html_parts += ["<h2>条纹、场景与搜索记录</h2><img src='optimization.png' alt='优化前后条纹、各扰动场景和搜索记录'>"]
    html_parts += [
        "<h2>物理约束与扰动场景</h2><pre>%s</pre>" % html.escape(json.dumps(_json_value(settings), ensure_ascii=False, indent=2)),
        "<h2>样本划分</h2><pre>%s</pre>" % html.escape(json.dumps(_json_value(study.get("source_split", {})), ensure_ascii=False, indent=2)),
        "<div class='scope'><p>约束范围与扰动幅度是可修改的工程边界，尚不能当作设备实测上限。绝对通量继承配置的已捕获原子注入通量，本项目通常设为10¹⁰原子/s；MOT模型不预测蒸气装载导致的绝对通量。</p>",
        "<p>验证池来自已有MOT源中未参与本轮联合搜索的独立原子，不代表此前从未查看过的全新实验盲测。反冲重复先在同一源原子编号内平均，再按独立源原子作配对bootstrap；重复反冲与扰动场景不增加独立原子数。置信区间不包括模型误差、未标定参数误差和实验系统误差。</p>",
        "<p>高斯过程后验用于候选搜索，不等于上述原子统计置信区间。声明场景下的最差值不保证整个连续误差盒的全局最坏情形，有限搜索预算也不能证明全局最优。</p>",
        "<p>上游采用24能级人口率方程与反冲运动；Raman仍为有效钟态对传播并保留旁观态人口。几何预算包含最低Rayleigh长度约束，但尚未显式传播束腰随位置的变化、Gouy相位和波前曲率；几何下限不等于完整高斯光束传播模型。啁啾时钟采用共同复位参考；当前未平均自由运行连续束的异步实验室到达时刻。</p></div>",
        "<p>可复算文件：<a href='study.json'>study.json</a> · <a href='parameters.csv'>parameters.csv</a> · <a href='comparison.csv'>comparison.csv</a> · <a href='nominal_fringe.csv'>nominal_fringe.csv</a></p></html>",
    ]
    (directory / "report.html").write_text("\n".join(html_parts), encoding="utf-8")


def export_joint_report(study, directory):
    """Export final validation only; optimization diagnostics never substitute for it."""
    validation = study.get("validation", {})
    if not all(name in validation and "scenarios" in validation[name] for name in ("before", "after")):
        raise ValueError("Joint report requires before/after independent-source validation summaries")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    study = _json_value(study)
    parameters = parameter_rows(study["baseline_config"], study["selected_config"])
    comparison = _comparison_rows(study["validation"])
    _write_csv(directory / "parameters.csv", parameters)
    _write_csv(directory / "comparison.csv", comparison)
    (directory / "study.json").write_text(json.dumps(study, indent=2, ensure_ascii=False,
                                                    allow_nan=False), encoding="utf-8")
    has_plot = True
    try:
        save_joint_plot(study, directory)
    except ModuleNotFoundError as error:
        if error.name != "matplotlib":
            raise
        has_plot = False
    _save_html(study, directory, parameters, comparison, has_plot)
    return dict(report=str((directory / "report.html").resolve()),
                plot=str((directory / "optimization.png").resolve()) if has_plot else None,
                comparison=str((directory / "comparison.csv").resolve()),
                parameters=str((directory / "parameters.csv").resolve()))
