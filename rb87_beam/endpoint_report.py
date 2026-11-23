"""Chinese report and scientific exports for the controlled endpoint comparison.

The report displays validation estimates, not the maxima of training scans.
It deliberately distinguishes identical resource bounds from identical realized
flux/scattering, and sampled feasible nodes from a continuous work interval.
"""

import csv
import html
import json
from pathlib import Path

import numpy as np

from .joint_report import _json_value, _peak_rabi_Hz


LABELS = {
    "joint": "末端荧光联合优化",
    "sequential_efficiency": "分阶段：先提高制备效率",
    "sequential_scattering": "分阶段：先降低散射",
}
ENGLISH_LABELS = {
    "joint": "Joint endpoint optimization",
    "sequential_efficiency": "Sequential: preparation first",
    "sequential_scattering": "Sequential: scattering first",
}
PHYSICAL_METRICS = (
    ("preparation_efficiency", "目标态制备效率", "%", 100.),
    ("effective_target_flux_per_s", "有效目标态通量", "原子/s", 1.),
    ("detected_flux_per_s", "探测总通量", "原子/s", 1.),
    ("mean_scattered_photons", "光泵浦平均散射", "光子/原子", 1.),
    ("fringe_contrast", "条纹对比度", "%", 100.),
    ("fringe_amplitude", "归一化条纹振幅", "1", 1.),
)
ENDPOINT_METRICS = (
    ("readout_snr_per_rad", "末端读出信噪斜率", "rad⁻¹ / 探测门", 1.),
    ("phase_noise_rad", "局部相位噪声", "mrad / 探测门", 1000.),
    ("F1_counts", "F1 通道期望计数", "计数 / 探测门", 1.),
    ("F2_counts", "F2 通道期望计数", "计数 / 探测门", 1.),
)


def _finite(value):
    return value is not None and np.ndim(value) == 0 and bool(np.isfinite(value))


def _format(value, scale=1.):
    return "%.6g" % (float(value) * scale) if _finite(value) else "未定义"


def _estimate(row, key, scale=1.):
    result = _format(row.get(key), scale)
    ci = row.get(key + "_95ci")
    if ci is not None and len(ci) == 2 and all(_finite(x) for x in ci):
        result += " [%s, %s]" % (_format(ci[0], scale), _format(ci[1], scale))
    return result


def _table(headers, rows):
    head = "".join("<th>%s</th>" % html.escape(str(x)) for x in headers)
    body = "".join("<tr>" + "".join("<td>%s</td>" % html.escape(str(x))
                                          for x in row) + "</tr>" for row in rows)
    return ("<div class='table'><table><thead><tr>%s</tr></thead>"
            "<tbody>%s</tbody></table></div>") % (head, body)


def _csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _names(study):
    methods = study["strategies"]
    return [name for name in LABELS if name in methods] + sorted(set(methods) - set(LABELS))


def _parameters(study):
    names = _names(study)
    reference = study.get("baseline_config", study["strategies"][names[0]]["config"])
    power_ref = _peak_rabi_Hz(reference) * reference["raman"]["waist_z_m"]
    rows = []
    for name in names:
        entry = study["strategies"][name]
        p, r = entry["config"]["pumping"], entry["config"]["raman"]
        peak = _peak_rabi_Hz(entry["config"])
        values = (
            ("pump_total_uW", "泵浦总功率", "µW", 1e6 * (p["transfer_power_W"] + p["dark_power_W"])),
            ("transfer_power_uW", "F2 转移光功率", "µW", p["transfer_power_W"] * 1e6),
            ("dark_power_uW", "F1 暗态制备光功率", "µW", p["dark_power_W"] * 1e6),
            ("transfer_detuning_MHz", "F2 转移光失谐", "MHz", p["transfer_detuning_Hz"] / 1e6),
            ("dark_detuning_MHz", "F1 暗态制备光失谐", "MHz", p["dark_detuning_Hz"] / 1e6),
            ("pump_length_mm", "泵浦截断光区全长", "mm", 2 * p["half_widths"] * p["waist_z_m"] * 1e3),
            ("raman_waist_um", "Raman z 向光强束腰", "µm", r["waist_z_m"] * 1e6),
            ("raman_peak_kHz", "第一/三光区峰值 Ω/(2π)", "kHz", peak / 1e3),
            ("raman_relative_power", "相对 Raman 光功率代理", "1", entry.get("parameters", {}).get(
                "raman_relative_power", peak * r["waist_z_m"] / power_ref)),
            ("raman_detuning_kHz", "Raman 双光子失谐", "kHz", r["two_photon_detuning_Hz"] / 1e3),
            ("chirp_Hz_s", "固定啁啾率", "Hz/s", r["chirp_rad_s2"] / (2 * np.pi)),
            ("readout_phase_rad", "冻结读出工作相位", "rad", entry["endpoint"].get("operating_phase_rad")),
        )
        for key, label, unit, value in values:
            rows.append(dict(strategy=name, family=entry.get("family", ""),
                             parameter=key, label=label, unit=unit, value=value))
    return rows


def _scenario_definitions(study):
    settings = study.get("settings", {})
    return settings.get("validation_scenarios", settings.get("scenarios", []))


def _feasible(row, constraints):
    tests = (
        ("preparation_efficiency", "min_efficiency", 1),
        ("effective_target_flux_per_s", "min_effective_target_flux_per_s", 1),
        ("mean_scattered_photons", "max_scattered_photons", -1),
        ("fringe_contrast", "min_fringe_contrast", 1),
    )
    return all(_finite(row.get(key)) and sign * (row[key] - constraints[bound]) >= 0
               for key, bound, sign in tests if bound in constraints)


def _comparison(study):
    settings = study.get("settings", {})
    constraints = settings.get("constraints", {})
    definitions = {row["name"]: row for row in _scenario_definitions(study)}
    rows = []
    for name in _names(study):
        entry = study["strategies"][name]
        for scenario, physical in entry["validation"]["scenarios"].items():
            definition = definitions.get(scenario, {})
            row = dict(strategy=name, scenario=scenario,
                       source_vz_factor=definition.get("source_vz_factor", 1.),
                       source_divergence_factor=definition.get("source_divergence_factor", 1.),
                       point_metric_feasible=_feasible(physical, constraints))
            endpoint = entry["endpoint"]["scenarios"].get(scenario, {})
            for block, metrics in ((physical, PHYSICAL_METRICS), (endpoint, ENDPOINT_METRICS)):
                for metric, _, _, _ in metrics:
                    row[metric] = block.get(metric)
                    ci = block.get(metric + "_95ci")
                    row[metric + "_95ci_low"] = ci[0] if ci is not None else None
                    row[metric + "_95ci_high"] = ci[1] if ci is not None else None
            rows.append(row)
    return rows


def _plot(study, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = _names(study)
    definitions = _scenario_definitions(study)
    xs = sorted({float(s.get("source_vz_factor", 1.)) for s in definitions})
    ys = sorted({float(s.get("source_divergence_factor", 1.)) for s in definitions})
    if not xs or not ys:
        raise ValueError("Report needs explicit validation speed/divergence scenarios")
    constraints = study.get("settings", {}).get("constraints", {})
    arrays = []
    for name in names:
        entry = study["strategies"][name]
        contrast = np.full((len(ys), len(xs)), np.nan)
        snr = contrast.copy()
        feasible = np.zeros(contrast.shape, bool)
        for definition in definitions:
            scene = definition["name"]
            if scene not in entry["validation"]["scenarios"]:
                continue
            ix = xs.index(float(definition.get("source_vz_factor", 1.)))
            iy = ys.index(float(definition.get("source_divergence_factor", 1.)))
            physical = entry["validation"]["scenarios"][scene]
            endpoint = entry["endpoint"]["scenarios"].get(scene, {})
            c, s = physical.get("fringe_contrast"), endpoint.get("readout_snr_per_rad")
            contrast[iy, ix] = 100 * c if _finite(c) else np.nan
            snr[iy, ix] = s if _finite(s) else np.nan
            feasible[iy, ix] = _feasible(physical, constraints)
        arrays.append((contrast, snr, feasible))
    fig, axes = plt.subplots(len(names), 2, figsize=(10.4, 3.15 * len(names)), squeeze=False,
                             constrained_layout=True)
    for column in (0, 1):
        finite = np.concatenate([a[column][np.isfinite(a[column])] for a in arrays])
        low, high = (float(np.min(finite)), float(np.max(finite))) if len(finite) else (0., 1.)
        if low == high:
            high = low + max(1., abs(low) * .01)
        for row, name in enumerate(names):
            ax = axes[row, column]
            values, feasible = arrays[row][column], arrays[row][2]
            im = ax.imshow(values, origin="lower", aspect="auto", vmin=low, vmax=high,
                           cmap="YlGnBu")
            for iy, ix in np.ndindex(values.shape):
                if np.isfinite(values[iy, ix]):
                    text = "%.2f" % values[iy, ix] if column == 0 else "%.0f" % values[iy, ix]
                    if not feasible[iy, ix]:
                        text += "\nFAIL"
                    color = "white" if values[iy, ix] > low + .6 * (high - low) else "#132b43"
                    ax.text(ix, iy, text, ha="center", va="center", color=color, fontsize=11)
            ax.set_xticks(range(len(xs)))
            ax.set_xticklabels(["%.2f" % x for x in xs])
            ax.set_yticks(range(len(ys)))
            ax.set_yticklabels(["%.2f" % y for y in ys])
            ax.set(xlabel="Beam-speed factor", ylabel="Angular-divergence factor",
                   title=ENGLISH_LABELS.get(name, name))
        fig.colorbar(im, ax=axes[:, column].tolist(), shrink=.8,
                     label="Fringe contrast (%)" if column == 0 else "Readout SNR slope (rad$^{-1}$ per gate)")
    fig.suptitle("87Rb: held-out source validation on a speed / angular-divergence grid\n"
                 "FAIL = at least one point-estimate metric constraint failed", fontsize=13)
    fig.savefig(output / "speed_divergence.png", dpi=170)
    plt.close(fig)


def _paired_rows(node, prefix=""):
    rows = []
    for key, value in node.items():
        if isinstance(value, dict):
            rows.extend(_paired_rows(value, prefix + LABELS.get(key, key) + " / "))
        elif key.endswith("_95ci"):
            base = key[:-5]
            if value is None or len(value) != 2:
                rows.append((prefix + base, _format(node.get(base)), "未定义", "区间不可用"))
            else:
                lower_better = "phase_noise" in base
                support = ("区间支持改善" if lower_better else "区间支持下降") if value[1] < 0 else (
                    ("区间支持恶化" if lower_better else "区间支持提高") if value[0] > 0 else "区间包含零")
                rows.append((prefix + base, _format(node.get(base)),
                             "[%s, %s]" % (_format(value[0]), _format(value[1])), support))
    return rows


def _plot_local(study, output):
    """Show every frozen local node without interpolating a feasible box."""
    region = study.get("local_work_region", [])
    if not region:
        return False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    keys = ("raman_relative_power", "raman_waist_um", "raman_detuning_kHz")
    xs, ys, ds = [sorted({round(float(r["parameters"][key]), 10) for r in region}) for key in keys]
    reference = study["strategies"]["joint"]["endpoint"]["robust"]["worst_readout_snr_per_rad"]
    if not _finite(reference) or reference <= 0:
        raise ValueError("Local-region plot requires a positive frozen joint reference SNR")
    slices = []
    for detuning in ds:
        values = np.full((len(ys), len(xs)), np.nan)
        feasible, retained = np.zeros(values.shape, bool), np.zeros(values.shape, bool)
        for row in region:
            p = row["parameters"]
            if round(float(p["raman_detuning_kHz"]), 10) != detuning:
                continue
            ix = xs.index(round(float(p["raman_relative_power"]), 10))
            iy = ys.index(round(float(p["raman_waist_um"]), 10))
            if np.isfinite(values[iy, ix]):
                raise ValueError("Duplicate local parameter node")
            value = row["endpoint"]["robust"].get("worst_readout_snr_per_rad")
            values[iy, ix] = 100 * value / reference if _finite(value) else np.nan
            feasible[iy, ix] = bool(row["passes_all_grid_constraints"])
            retained[iy, ix] = feasible[iy, ix] and bool(row["within_95percent_selected_validation_snr"])
        slices.append((values, feasible, retained))
    finite = np.concatenate([data[0][np.isfinite(data[0])] for data in slices])
    low, high = float(np.min(finite)), max(100., float(np.max(finite)))
    if low == high:
        low -= 1
    fig, axes = plt.subplots(1, len(ds), figsize=(4.1 * len(ds) + .65, 4.6),
                             constrained_layout=True, squeeze=False)
    for ax, detuning, (values, feasible, retained) in zip(axes[0], ds, slices):
        im = ax.imshow(values, origin="lower", aspect="auto", cmap="YlGnBu", vmin=low, vmax=high)
        for iy, ix in np.ndindex(values.shape):
            value = values[iy, ix]
            if not np.isfinite(value):
                continue
            text_color = "white" if value > low + .6 * (high - low) else "#132b43"
            ax.text(ix, iy, "%.1f%%" % value, ha="center", va="center", fontsize=12, color=text_color)
            if not feasible[iy, ix]:
                ax.plot(ix + .32, iy + .32, marker="x", markersize=10,
                        markeredgewidth=2.4, color="#c91f2d", linestyle="none")
            if retained[iy, ix]:
                ax.add_patch(Rectangle((ix - .46, iy - .46), .92, .92, fill=False,
                                       linewidth=2, edgecolor="#20bfa3"))
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(["%.2f" % x for x in xs])
        ax.set_yticks(range(len(ys)))
        ax.set_yticklabels(["%g" % y for y in ys])
        ax.set(xlabel="Relative Raman power proxy", ylabel="Raman z waist (micrometers)",
               title="Two-photon detuning = %g kHz" % detuning)
    fig.colorbar(im, ax=axes[0].tolist(), shrink=.88,
                 label="Worst-grid SNR / frozen joint SNR (%)")
    fig.suptitle("Frozen local tolerance nodes: 27 combinations, one fixed training readout phase\n"
                 "Red X: constraint failed  |  Green outline: feasible and SNR >= 95%  |  No interpolation", fontsize=12)
    fig.savefig(output / "local_region.png", dpi=180)
    plt.close(fig)
    return True


def _decision(study):
    if study.get("decision", {}).get("note"):
        return study["decision"]["note"]
    strategies = study["strategies"]
    if "joint" not in strategies:
        return "请结合配对区间、约束是否通过及两种分阶段规则判断。"
    joint = strategies["joint"]
    parts = []
    for method in ("sequential_efficiency", "sequential_scattering"):
        if method not in strategies:
            continue
        same = joint["config"] == strategies[method]["config"]
        parts.append("与%s相比，联合优化选择了%s参数。" %
                     (LABELS[method], "相同" if same else "不同"))
        comparison = study.get("paired", {}).get(method, {})
        interval = comparison.get("worst_readout_snr_per_rad_difference_95ci")
        if interval is None:
            parts.append("最差末端信噪的配对区间不可用。")
        elif interval[0] > 0:
            parts.append("独立源配对区间支持最差末端信噪提高。")
        elif interval[1] < 0:
            parts.append("独立源配对区间显示最差末端信噪下降。")
        elif not same:
            parts.append("最差末端信噪的配对区间包含零，尚不能确认改善。")
    feasible = bool(joint["validation"]["robust"].get("feasible"))
    parts.append("联合方案%s全部声明速度/散度网格的点估计约束；这不构成连续区间保证。" %
                 ("通过" if feasible else "未通过"))
    return " ".join(parts)


def write_report(study, output_dir):
    """Export a fully evaluated controlled study; the caller owns study.json."""
    study = _json_value(study)
    names = _names(study)
    for name in names:
        for key in ("config", "validation", "endpoint"):
            if key not in study["strategies"][name]:
                raise ValueError("Missing strategy report input: %s.%s" % (name, key))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    parameters, comparison = _parameters(study), _comparison(study)
    _csv(output / "parameters.csv", parameters)
    _csv(output / "comparison.csv", comparison)
    _plot(study, output)
    has_local_plot = _plot_local(study, output)
    headers = [LABELS.get(name, name) for name in names]
    nominal = []
    for kind, metrics in (("validation", PHYSICAL_METRICS), ("endpoint", ENDPOINT_METRICS)):
        for key, label, unit, scale in metrics:
            nominal.append([label, unit] + [_estimate(study["strategies"][name][kind]["scenarios"]["nominal"],
                                                      key, scale) for name in names])
    robust = []
    for kind, key, label, scale in (
        ("validation", "worst_contrast", "最差条纹对比度 / %", 100.),
        ("validation", "min_efficiency", "最低制备效率 / %", 100.),
        ("validation", "min_effective_target_flux_per_s", "最低有效通量 /s", 1.),
        ("validation", "max_scattered_photons", "最高泵浦平均散射", 1.),
        ("endpoint", "worst_readout_snr_per_rad", "最差末端读出信噪斜率 / rad⁻¹", 1.),
        ("endpoint", "worst_phase_noise_rad", "最差局部相位噪声 / mrad", 1000.),
    ):
        robust.append([label] + [_estimate(study["strategies"][name][kind]["robust"], key, scale)
                                 for name in names])
    robust.append(["所有声明网格点约束通过"] + [str(study["strategies"][name]["validation"]["robust"].get("feasible"))
                                             for name in names])
    robust.append(["条件重采样中全部指标约束通过的比例 / %"] +
                  [_format(study["strategies"][name]["validation"]["robust"].get(
                      "bootstrap_constraint_feasible_fraction"), 100.) for name in names])
    rows_by_parameter = {}
    for row in parameters:
        rows_by_parameter.setdefault(row["parameter"], {})[row["strategy"]] = row
    param_rows = [[next(iter(items.values()))["label"], next(iter(items.values()))["unit"]] +
                  [_format(items[name]["value"]) for name in names]
                  for items in rows_by_parameter.values()]
    note = _decision(study)
    settings = study.get("settings", {})
    p = ["<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>",
         "<meta name='viewport' content='width=device-width,initial-scale=1'>",
         "<title>Rb87：末端荧光与分阶段调参的受控比较</title>",
         "<style>body{font-family:'Microsoft YaHei',system-ui,sans-serif;color:#172b3b;background:#fbfcfd;max-width:1300px;margin:36px auto;padding:0 22px;line-height:1.7}h1{font-size:29px;line-height:1.35}h2{font-size:21px;margin-top:34px}p{max-width:1120px}.table{overflow-x:auto;margin:15px 0}table{width:100%;border-collapse:collapse;font-size:13px;background:white}th,td{padding:9px 11px;border-bottom:1px solid #d9e1e8;text-align:left;vertical-align:top}th{background:#eaf1f5}td:not(:first-child){font-variant-numeric:tabular-nums}img{max-width:100%;height:auto;background:white}.decision{padding:16px 20px;border-left:4px solid #287899;background:#edf5f8}.scope{padding:14px 20px;background:#f2f5f8}.eyebrow{color:#48728a;font-size:13px;letter-spacing:.1em}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f0f3f6;padding:15px;font-size:12px}a{color:#17648d}.caption{font-size:13px;color:#4f6373}</style></head><body>",
         "<div class='eyebrow'>87Rb · 光泵浦—三脉冲 Raman—末端荧光</div>",
         "<h1>末端信号是否会选出不同、且可跨束流条件工作的参数？</h1>",
         "<p class='decision'>%s</p>" % html.escape(str(note)),
         "<p>本次比较共用有限泵浦候选库、Raman 候选库及物理约束。联合方法直接选择末端荧光读出表现；两种分阶段方法分别先提高制备效率、或在效率与通量达标后先降低散射，再在各自冻结的泵浦组内优化 Raman。分阶段的第二步同样获得完整的公共 Raman 候选库。</p>",
         "<h2>指标与比较口径</h2>",
         "<p>末端目标为归一化双通道荧光信号的局部相位斜率除以读出噪声。计算包含原子到达的泊松涨落、每原子荧光光子涨落、背景与电子噪声；局部相位噪声为该指标的倒数。它是当前探测模型下的线性噪声传播估计，不是严格 Fisher 信息，也不是实测噪声。读出工作相位由训练冻结，验证时不因束速或散度变化重新调参。</p>",
         "<p>相同通量下限与相同散射上限表示相同资源预算，并不表示两方案实际通量或散射恰好相同。下表并列实际值；若提供共同有效通量对照，该对照仅按统一比例降低原子到达率，固定探测背景保持不变。</p>",
         "<h2>冻结后参数</h2>", _table(["参数", "单位"] + headers, param_rows),
         "<p class='caption'>Raman 中间光区峰值为第一/三光区的两倍。Ωwz 仅为匹配光束条件下的相对光功率代理，尚未标定为 mW；失谐与啁啾率分别列出。</p>",
         "<h2>独立源验证：标称场景</h2>", _table(["指标", "单位"] + headers, nominal),
         "<h2>跨束速、角散度网格的最差表现</h2>", _table(["指标"] + headers, robust),
         "<p class='caption'>方括号为按独立源编号重采样的 95% 条件统计区间。每个源编号内部先平均反冲重复；反冲次数与网格节点数不增加独立源样本量。条件重采样的约束通过比例反映本样本下的统计裕量，不是实验成功率或连续区域的可行概率。</p>",
         "<img src='speed_divergence.png' alt='三个调参方法在速度和角散度网格上的条纹对比度和末端荧光读出信噪'>",
         "<p class='caption'>每一列使用共同色标。FAIL 表示该节点至少一项指标约束未通过。单点与有限网格可行不等于整个连续区间已获得保证。</p>",
         "<h2>联合方法相对分阶段方法的配对变化</h2>",
         _table(["比较 / 指标", "点估计变化", "95% 配对区间", "区间含义"], _paired_rows(study.get("paired", {}))),
    ]
    matched = []
    for comparison_method, entries in study.get("matched_flux", {}).items():
        for name, result in entries.items():
            scenarios = list(result.get("scenarios", {}).values())
            rates = [row["arrival_rate_scale"] for row in scenarios]
            fluxes = [row["matched_effective_flux_per_s"] for row in scenarios]
            matched.append((LABELS.get(comparison_method, comparison_method), LABELS.get(name, name),
                            "%s–%s" % (_format(min(rates)), _format(max(rates))) if rates else "未定义",
                            "%s–%s" % (_format(min(fluxes)), _format(max(fluxes))) if fluxes else "未定义",
                            _format(result.get("worst_readout_snr_per_rad"))))
    if matched:
        p.extend(["<h2>共同有效通量对照</h2>",
                  _table(("配对基线", "方法", "各场景到达率缩放", "各场景共同有效通量 /s", "最差信噪斜率 / rad⁻¹"), matched),
                  "<p class='caption'>每个场景内将两方法均匀稀释到同一个有效通量点估计，再保持该缩放进行配对重采样；不按速度或态筛原子。不同场景的共同目标值可以不同。上方 matched_flux_worst_snr_difference 的区间给出该对照的配对变化；此操作不使每原子的泵浦散射相等。</p>"])
    if "local_work_region" in study or "work_region" in study:
        local = "local_work_region" in study
        region = study["local_work_region"] if local else study["work_region"]
        region_rows = []
        for row in region:
            parameters = row.get("parameters", {})
            region_rows.append((row.get("case"), row.get("family", study["strategies"].get("joint", {}).get("family", "")),
                _format(parameters.get("raman_waist_um")), _format(parameters.get("raman_relative_power")),
                _format(parameters.get("raman_detuning_kHz")),
                _estimate(row.get("validation", {}).get("robust", {}), "worst_contrast", 100),
                _estimate(row.get("endpoint", {}).get("robust", {}), "worst_readout_snr_per_rad"),
                str(row.get("passes_all_grid_constraints")), str(row.get("within_95percent_selected_validation_snr"))))
        valid = sum(bool(row.get("passes_all_grid_constraints")) for row in region)
        stable = sum(bool(row.get("passes_all_grid_constraints")) and
                     bool(row.get("within_95percent_selected_validation_snr")) for row in region)
        p.extend(["<h2>%s</h2>" % ("冻结的局部容差网格" if local else "工作点邻域检查"),
            "<p>共检查 %d 个参数组合节点，其中 %d 个通过全部束速/散度场景的点估计约束；%d 个同时保留冻结联合工作点至少95%%的最差读出信噪。</p>" %
            (len(region), valid, stable)])
        if local and has_local_plot:
            p.extend(["<img src='local_region.png' alt='三个Raman失谐切片上的局部容差节点，红叉显示约束失败，绿色边框标出通过约束并保留至少95%信噪的节点'>",
                "<p class='caption'>数字与共同色标表示各节点在九个束流场景中的最差信噪，相对冻结联合工作点的百分比。红叉为约束失败；绿色边框表示约束通过且信噪保留至少95%。图中直接保留失败节点，不对离散点插值，也不将参数最小值与最大值之间的整个区域视为可行。</p>"])
        p.extend([
            _table(("冻结邻域点", "泵浦组", "Raman 束腰 /µm", "功率代理", "失谐 /kHz", "最差 C /%",
                    "最差信噪斜率 /rad⁻¹", "网格约束通过", "≥联合冻结点的95%信噪"), region_rows),
            "<p class='caption'>邻域点在验证前冻结，验证结果仅用于检查这些点能否保留可行性与信噪。这里的95%是性能比例，不是95%置信水平；通过的离散节点不能直接拼成无空洞的连续参数区间。</p>"])
        if local:
            p.append("<p class='caption'>局部网格固定联合训练方案的读出工作相位，未按各验证节点重新调相，也不利用验证结果回选新的推荐方案。原粗网格记录保留在完整 study.json 的 work_region 字段中。</p>")
            protocol = study.get("local_work_region_grid", study.get("local_region_protocol"))
            if protocol:
                p.append("<details><summary>查看局部网格冻结协议</summary><pre>%s</pre></details>" %
                         html.escape(json.dumps(protocol, ensure_ascii=False, indent=2)))
            p.append("<p class='caption'>完整冻结协议：<a href='local_region_protocol.json'>local_region_protocol.json</a>。</p>")
    if study.get("convergence"):
        convergence_rows = []
        for count, methods in study["convergence"].items():
            for name in names:
                row = methods[name]
                convergence_rows.append((count, LABELS.get(name, name),
                    _estimate(row["validation"]["scenarios"]["nominal"], "fringe_contrast", 100),
                    _estimate(row["validation"]["robust"], "worst_contrast", 100),
                    _estimate(row["endpoint"]["robust"], "worst_readout_snr_per_rad"),
                    str(row["validation"]["robust"].get("feasible"))))
        p.extend(["<h2>增加独立源样本数的检查</h2>",
                  _table(("源原子数", "方法", "标称 C /%", "最差 C /%", "最差信噪斜率 /rad⁻¹", "网格约束通过"), convergence_rows),
                  "<p class='caption'>小样本是大样本的嵌套前缀，采用同一完整样本定义的源扰动；两行不是互相独立的重复实验，也不能仅因区间重叠就宣称完全收敛。</p>"])
    p.extend([
        "<h2>共同约束与场景定义</h2><pre>%s</pre>" % html.escape(json.dumps(settings, ensure_ascii=False, indent=2)),
        "<h2>源样本划分</h2><pre>%s</pre>" % html.escape(json.dumps(study.get("source_split", {}), ensure_ascii=False, indent=2)),
        "<div class='scope'><p><b>结果适用范围。</b>入射通量 10¹⁰ 原子/s 为当前模型假设，未由蒸气装载率或实验通量标定获得。散射预算只包含光泵浦散射，当前有效 Raman 模型未计算 Raman 自发散射；因此不能把它解释为总散射约束。</p>",
        "<p>速度和散度变化施加于 MOT 出口的已有分布：对中心化的横向角度 v⊥/vz 缩放，再与纵向速度独立组合，并从该出口重新进行光泵浦与 Raman 传播。这是对源分布变化的鲁棒性检查，不是改变 MOT 实验设置后重新生成原子束。</p>",
        "<p>验证源与本轮训练源独立，但该源池在先前研究中已使用，不能称为全新盲测。置信区间不含模型误差、实验系统误差或未标定参数的误差。有限的两个泵浦家族和 Raman 网格不代表连续参数空间的全局最优；有限样本也可能无法分辨很小的改善。</p>",
        "<p>上游保留 24 能级人口与反冲运动；下游采用有效 Raman 钟态对并保留旁观态人口。探测器背景与电子噪声来自配置假设。本报告不将模拟的末端期望信号表述为实验闭环测量。</p></div>",
        "<p>结果文件：<a href='study.json'>完整 study.json</a> · <a href='parameters.csv'>参数 CSV</a> · <a href='comparison.csv'>各场景指标 CSV</a> · <a href='speed_divergence.png'>束流场景图</a>%s</p>" %
        (" · <a href='local_region.png'>局部容差节点图</a>" if has_local_plot else ""),
        "</body></html>",
    ])
    (output / "report.html").write_text("\n".join(p), encoding="utf-8")
    paths = {key: str((output / filename).resolve()) for key, filename in (
        ("report", "report.html"), ("comparison", "comparison.csv"),
        ("parameters", "parameters.csv"), ("plot", "speed_divergence.png"))}
    if has_local_plot:
        paths["local_plot"] = str((output / "local_region.png").resolve())
    return paths
