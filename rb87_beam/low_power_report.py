"""Paired source-ID statistics and discrete working ranges for the pump comparison."""
import csv
import functools
import hashlib
import json
from pathlib import Path

import numpy as np

from .joint_statistics import _cluster_contributions, _estimates, _integer


_METRICS = {"flux_per_s": "effective_target_flux_per_s", "photons": "mean_scattered_photons",
            "efficiency": "preparation_efficiency", "contrast": "fringe_contrast"}
_JOINT = ("baseline_flux_per_s", "candidate_flux_per_s", "photon_saving")


@functools.lru_cache(maxsize=2)
def _resamples(n, draws, seed):
    counts = np.random.default_rng(seed).multinomial(n, np.full(n, 1 / n), size=draws)
    return counts, hashlib.sha256(counts.astype("<i8", copy=False).tobytes()).hexdigest()


def _number(x):
    return float(x) if np.isfinite(x) else None


def _store(row, key, point, draws):
    row[key] = _number(point)
    valid = np.isfinite(draws).all()
    row[key + "_lo95"], row[key + "_hi95"] = (
        tuple(map(float, np.quantile(draws, [.025, .975]))) if valid else (None, None))


def paired_metrics(baseline_records, candidate_records, incident_samples, incident_flux_per_s,
                   min_flux_per_s, *, bootstrap=1000, seed=20261120, retain_draws=False):
    """Compare a conventional baseline to a candidate on the SAME independent source sample.

    Each list contains trajectory replicas in the native joint_statistics format. Missing
    source IDs contribute zero flux. Photon means are per extracted atom, including atoms
    missed at detection. The calling runner guarantees identical source provenance.
    Selection must be completed on training data before these validation estimates.
    Positive photon_saving means the candidate scatters fewer photons than the baseline.
    With retain_draws=True, pass a whole family of rows to certify_family before reporting.
    """
    n = _integer(incident_samples, "incident_samples")
    b = _integer(bootstrap, "bootstrap")
    if b < 2 or not isinstance(seed, (np.integer, int)) or isinstance(seed, bool) or seed < 0:
        raise ValueError("Need at least two resamples and a nonnegative integer seed")
    if not np.isfinite(incident_flux_per_s) or incident_flux_per_s <= 0:
        raise ValueError("incident_flux_per_s must be positive")
    if not np.isfinite(min_flux_per_s) or min_flux_per_s < 0:
        raise ValueError("min_flux_per_s must be finite and nonnegative")
    counts, digest = _resamples(n, b, seed)
    row = dict(incident_samples=n, bootstrap=b, bootstrap_seed=int(seed),
               bootstrap_sha256=digest, incident_flux_per_s=float(incident_flux_per_s),
               min_flux_per_s=float(min_flux_per_s))
    points, draws = {}, {}
    for label, records in (("baseline", baseline_records), ("candidate", candidate_records)):
        contribution = _cluster_contributions(records, n)
        # Treat the original sample as a separate point estimate; use precisely
        # the same incident-ID counts for both cases and every comparison.
        p = _estimates(np.ones((1, n)), contribution, n, incident_flux_per_s)
        d = _estimates(counts, contribution, n, incident_flux_per_s)
        for name, metric in _METRICS.items():
            key = label + "_" + name
            points[key], draws[key] = p[metric][0], d[metric]
            _store(row, key, points[key], draws[key])
    for key, metric, sign in (("photon_saving", "photons", -1),
                              ("flux_change_per_s", "flux_per_s", 1),
                              ("contrast_change", "contrast", 1)):
        before, after = "baseline_" + metric, "candidate_" + metric
        points[key] = sign * (points[after] - points[before])
        draws[key] = sign * (draws[after] - draws[before])
        _store(row, key, points[key], draws[key])
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = draws["candidate_flux_per_s"] / draws["baseline_flux_per_s"]
        point = np.divide(points["candidate_flux_per_s"], points["baseline_flux_per_s"])
    _store(row, "flux_ratio", point, ratio)
    row["matched_flux_1pct"] = (row["flux_ratio_lo95"] is not None
                                and row["flux_ratio_lo95"] >= .99 and row["flux_ratio_hi95"] <= 1.01)
    row["common_flux_point"] = all(row[k] is not None and row[k] >= min_flux_per_s
                                   for k in _JOINT[:2])
    if retain_draws:
        row["_draws"] = {k: draws[k] for k in _JOINT}
    return row


def certify_family(rows, min_flux_per_s):
    """Simultaneous approximate 95% max-standardized-bootstrap limits on a fixed grid.

    Jointly cover both fluxes and both directions of the paired scattering difference
    over all supplied nodes, retaining correlations across quantities and settings.
    Undefined bootstrap quantities never pass. Deletes the private draw arrays after use.
    The intervals are conditional on this finite source sample, recoil model and grid.
    """
    if not np.isfinite(min_flux_per_s) or min_flux_per_s < 0:
        raise ValueError("Invalid common flux floor")
    if not rows:
        return None
    maxima = np.zeros(rows[0]["bootstrap"])
    scales = []
    identity = (rows[0]["bootstrap_sha256"], rows[0]["incident_flux_per_s"])
    for row in rows:
        if (row["bootstrap_sha256"], row["incident_flux_per_s"]) != identity:
            raise ValueError("All nodes must share source normalization and paired resamples")
        if row["min_flux_per_s"] != min_flux_per_s or "_draws" not in row:
            raise ValueError("All nodes need the same floor and retained paired draws")
        errors = {}
        for k in _JOINT:
            draws, point = row["_draws"][k], row[k]
            if draws.shape != maxima.shape:
                raise ValueError("Mismatched bootstrap draw shape")
            if point is None or not np.isfinite(draws).all():
                errors[k] = None
                continue
            centered = draws - point
            tolerance = 32 * np.finfo(float).eps * max(1., abs(point))
            error = float(draws.std(ddof=1))
            if np.max(np.abs(centered)) <= tolerance:
                errors[k] = 0.  # numerically constant contrast between paired paths
            elif error <= 0:
                errors[k] = None
            else:
                errors[k] = error
                maxima = np.maximum(maxima, np.abs(centered) / error)
        scales.append(errors)
    critical = float(np.quantile(maxima, .95))
    for row, errors in zip(rows, scales):
        for k, scale in errors.items():
            value = row[k]
            row[k + "_family_lo95"] = None if scale is None else value - critical * scale
            row[k + "_family_hi95"] = None if scale is None else value + critical * scale
        row["common_flux_supported"] = all(row[k + "_family_lo95"] is not None and
            row[k + "_family_lo95"] >= min_flux_per_s for k in _JOINT[:2])
        lower, upper = row["photon_saving_family_lo95"], row["photon_saving_family_hi95"]
        row["advantage"] = bool(row["common_flux_supported"] and lower is not None and lower > 0)
        row["baseline_advantage"] = bool(row["common_flux_supported"] and upper is not None and upper < 0)
        del row["_draws"]
    return critical


def power_runs(rows, powers_uW, flag="advantage", *, group_by=("length_mm", "detuning_MHz")):
    """Group adjacent *measured nodes*, with missing or unsupported nodes breaking a run."""
    powers = sorted(map(float, powers_uW))
    if len(set(powers)) != len(powers) or any(not np.isfinite(v) for v in powers):
        raise ValueError("Power axis must consist of unique finite values")
    group = {}
    for row in rows:
        key = tuple(row[k] for k in group_by)
        power = float(row["power_uW"])
        by_power = group.setdefault(key, {})
        if power not in powers or power in by_power:
            raise ValueError("Duplicate or off-grid power; check group_by includes all fixed axes")
        by_power[power] = row[flag]
    runs = []
    for key, data in sorted(group.items()):
        start = []
        for p in powers + [None]:
            if p is not None and data.get(p, False):
                start.append(p)
            elif start:
                runs.append(dict(zip(group_by, key), powers_uW=start, power_min_uW=start[0],
                                 power_max_uW=start[-1], n_nodes=len(start)))
                start = []
    return runs


def _write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value, nd=3):
    return "—" if value is None else ("{:." + str(nd) + "f}").format(value)


def _plot(rows, powers, path):
    # A node map intentionally leaves unmeasured coordinates blank.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    detunings = sorted(set(r["detuning_MHz"] for r in rows))
    lengths = sorted(set(r["length_mm"] for r in rows))
    powers = sorted(powers)
    ncols = min(len(detunings), 3)
    nrows = (len(detunings) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, sharex=True, sharey=True,
                             figsize=(max(7.5, 3.9*ncols), 2.7*nrows + 1.7))
    styles = {"candidate": ("o", "#0072B2", "3:1 scatters less"),
              "baseline": ("^", "#D55E00", "1:1 scatters less"),
              "unresolved": ("s", "#8d8d8d", "Difference unresolved"),
              "flux": ("x", "#777777", "Common flux floor unconfirmed")}
    for axis, delta in zip(axes.flat, detunings):
        for row in rows:
            if row["detuning_MHz"] != delta:
                continue
            label = ("candidate" if row["advantage"] else "baseline" if row["baseline_advantage"]
                     else "unresolved" if row["common_flux_supported"] else "flux")
            marker, color, _ = styles[label]
            x, y = powers.index(row["power_uW"]), lengths.index(row["length_mm"])
            axis.scatter(x, y, s=50, marker=marker, color=color, zorder=3)
            if row["matched_flux_1pct"] and row["common_flux_supported"]:
                axis.scatter(x, y, s=115, marker="o", edgecolors="#222", facecolors="none", linewidths=.75)
        axis.set_title("Common detuning: {:g} MHz".format(delta), fontsize=10)
        axis.set_xticks(range(len(powers))); axis.set_xticklabels(["{:g}".format(p) for p in powers], rotation=30)
        axis.set_yticks(range(len(lengths))); axis.set_yticklabels(["{:g}".format(v) for v in lengths])
        axis.set_xlim(-.55, len(powers)-.45); axis.set_ylim(-.55, len(lengths)-.45)
        axis.grid(alpha=.15)
        axis.set_xlabel("Total pump power (µW)"); axis.set_ylabel("Length (mm)")
    for axis in list(axes.flat)[len(detunings):]:
        axis.set_visible(False)
    handles = [Line2D([], [], marker=m, color=c, ls="", label=label) for m, c, label in styles.values()]
    handles.append(Line2D([], [], marker="o", color="#222", markerfacecolor="none", ls="",
                   label="Flux ratio CI within ±1%"))
    fig.legend(handles=handles, loc="lower center", ncol=2 if ncols < 2 else 3, fontsize=8)
    fig.tight_layout(rect=[0, min(.20, .60/(nrows+1)), 1, 1])
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_report(outdir, rows, powers_uW, metadata):
    """Export certified validation rows; preserve unresolved and contrary results.

    Required metadata: baseline (three grid coordinates), incident_samples, replicas,
    incident_flux_per_s, min_flux_per_s. Optional: selection_notes (plain Chinese text).
    Axes power_uW, length_mm, detuning_MHz in rows refer to the 3:1 candidate.
    Selection must not be repeated after examining this output.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (r["detuning_MHz"], r["length_mm"], r["power_uW"]))
    for r in rows:
        if "_draws" in r or "common_flux_supported" not in r:
            raise ValueError("Run certify_family before writing the report")
    public = [dict(r, near_flux_advantage=bool(r["advantage"] and r["matched_flux_1pct"])) for r in rows]
    axes = ("power_uW", "length_mm", "detuning_MHz")
    fields = list(axes) + sorted(set().union(*(r.keys() for r in public)) - set(axes))
    _write_csv(out / "validation.csv", public, fields)
    ranges = {label: power_runs(public, powers_uW, flag) for label, flag in
              (("candidate_advantage", "advantage"), ("baseline_advantage", "baseline_advantage"),
               ("near_flux_advantage", "near_flux_advantage"))}
    table = [dict(run, category=label, powers_uW=json.dumps(run["powers_uW"])) for
             label, pieces in ranges.items() for run in pieces]
    _write_csv(out / "power_ranges.csv", table,
               ("category", "length_mm", "detuning_MHz", "powers_uW", "n_nodes",
                "power_min_uW", "power_max_uW"))
    if public:
        _plot(public, powers_uW, out / "comparison.png")
    elif (out / "comparison.png").exists():
        (out / "comparison.png").unlink()
    base = metadata["baseline"]
    n_good = sum(r["advantage"] for r in public)
    n_reverse = sum(r["baseline_advantage"] for r in public)
    n_both = sum(r["common_flux_supported"] for r in public)
    n_eq = sum(r["near_flux_advantage"] for r in public)
    lines = ["# 常规低功率基线：独立源配对验证", "",
             "常规方案固定转移光与暗态制备光功率比为 1:1，对照方案为 3:1。"
             "两者使用同一注入通量、相同 Raman 和探测条件。功率均为两束制备光的总功率。", "",
             "训练冻结的 1:1 基线：总功率 **{:g} µW**，作用长度 **{:g} mm**，共同失谐 **{:+g} MHz**。"
             .format(base["power_uW"], base["length_mm"], base["detuning_MHz"]),
             "有效目标态通量门槛为 **{:.4g} 原子/s**，入射通量为 {:.4g} 原子/s；"
             "使用 {} 个独立入射源编号，每编号 {} 条反冲轨迹。"
             .format(metadata["min_flux_per_s"], metadata["incident_flux_per_s"],
                     metadata["incident_samples"], metadata["replicas"]), ""]
    if metadata.get("selection_notes"):
        lines.extend([metadata["selection_notes"], ""])
    lines += ["独立验证 {} 个 3:1 节点。其中 {} 个节点与基线均有证据满足通量门槛；"
              "在这个集合内，**{} 个节点是 3:1 散射更低，{} 个是 1:1 散射更低，"
              "{} 个未分辨出散射差异**。其余 {} 个的共同门槛未被确认。"
              .format(len(public), n_both, n_good, n_reverse, n_both-n_good-n_reverse, len(public)-n_both),
              "在 3:1 散射更低的节点中，{} 个的通量比（3:1 / 1:1）逐点配对 95% 区间还完全处于 "
              "[0.99, 1.01]。两侧都过门槛不等于实际通量相等。".format(n_eq), ""]
    if public:
        b = public[0]
        lines += ["1:1 基线验证值：有效通量 {:.5g} 原子/s，散射 {} 光子/出射原子，"
                  "荧光条纹对比度 {}。".format(b["baseline_flux_per_s"], _fmt(b["baseline_photons"]),
                                           _fmt(b["baseline_contrast"], 4)), ""]
        # Include an a-priori comparison: the candidate with the baseline coordinates,
        # and the two previously published-in-model (80 µW, 4 mm) signed detunings.
        # Never pick the lowest-scattering validation node and call it a new optimum.
        def coord(row):
            return tuple(row[k] for k in axes)
        keyset = {coord(base), (80., 4., -1.5), (80., 4., 1.5)}
        examples = public if len(public) <= 12 else [r for r in public if coord(r) in keyset]
        if examples:
            lines += ["所有验证点的指标如下：" if len(public) <= 12 else
                      "以下列出已验证的代表点：与 1:1 基线相同的功率、长度、失谐，以及原有 80 µW、"
                      "4 mm、±1.5 MHz 设置。其余点详见逐点数据。", "",
                      "| 3:1 总功率 (µW) | 长度 (mm) | 失谐 (MHz) | 有效通量 (10⁹ 原子/s) | "
                      "光子/出射原子 | 条纹对比度 | Δ对比度 [逐点95%] |",
                      "|---:|---:|---:|---:|---:|---:|---:|"]
            for r in examples:
                f = r["candidate_flux_per_s"]
                lines += ["| {:g} | {:g} | {:+g} | {} | {} | {} | {} [{}, {}] |".format(
                    *coord(r), _fmt(f / 1e9 if f is not None else None, 5),
                    _fmt(r["candidate_photons"]), _fmt(r["candidate_contrast"], 4),
                    _fmt(r["contrast_change"], 4), _fmt(r["contrast_change_lo95"], 4),
                    _fmt(r["contrast_change_hi95"], 4))]
            lines.append("")
        lines += ["**3:1 低散射的已验证功率节点**（固定长度与失谐）：", ""]
        if ranges["candidate_advantage"]:
            lines += ["| 作用长度 (mm) | 共同失谐 (MHz) | 逐点验证的总功率 (µW) |", "|---:|---:|---|"]
            for run in ranges["candidate_advantage"]:
                lines += ["| {:g} | {:+g} | {} |".format(run["length_mm"], run["detuning_MHz"],
                          "、".join("{:g}".format(p) for p in run["powers_uW"]))]
        else:
            lines += ["本次没有达到判据的 3:1 节点；不能据此推断所有未测参数都没有优势。"]
        lines += ["", "图中每个符号均是独立源验证过的节点；空白表示未验证。", "",
                  "![独立源配对验证](comparison.png)", ""]
    lines += ["[逐点原始指标、配对区间及判据](validation.csv) · "
              "[两种方向及近等通量的功率节点组](power_ranges.csv)", "",
              "统计口径：以入射源编号配对重采样，先在每个编号内平均反冲轨迹。"
              "有效通量计入目标态概率和探测接受度；散射是每个出射原子在制备区平均自发散射的光子数；"
              "对比度是荧光相位条纹的一次谐波振幅除以直流项。"
              "散射节省量 = 基线 − 3:1，正值有利于 3:1。联合判据要求两侧通量的同时下界均不低于门槛，"
              "且散射差的同时区间排除零。该区间由配对重采样中跨全部验证点与上述三个量的"
              "最大标准化偏差估计（近似 95%，条件于仿真模型、已冻结的基线和有限反冲采样）。"
              "通量比与对比度变化另给逐点 95% 区间，不作为散射优势判据。", "",
              "功率段只合并声明功率轴上相邻且通过的节点；不跨空白或失败点，"
              "不对两点之间插值。结论比较 3:1 设置与一个经过训练选择的固定 1:1 基线，"
              "不表示 3:1 在相同参数处必胜、在所有传统方案中占优，或已经过实验验证。"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ranges
