"""A prespecified near-equal-flux comparison independent of the search for a common floor."""
import json
import math
from pathlib import Path


_TOL = 0.003


def _finite(value):
    x = float(value)
    if not math.isfinite(x):
        raise ValueError("Matched-flux comparison needs finite inputs")
    return x


def select_matched(rows, anchor_row, floor):
    """Freeze the least-scattering 1:1 training point within ±0.3% of the 3:1 flux.

    Both training fluxes must meet floor. Exact photon ties prefer the closest
    flux, then lower power, shorter length, and lower detuning (in this order).
    No interpolation, validation-data selection, or relaxed fallback.
    """
    floor = _finite(floor)
    if floor <= 0 or anchor_row["family"] != "3:1":
        raise ValueError("Expected a positive flux floor and a 3:1 training anchor")
    target = _finite(anchor_row["flux_per_s"])
    if target < floor:
        return None
    lo, hi = (1 - _TOL) * target, (1 + _TOL) * target
    allowed = []
    for row in rows:
        if row["family"] != "1:1":
            continue
        flux, photons = _finite(row["flux_per_s"]), _finite(row["photons"])
        if photons < 0 or flux < 0:
            raise ValueError("Negative photon count or flux")
        # Compare fluxes directly; include endpoint rounding at one floating-point ulp.
        # Older Python 3.8 environments do not provide math.ulp or math.nextafter.
        if flux >= floor and (flux >= lo or math.isclose(flux, lo, rel_tol=2.22e-16)) and (
                flux <= hi or math.isclose(flux, hi, rel_tol=2.22e-16)):
            coords = tuple(_finite(row[k]) for k in ("power_uW", "length_mm", "detuning_MHz"))
            allowed.append(((photons, abs(flux / target - 1)) + coords, row))
    return dict(min(allowed, key=lambda pair: pair[0])[1]) if allowed else None


def _entry(row):
    return {k: (row[k] if k == "family" else _finite(row[k])) for k in
            ("family", "power_uW", "length_mm", "detuning_MHz", "flux_per_s", "photons")}


def _triple(row, key):
    return [None if row[k] is None else _finite(row[k])
            for k in (key, key + "_lo95", key + "_hi95")]


def _display(triple, scale=1, spec=".4f"):
    def fmt(x):
        return "未确定" if x is None else format(scale * x, spec)
    return "{} [{}, {}]".format(*(fmt(x) for x in triple))


def write_matched(outdir, choice, anchor_grid, metrics):
    """Write the training selection and separate, paired validation results.

    metrics: paired_metrics(baseline_records_1to1, anchor_records_3to1, ...)
    or None when no held-out result is available. Emits matched_flux.{json,md}.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    payload = dict(training_flux_ratio_tolerance=_TOL, anchor=_entry(anchor_grid),
                   baseline=None if choice is None else _entry(choice), validation=None)
    lines = ["# 历史 3:1 工作点的近等通量对照", "",
             "固定 3:1 工作点。从满足同一最低有效通量的 1:1 训练节点中，"
             "只保留有效通量比 Φ₁:₁/Φ₃:₁ 在 [0.997, 1.003] 内的节点，选择其中散射最少的一点。"
             "若散射相同，依次优先接近目标通量、较低功率、较短作用长度、较低失谐。", "",
             "| 训练节点 | 总功率 (µW) | 全长 (mm) | 共同失谐 (MHz) | "
             "有效通量 (10⁹ 原子/s) | 光子/出射原子 |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in (anchor_grid, choice):
        if row is not None:
            e = _entry(row)
            lines.append("| {} | {:g} | {:g} | {:+g} | {:.5f} | {:.4f} |".format(
                e["family"], e["power_uW"], e["length_mm"], e["detuning_MHz"],
                e["flux_per_s"] / 1e9, e["photons"]))
    if choice is None:
        if metrics is not None:
            raise ValueError("Cannot report matched validation without a selected baseline")
        lines += ["", "没有训练节点同时通过这两项通量约束，未进行该配对验证。"]
    elif metrics is None:
        lines += ["", "上述节点仅在训练集上完成通量匹配；没有独立样本验证结果。"]
    else:
        # The shared metric stores candidate/baseline; use its reciprocal throughout
        # this comparison. For a positive ratio, reversing interval endpoints is exact.
        raw = _triple(metrics, "flux_ratio")
        ratio = [(None if x is None or x <= 0 else 1 / x) for x in (raw[0], raw[2], raw[1])]
        bands = (_triple(metrics, "baseline_flux_per_s"), _triple(metrics, "candidate_flux_per_s"))
        floor = _finite(metrics["min_flux_per_s"])
        both_pass = all(v[1] is not None and v[1] >= floor for v in bands)
        equal_flux = (ratio[1] is not None and ratio[2] is not None and
                      ratio[1] >= 1 - _TOL and ratio[2] <= 1 + _TOL)
        saving = _triple(metrics, "photon_saving")
        lower_scattering = saving[1] is not None and saving[1] > 0
        v = {"incident_samples": int(metrics["incident_samples"]), "bootstrap": int(metrics["bootstrap"]),
             "floor_per_s": floor, "flux_ratio_1to1_over_3to1": ratio, "both_flux_lower_bounds_pass": both_pass,
             "paired_ratio_interval_inside_training_band": equal_flux,
             "photon_saving_lower_bound_positive": lower_scattering}
        for key in ("baseline_photons", "candidate_photons", "baseline_contrast", "candidate_contrast",
                    "baseline_flux_per_s", "candidate_flux_per_s", "photon_saving", "contrast_change"):
            v[key] = _triple(metrics, key)
        payload["validation"] = v
        lines += ["", "独立验证用同一组 {} 个入射源编号；各编号先平均反冲重复，再对编号配对重采样 {} 次。"
                  "下表为点估计 [逐项 95% 区间]。".format(v["incident_samples"], v["bootstrap"]), "",
                  "| 独立验证指标 | 常规 1:1 | 历史 3:1 |", "|---|---:|---:|",
                  "| 有效目标态通量 (10⁹ 原子/s) | {} | {} |".format(
                      _display(bands[0], 1e-9, ".5f"), _display(bands[1], 1e-9, ".5f")),
                  "| 泵浦散射 (光子/出射原子) | {} | {} |".format(
                      _display(v["baseline_photons"]), _display(v["candidate_photons"])),
                  "| 荧光条纹对比度 (%) | {} | {} |".format(
                      _display(v["baseline_contrast"], 100, ".2f"), _display(v["candidate_contrast"], 100, ".2f")),
                  "", "通量比（1:1 / 3:1）：**{}**。两侧通量下界均通过 {:.5g} 原子/s 门槛：**{}**；"
                  "通量比的配对区间完全落入 [0.997, 1.003]：**{}**。".format(
                      _display(ratio, spec=".6f"), floor, "是" if both_pass else "否", "是" if equal_flux else "否"),
                  "", "散射节省（1:1 − 3:1，光子/出射原子）：**{}**。"
                  "对比度变化（3:1 − 1:1，百分点）：**{}**。".format(
                      _display(saving), _display(v["contrast_change"], 100, ".2f")),
                  "", "以上区间仅描述这个预设工作点的仿真统计误差；训练时近等通量并不保证在独立验证时仍近等。"]
    (out / "matched_flux.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf8")
    (out / "matched_flux.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    return payload
