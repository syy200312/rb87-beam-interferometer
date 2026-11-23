"""Paired source-cluster inference for fixed independent-optics validation points.

The incident atom IDs are paired across every optical perturbation and both
recipes. Uncertainty scenarios are fixed settings, not a random sample.
"""
import csv
import json
from pathlib import Path

import numpy as np


PAIRS = {
    "matched80": ("1:1 / 3:1", .003, "历史 80 µW 与近等通量 1:1 对照"),
    "winner55": ("3:1 / 1:1", .01, "55 µW 推荐点与 60 µW 常规基线"),
}
METRICS = ("baseline_efficiency", "candidate_efficiency", "baseline_flux_per_s",
           "candidate_flux_per_s", "baseline_photons", "candidate_photons", "flux_ratio",
           "photon_saving", "photon_reduction")
JOINT = ("baseline_flux_per_s", "candidate_flux_per_s", "flux_ratio", "photon_saving")
SIMULTANEOUS = JOINT + ("photon_reduction",)
PRIOR_REDUCTION = .079  # fixed, rounded reference from historical matched80 rate-equation run


def _positive_integer(x, minimum, name):
    if not isinstance(x, (int, np.integer)) or isinstance(x, (bool, np.bool_)) or x < minimum:
        raise ValueError("%s must be an integer >= %s" % (name, minimum))
    return int(x)


def _cluster(rec, n):
    raw_ids = np.asarray(rec["atom_ids"])
    if (raw_ids.dtype.kind not in "iu" or raw_ids.ndim != 1 or raw_ids.size == 0
            or np.min(raw_ids) < 0 or np.max(raw_ids) >= n or len(set(raw_ids)) != raw_ids.size):
        raise ValueError("Expected nonempty, unique extracted atom IDs within incident source sample")
    values = []
    for key in ("target_population", "accepted_target", "scattered_photons"):
        x = np.asarray(rec[key], dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != len(raw_ids) or x.shape[0] < 1:
            raise ValueError(key + " must be a vector per atom, or (replica, atom)")
        if not np.isfinite(x).all() or np.min(x) < -1e-11 or (key != "scattered_photons" and
                                                               np.max(x) > 1 + 1e-11):
            raise ValueError(key + " is not a physically valid output")
        values.append(np.maximum(0, x.mean(axis=0)))
    if (values[1] > values[0] + 1e-11).any():
        raise ValueError("Accepted clock-state contributions may not exceed the clock-state population")
    # Absent launches contribute zero useful flux; photon and preparation means
    # condition on extraction, including extracted atoms missed by detection.
    result = np.zeros((n, 4))
    result[raw_ids, :3] = np.array(values).T
    result[raw_ids, 3] = 1
    return result


def _ratio(x, y):
    return np.divide(x, y, out=np.full_like(x, np.nan, dtype=float), where=(y > 0))


def _estimates(counts, b, c, incident_flux, n, pair):
    # The multinomial row sum is n, including IDs which were not extracted.
    sums = counts @ np.column_stack((b, c))
    eb, fb, pb, nb, ec, fc, pc, nc = sums.T
    fb, fc = fb * incident_flux / n, fc * incident_flux / n
    nb_photons, nc_photons = _ratio(pb, nb), _ratio(pc, nc)
    return dict(zip(METRICS, (_ratio(eb, nb), _ratio(ec, nc), fb, fc, nb_photons, nc_photons,
                              _ratio(fb, fc) if pair == "matched80" else _ratio(fc, fb),
                              nb_photons - nc_photons, 1. - _ratio(nc_photons, nb_photons))))


def _number(x):
    return float(x) if np.isfinite(x) else None


def _status(lo, hi, minimum=None, maximum=None, strict=False):
    if lo is None or hi is None:
        return "unresolved"
    if minimum is not None and ((hi <= minimum) if strict else (hi < minimum)):
        return "excluded"
    if maximum is not None and lo > maximum:
        return "excluded"
    passes_low = minimum is None or ((lo > minimum) if strict else (lo >= minimum))
    return "supported" if passes_low and (maximum is None or hi <= maximum) else "unresolved"


def _combine(status):
    return "excluded" if "excluded" in status else "supported" if all(s == "supported" for s in status) else "unresolved"


def summarize_obe_pairs(cases, incident_samples, incident_flux_per_s=1e10, floor=9.5e9,
                        bootstrap=2500, seed=20261121):
    """Return (rows, summary) with joint CIs, conditional on the trajectories and model.

    Each case is {pair: matched80|winner55, scenario: str, baseline: rec, candidate: rec}.
    baseline=1:1, candidate=3:1. Each rec has atom_ids and target_population,
    accepted_target (clock probability times acceptance), scattered_photons; the
    latter three are N-vectors or (replicas,N) arrays. All cases must refer to
    the same ensemble and extracted ID set (ordering may differ). The caller
    supplies source provenance and constructs paired stochastic trajectories.
    """
    n = _positive_integer(incident_samples, 2, "incident_samples")
    draws = _positive_integer(bootstrap, 2, "bootstrap")
    seed = _positive_integer(seed, 0, "seed")
    if not np.isfinite([incident_flux_per_s, floor]).all() or not 0 < floor <= incident_flux_per_s:
        raise ValueError("Common minimum flux must be positive and no larger than incident flux")
    cases = list(cases)
    if not cases:
        raise ValueError("An empty validation family cannot support a robustness claim")
    seen = set()
    counts = np.random.default_rng(seed).multinomial(n, np.full(n, 1./n), size=draws)
    ones = np.ones((1, n))
    maxima, rows, scales, ids_mask = np.zeros(draws), [], [], None
    for case in cases:
        pair, scenario = case["pair"], case["scenario"]
        if pair not in PAIRS or not isinstance(scenario, str) or not scenario or (pair, scenario) in seen:
            raise ValueError("Unknown pair, blank scenario or repeated (pair, scenario)")
        seen.add((pair, scenario))
        b, c = (_cluster(case[key], n) for key in ("baseline", "candidate"))
        if ids_mask is None:
            ids_mask = b[:, 3]
        if not np.array_equal(ids_mask, b[:, 3]) or not np.array_equal(ids_mask, c[:, 3]):
            raise ValueError("Cannot pair different extracted source ID sets across cases")
        points = _estimates(ones, b, c, incident_flux_per_s, n, pair)
        boot = _estimates(counts, b, c, incident_flux_per_s, n, pair)
        row = dict(pair=pair, scenario=scenario, ratio_orientation=PAIRS[pair][0], tolerance=PAIRS[pair][1])
        scale = {}
        for key in METRICS:
            point = points[key][0]
            vals = boot[key]
            valid = np.isfinite(vals).all() and np.isfinite(point)
            row[key] = _number(point)
            row[key+"_lo95"], row[key+"_hi95"] = (map(float, np.quantile(vals, [.025,.975]))
                                                    if valid else (None,None))
            if key not in SIMULTANEOUS:
                continue
            scale[key] = None
            if not valid:
                continue
            centered = np.abs(vals-point)
            sd = float(vals.std(ddof=1))
            if centered.max() <= 128*np.finfo(float).eps*max(1.,abs(point)):
                scale[key] = 0.
            elif sd > 0:
                maxima = np.maximum(maxima, centered / sd)
                scale[key] = sd
        rows.append(row)
        scales.append(scale)
    paired_sets = {p: {r["scenario"] for r in rows if r["pair"] == p}
                   for p in PAIRS if any(r["pair"] == p for r in rows)}
    if len(paired_sets) == 2 and len(set(map(frozenset, paired_sets.values()))) != 1:
        raise ValueError("Both pairs must be evaluated on the same declared uncertainty scenarios")
    critical = float(np.quantile(maxima, .95))
    for r, scale in zip(rows, scales):
        tol = r["tolerance"]
        criteria = {"baseline_flux_per_s": (floor, None, False), "candidate_flux_per_s": (floor, None, False),
                    "flux_ratio": (1-tol, 1+tol, False), "photon_saving": (0., None, True)}
        for k in SIMULTANEOUS:
            sd, point = scale[k], r[k]
            lo = (point - critical*sd) if sd is not None else None
            hi = (point + critical*sd) if sd is not None else None
            r[k+"_sim_lo95"], r[k+"_sim_hi95"] = lo, hi
            if k in JOINT:
                r[k+"_status"] = _status(lo, hi, *criteria[k])
        if r["pair"] == "matched80":
            for s in ("","_lo95","_hi95","_sim_lo95","_sim_hi95"):
                value = r["photon_reduction"+s]
                r["retained_fraction_of_prior"+s] = None if value is None else value / PRIOR_REDUCTION
        r["joint_status"] = _combine([r[k+"_status"] for k in JOINT])
        r["joint_point"] = (all(r[k] is not None for k in JOINT) and
                            r["baseline_flux_per_s"] >= floor and r["candidate_flux_per_s"] >= floor
                            and 1-tol <= r["flux_ratio"] <= 1+tol and r["photon_saving"] > 0)
    summary = dict(incident_samples=n, extracted_samples=int(ids_mask.sum()), bootstrap=draws, seed=seed,
                   incident_flux_per_s=incident_flux_per_s, min_flux_per_s=floor,
                   simultaneous_critical=critical, pairs={})
    for p in paired_sets:
        rs = [r for r in rows if r["pair"] == p]
        worst = {}
        for key in JOINT:
            def margin(r):
                lo, hi = r[key+"_sim_lo95"], r[key+"_sim_hi95"]
                if lo is None or hi is None:
                    return -np.inf
                if key == "flux_ratio":
                    return min(lo - (1-r["tolerance"]), 1+r["tolerance"] - hi)
                return lo - (0. if key == "photon_saving" else floor)
            r = min(rs, key=margin)
            worst[key] = dict(scenario=r["scenario"], point=r[key], lo95=r[key+"_sim_lo95"],
                              hi95=r[key+"_sim_hi95"], margin=_number(margin(r)))
        summary["pairs"][p] = dict(scenarios=len(rs), joint_status=_combine([r["joint_status"] for r in rs]),
                                   supported=sum(r["joint_status"]=="supported" for r in rs),
                                   excluded=sum(r["joint_status"]=="excluded" for r in rs),
                                   unresolved=sum(r["joint_status"]=="unresolved" for r in rs),
                                   worst=worst,
                                   excluded_scenarios=[r["scenario"] for r in rs if r["joint_status"]=="excluded"])
        lowest = min(rs,key=lambda rr: rr["photon_reduction_sim_lo95"] if
                     rr["photon_reduction_sim_lo95"] is not None else -np.inf)
        summary["pairs"][p]["worst_reduction"] = dict(scenario=lowest["scenario"],point=lowest["photon_reduction"],
            lo95=lowest["photon_reduction_sim_lo95"], hi95=lowest["photon_reduction_sim_hi95"])
        if p == "matched80":
            summary["pairs"][p]["prior_model_reference_reduction"] = PRIOR_REDUCTION
            summary["pairs"][p]["worst_reduction"]["fraction_of_prior"] = lowest["retained_fraction_of_prior"]
    return rows, summary


LABELS = {"supported":"成立","excluded":"不成立","unresolved":"未分辨"}


def _fmt(x, decimals=4):
    return "未定" if x is None else format(x, "."+str(decimals)+"f")


def write_obe_report(outdir, cases, incident_samples, incident_flux_per_s=1e10, floor=9.5e9,
                     bootstrap=2500, seed=20261121, metadata=None):
    """Write obe_scenarios.csv, obe_summary.json and report.md; return (rows, summary)."""
    rows, summary = summarize_obe_pairs(cases, incident_samples, incident_flux_per_s, floor, bootstrap, seed)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    summary["metadata"] = metadata or {}
    with (out/"obe_scenarios.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    (out/"obe_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    lines = ["# 独立内态动力学复算：通量匹配与泵浦散射", "",
             "比较冻结的 1:1（基线）与 3:1（候选）设置。有效通量计入探测接受度；"
             "入射通量为 {:.4g} 原子/s，双方通量门槛为 {:.4g} 原子/s。"
             "独立入射源编号 {} 个，其中 {} 个出射。".format(incident_flux_per_s, floor, incident_samples,
                                                       summary["extracted_samples"]), ""]
    for k in ("model_note", "uncertainty_note", "source_note", "numerical_note"):
        if (metadata or {}).get(k):
            lines += [str(metadata[k]),""]
    lines += ["| 工作点对 | 通量比定义及匹配容差 | 声明场景数 | 联合成立 / 不成立 / 未分辨 | 所有已测场景均成立？ |",
              "|---|---|---:|---:|---|"]
    for p, s in summary["pairs"].items():
        orientation, tol, label = PAIRS[p]
        lines += ["| {} | {}，±{:g}% | {} | {} / {} / {} | {} |".format(
            label,orientation,tol*100,s["scenarios"],s["supported"],s["excluded"],s["unresolved"],
            {"supported":"是","excluded":"否","unresolved":"证据不足"}[s["joint_status"]])]
    nominal = [r for r in rows if r["scenario"] == "nominal"]
    if nominal:
        lines += ["", "**标称工况**（每对中均按 1:1、3:1 的顺序列出）：", "",
                  "| 工作点对 | 目标态制备效率 (%) | 有效通量 (10⁹ 原子/s) | 散射数 (光子/出射原子) | 通量比 [同时 95%] | 散射节省 [同时 95%] |",
                  "|---|---:|---:|---:|---:|---:|"]
        for r in nominal:
            def both(k, factor=1, decimals=4):
                return " / ".join(_fmt(None if r[x+k] is None else r[x+k]*factor,decimals) for x in ("baseline_","candidate_"))
            def ci(k):
                return "{} [{}, {}]".format(*(_fmt(r[k+suffix],5) for suffix in ("","_sim_lo95","_sim_hi95")))
            lines += ["| {} | {} | {} | {} | {} | {} |".format(PAIRS[r["pair"]][2],both("efficiency",100,2),
                        both("flux_per_s",1e-9),both("photons"),ci("flux_ratio"),ci("photon_saving"))]
    lines += ["","**各判据最不利的已测工况**（按同时区间距门槛的余量，各量未必在同一工况最差）：", "",
              "| 工作点对 | 判据 | 最不利场景 | 点估计 [同时 95%] |","|---|---|---|---:|"]
    for pair, info in summary["pairs"].items():
        for key, label in zip(JOINT, ("1:1 通量 (10⁹/s)", "3:1 通量 (10⁹/s)", "通量比", "散射节省 (光子/出射原子)")):
            x = info["worst"][key]
            factor = 1e-9 if key.endswith("_per_s") else 1.
            data = [_fmt(x[m]*factor if x[m] is not None else None, 6) for m in ("point","lo95","hi95")]
            lines += ["| {} | {} | {} | {} [{}, {}] |".format(PAIRS[pair][2],label,x["scenario"],*data)]
        x = info["worst_reduction"]
        lines += ["| {} | 散射降幅 (%) | {} | {} [{}, {}] |".format(
            PAIRS[pair][2],x["scenario"],*[_fmt(x[k]*100 if x[k] is not None else None, 3)
                                           for k in ("point","lo95","hi95")])]
    for pair, info in summary["pairs"].items():
        excluded = info["excluded_scenarios"]
        if excluded:
            lines += ["", "{}：联合条件不成立的场景：{}。".format(PAIRS[pair][2], "、".join(excluded))]
    if "matched80" in summary["pairs"]:
        lines += ["", "**与原 7.90% 散射降幅比较。** 新降幅为 1 − (3:1 平均散射 / 1:1 平均散射)；"
                  "保留原降幅的比例 = 新降幅 / 7.90%。以历史舍入点估计为固定参照，不将它视作实测真值。", "",
                  "| 工况 | 散射降幅，% [同时 95%] | 为原降幅的百分比 [同时 95%] |","|---|---:|---:|"]
        worst_name = summary["pairs"]["matched80"]["worst_reduction"]["scenario"]
        for r in rows:
            if r["pair"] == "matched80" and r["scenario"] in ("nominal", worst_name):
                def percent_ci(k):
                    return "{} [{}, {}]".format(*[_fmt(r[k+s]*100 if r[k+s] is not None else None,3)
                                                  for s in ("","_sim_lo95","_sim_hi95")])
                lines += ["| {}{} | {} | {} |".format(r["scenario"], "（按降幅下界最不利）" if r["scenario"]==worst_name else "",
                          percent_ci("photon_reduction"),percent_ci("retained_fraction_of_prior"))]
    lines += ["", "[逐场景结果和各判据](obe_scenarios.csv) · [机器可读摘要](obe_summary.json)", "",
              "散射节省 = 每个出射原子的基线平均泵浦散射光子数 − 候选平均数；包含未进入探测孔径的出射原子。"
              "制备效率是出射原子的平均目标态布居；有效通量按全部入射源编号归一，未出射编号贡献零。", "",
              "用同一组入射编号对全部方案、全部设置进行配对 bootstrap（{} 次）；若有重复轨迹，先在编号内取平均。"
              "逐项 95% 区间见 CSV。联合判断使用所有已评估工作点对、所有扰动设置、四个判据及散射相对降幅的"
              "最大标准化 bootstrap 偏差构造近似同时 95% 区间。成立要求两组通量下界均达到门槛，"
              "通量比的整个区间落在规定匹配带内，且散射节省的下界严格大于零。若一个区间完全落在不满足条件的一侧，"
              "标记不成立；其余情况为未分辨。".format(bootstrap), "",
              "场景计数描述离散压力测试，不是出现概率或整段连续参数的证明。区间只衡量给定物理模型、"
              "源和光学设置下有限原子抽样的不确定性，不含模型近似、标定偏差或数值积分误差。"]
    (out/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return rows, summary
