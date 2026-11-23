"""Small 80-uW paired sensitivity check on selected slow, typical and faster atoms.

The atoms are deliberately not a representative sample. These numbers are
diagnostics of assumptions, never estimates of the beam's population mean.
"""
import argparse
import csv
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np


def one(job):
    from .batch import subset_ensemble
    from .io import read_stateful_ensemble
    from .independent_obe import propagate_ensemble
    from .obe_80_study import SETTINGS
    path, atom_id, role, mode, slices, order = job
    s = read_stateful_ensemble(path)
    ids = np.asarray(s.source_metadata["sample_ids"])
    source = subset_ensemble(s, np.flatnonzero(ids==atom_id))
    p, q, length, delta = SETTINGS[role]
    res = propagate_ensemble(source, transfer_uW=p, dark_uW=q, length_mm=length,
        detuning_MHz=delta, group_phase_rad=np.pi/2 if mode=="standing_offset" else 0.,
        secular_hyperfine=mode!="hyperfine_coherent", grating_order=order, slices=slices)
    return dict(atom_id=int(atom_id), role=role, mode=mode, vx_m_s=float(source.v_m_s[0,0]),
                target=float(res["target_population"][0]), photons=float(res["scattered_photons"][0]),
                ground_density_min_eig=float(res["min_exit_eigenvalue"][0]),
                trace_error=float(res["trace_error"][0]), slices=slices, spatial_order=order)


def run(args):
    from .io import read_stateful_ensemble
    from . import independent_obe
    if args.workers not in (1,2,3):
        raise ValueError("This diagnostic uses at most three workers")
    s = read_stateful_ensemble(args.source)
    ids = np.asarray(s.source_metadata["sample_ids"])
    inds = np.flatnonzero(ids < 64)
    order = inds[np.argsort(np.abs(s.v_m_s[inds,0]))]
    selected = list(map(int,ids[order[[0,len(order)//2,7*len(order)//8]]]))
    # Targeted hypothesis tests only: small-|vx| atom for both hyperfine models;
    # three velocities for standing-wave phase. No parameter tuning after comparisons.
    jobs = [(args.source, i, role, mode, args.slices,args.order) for i in selected
            for mode in ("reference","standing_offset","hyperfine_coherent")
            if mode!="hyperfine_coherent" or i==selected[0]
            for role in ("baseline","candidate")]
    folder=Path(args.output);folder.mkdir(exist_ok=True,parents=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        records=list(pool.map(one,jobs))
    def keyed(role,mode,i):
        return next(r for r in records if (r["role"],r["mode"],r["atom_id"])==(role,mode,i))
    rows = []
    for i in selected:
        for mo in ("reference","standing_offset","hyperfine_coherent"):
            if mo=="hyperfine_coherent" and i!=selected[0]:
                continue
            b,c=keyed("baseline",mo,i),keyed("candidate",mo,i)
            refb,refc=keyed("baseline","reference",i),keyed("candidate","reference",i)
            ratio=b["target"]/c["target"]
            gain=1-c["photons"]/b["photons"]
            rows.append(dict(atom_id=i,model=mo,vx_m_s=b["vx_m_s"],
                    target_baseline=b["target"],target_candidate=c["target"],target_ratio=ratio,
                    photons_baseline=b["photons"],photons_candidate=c["photons"],
                    photon_saving=b["photons"]-c["photons"], reduction=gain,
                    ratio_change=ratio-refb["target"]/refc["target"],
                    reduction_change=gain-(1-refc["photons"]/refb["photons"])))
    with (folder/"comparisons.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    meta=dict(slices=args.slices,order=args.order,source_ids=selected,source_sha256=hashlib.sha256(
        Path(args.source).read_bytes()).hexdigest(),solver_sha256=hashlib.sha256(
            Path(independent_obe.__file__).read_bytes()).hexdigest(),records=records)
    (folder/"records.json").write_text(json.dumps(meta,indent=2,allow_nan=False),encoding="utf-8")
    lines=["# 双 80 µW 工作点：物理假设敏感性抽查", "",
           "取验证源库前 64 个入射编号，按 |vₓ| 排序选最慢、中位、7/8 分位的三个出射原子。"
           "这些原子按速度刻意选取，单原子的目标态概率比不等于整束有效通量比。两组沿用各自泵浦功率、长度与失谐，"
           "标称功率和理想 y 线偏振。共用同一组速度、位置、初始态和空间/时间分辨率。", "",
           "reference：同频两行波相干，两种颜色的驻波波腹对齐，并忽略不同 F 或 F′ 的同流形相干；"
           "standing_offset：只改变两种颜色的驻波相对位置，给转移光的 ±k 场系数加 ±π/2 相位，"
           "从而使一组波腹移到另一组波节；hyperfine_coherent：保持波腹对齐，恢复不同 F、不同 F′ 的密度矩阵元，"
           "两频相对相干但每个色只驱动其近共振基态 F，仍忽略距 6.8 GHz 的另一颜色。", "",
           "固定空间阶 {}、每个光区 {} 个时间片。数值误差另行核验。".format(args.order,args.slices), "",
           "| 源 ID | vₓ (m/s) | 模型 | 目标态比 1:1/3:1 | 散射 1:1 / 3:1 | 散射降幅 (%) | 降幅相对 reference 的变化 (百分点) |",
           "|---:|---:|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append("| {atom_id} | {vx_m_s:.6f} | {model} | {target_ratio:.6f} | {photons_baseline:.5f} /"
                     " {photons_candidate:.5f} | {gain:.3f} | {delta:.3f} |".format(**r,gain=100*r["reduction"],
                                                                            delta=100*r["reduction_change"]))
    lines+=["","[逐原子比较](comparisons.csv) · [各处方的原始数值和模型摘要](records.json)"]
    (folder/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(folder/"report.md")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",default="beam_outputs/pumping_optimization/source/source_ensemble.h5")
    p.add_argument("--slices",default=64,type=int)
    p.add_argument("--order",default=1,type=int)
    p.add_argument("--workers",default=2,type=int)
    p.add_argument("--output",default="beam_outputs/obe_80/physics_checks")
    run(p.parse_args())
