"""Read a finished archive; show matching and scattering on the same scenario rows."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

def make(out):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    out=Path(out)
    rows=list(csv.DictReader((out/"obe_scenarios.csv").open(encoding="utf-8-sig")))
    summary=json.loads((out/"obe_summary.json").read_text(encoding="utf-8"))
    groups = [("标称",lambda x:x=="nominal"),("只改偏振（两组共享）",lambda x:x.startswith(("polarization","elliptical"))),
              ("功率四角（两组共享，理想偏振）",lambda x:x.startswith("T") and x.endswith("ideal")),
              ("功率四角（两组共享，30 dB）",lambda x:x.startswith("T") and x.endswith("30dB")),
              ("两次设置反向漂移",lambda x:x.startswith("reset"))]
    lines=["# 两个 80 µW 工作点：通量与散射一起看","",
           "基线：40/40 µW，16 mm，共振。候选：60/20 µW，4 mm，两频均 +1.5 MHz。"
           "以下通量比为基线/候选；降幅为 1 − 候选散射/基线散射。"
           "各行范围是该组离散工况的点估计范围，不是误差棒。","",
           "| 工况 | 点数 | 通量比范围 | 散射降幅范围 (%) | 联合成立 / 排除 / 未分辨* |",
           "|---|---:|---:|---:|---:|"]
    for name,pred in groups:
        sel=[r for r in rows if pred(r["scenario"])]
        if not sel: continue
        ratio=[float(r["flux_ratio"]) for r in sel]
        gain=[100*float(r["photon_reduction"]) for r in sel]
        counts=[sum(r["joint_status"]==q for r in sel) for q in ("supported","excluded","unresolved")]
        lines+=["| {} | {} | {:.6f} 至 {:.6f} | {:.3f} 至 {:.3f} | {} / {} / {} |".format(
            name,len(sel),min(ratio),max(ratio),min(gain),max(gain),*counts)]
    lines+=["","* 联合判断同时要求两侧有效通量 ≥ 9.5×10⁹ 原子/s、通量比在 [0.997,1.003] 内、散射节省为正。"
            "表中判断只计同一 OBE 模型下的源抽样不确定性；[数值与物理假设核验](validation_notes.md)需结合阅读。",
            "","![各场景的通量差和散射降幅](uncertainty.png)","",
            "[数值判据与配对区间](report.md) · [逐场景数值](obe_scenarios.csv) · [预设工况、误差来源与求解参数](protocol.json)"]
    (out/"comparison.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    fig,(a,b)=plt.subplots(1,2,sharey=True,figsize=(10,max(4,1.6+.3*len(rows))))
    for axis,key,fac,bias in ((a,"flux_ratio",100,-100),(b,"photon_reduction",100,0)):
        num=lambda r,k:float(r[k])*fac+bias
        x=np.array([num(r,key) for r in rows]); lo=np.array([num(r,key+"_sim_lo95") for r in rows]); hi=np.array([num(r,key+"_sim_hi95") for r in rows])
        y=np.arange(len(rows))
        axis.errorbar(x,y,xerr=np.maximum(0,[x-lo,hi-x]),fmt="o",ms=4,color="#184c72",capsize=2,lw=1)
        axis.grid(axis="x",alpha=.22)
    a.axvspan(-.3,.3,color="#149065",alpha=.15,label="matching band"); a.axvline(0,color=".7",lw=.8)
    b.axvline(7.9,color="#a24d1f",ls="--",lw=1.2,label="prior 7.90%")
    a.set_xlabel("Flux difference, 100 × (1:1 / 3:1 − 1) [%]");b.set_xlabel("Pump photon reduction [%]")
    a.legend(fontsize=8,loc="best");b.legend(fontsize=8,loc="best")
    a.set_yticks(y);a.set_yticklabels([r["scenario"].replace("_"," ") for r in rows],fontsize=8);a.invert_yaxis()
    fig.suptitle("80 µW: paired optical Bloch simulations ({} incident trajectories)".format(summary["incident_samples"]),fontsize=11)
    fig.tight_layout();fig.savefig(out/"uncertainty.png",dpi=155);plt.close(fig)

def assemble_checks(root):
    """Combine the 128-ID controls and repeat the matched temporal resolution check."""
    from .obe_pair_report import write_obe_report
    from .obe_convergence_check import compare_resolutions
    root=Path(root)
    load=lambda path:json.loads(path.read_text(encoding="utf-8"))
    nominal=root/"nominal128_checked"; power=root/"power128_checked"
    cases=load(nominal/"atom_cases.json")+load(power/"atom_cases.json")
    for c in cases:
        for key in ("baseline","candidate"):
            if c[key]["atom_ids"]!=cases[0]["baseline"]["atom_ids"]:
                raise ValueError("The 128-source controls are not paired")
    original=load(nominal/"protocol.json")["original_protocol"]
    check=load(power/"protocol.json")["original_protocol"]
    if any(original[k]!=check[k] for k in ("n_incident","source_sha256","config_sha256","solver_sha256")):
        raise ValueError("Source or solver mismatch across the 128-source controls")
    meta=load(nominal/"obe_summary.json")["metadata"];meta["settings"]={
        "notes":"Three settings in a single simultaneous family; source and solver hashes in the input protocols.",
        "inputs":["../nominal128_checked/protocol.json","../power128_checked/protocol.json"]}
    dest=root/"key128"
    write_obe_report(dest,cases,original["n_incident"],metadata=meta)
    (dest/"atom_cases.json").write_text(json.dumps(cases,indent=2,allow_nan=False),encoding="utf-8")
    fine=load(root/"refine96"/"atom_cases.json")
    coarse=[c for c in load(root/"atom_cases.json") if c["scenario"] in {r["scenario"] for r in fine}]
    compare_resolutions(coarse,fine,root/"time_refinement")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("out")
    p.add_argument("--assemble-checks",action="store_true")
    a=p.parse_args()
    if a.assemble_checks:assemble_checks(a.out)
    else:make(a.out)
