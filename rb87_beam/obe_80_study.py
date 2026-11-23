"""Paired independent optical-Bloch calculation for two frozen 80-uW recipes.

python -m rb87_beam.obe_80_study --help
"""
import argparse
from concurrent.futures import as_completed, ProcessPoolExecutor
from hashlib import sha256
import itertools
import json
import os
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
import numpy as np
from .batch import subset_ensemble, source_prefix
from .config import SimulationConfig
from .io import read_stateful_ensemble
from .physics import flight_time, trajectory, reference_y
from .obe_pair_report import write_obe_report

SETTINGS = {"baseline": (40.,40.,16.,0.), "candidate": (60.,20.,4.,1.5)}
SOURCE = "beam_outputs/pumping_optimization/source/source_ensemble.h5"
CONFIG = "beam_configs/optical_pumping_optimized.json"
REFS = {
    "power": "https://www.thorlabs.com/catalogpages/V21/1560.PDF",
    "polarization_in_beam_experiment": "https://libra.unine.ch/server/api/core/bitstreams/541f7e28-0ea5-4827-ab34-09a04635c5a0/content",
    "polarizing_cube": "https://www.edmundoptics.com/p/35mm-780nm-laser-line-polarizing-cube-beamsplitter/7648/",
    "atom": "https://steck.us/alkalidata/rubidium87numbers.pdf"}


def fields(a=1., b=1., pi=0., phase=0.):
    return (a,b,pi,phase)


def scenarios():
    # In common-mode/channel cases each arm sees the same multiplicative error.
    # Last two compare sequential retuning with opposite power biases.
    high, med = 1/1001., 1/10001.
    nominal = fields()
    sc = {"nominal": (nominal,nominal)}
    for label, f in (("polarization40dB",fields(pi=med)), ("polarization30dB",fields(pi=high)),
                     ("elliptical30dB",fields(pi=high,phase=np.pi/2)),
                     ("elliptical30dB_opposite",fields(pi=high,phase=-np.pi/2))):
        sc[label] = (f, f)
    for pi, pol in [(0., "ideal"), (high, "30dB")]:
        for a, b in itertools.product((.97,1.03), repeat=2):
            f = fields(a,b,pi)
            label = "T%d_D%d_%s" % (round(a*100),round(b*100),pol)
            sc[label] = (f,f)
        low_power, high_power = fields(.97,.97,pi), fields(1.03,1.03,pi)
        sc["reset_1low_3high_"+pol] = (low_power, high_power)
        sc["reset_1high_3low_"+pol] = (high_power, low_power)
    # Wave plate/window drift between runs; deliberately stress, not a measured distribution.
    sc["reset_1pure_3leak"] = (nominal, fields(pi=high))
    sc["reset_1leak_3pure"] = (fields(pi=high), nominal)
    return sc


def save(path, obj):
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def _digest(f):
    return sha256(Path(f).read_bytes()).hexdigest()


def _worker_init(source_file, cfg_file, n):
    global _source, _cfg
    f = read_stateful_ensemble(source_file)
    if not 0 < n <= f.source_metadata["sampled"]:
        raise ValueError("Source ID prefix exceeds available incident sample")
    _source = source_prefix(f, n)
    _cfg = SimulationConfig.from_dict(json.loads(Path(cfg_file).read_text(encoding="utf-8")))
    assert _cfg.raman.state_handling == "inherit" and _cfg.raman.selection_sigma_vx_m_s == 0


def _block(args):
    from .independent_obe import propagate_ensemble
    role, field, slices, order, ids = args
    src = subset_ensemble(_source, ids)
    a,b,pi,phase = field
    p,q,length,delta = SETTINGS[role]
    result = propagate_ensemble(src, transfer_uW=p*a, dark_uW=q*b, length_mm=length,
                        detuning_MHz=delta, pi_fraction=pi, polarization_phase_rad=phase,
                        slices=slices, grating_order=order)
    r,_ = trajectory(src, flight_time(src, _cfg.detection.z_m), _cfg.raman)
    rad = np.hypot(r[:,0], r[:,1]-reference_y(_cfg.detection.z_m, _cfg.raman))
    target = np.asarray(result["target_population"])
    photons = np.asarray(result["scattered_photons"])
    if (target.shape != (len(ids),) or photons.shape != target.shape or
        not np.isfinite(target).all() or not np.isfinite(photons).all() or
        np.min(target) < -1e-10 or np.max(target) > 1+1e-10 or np.min(photons) < 0):
        raise ArithmeticError("Invalid atom-resolved OBE result")
    return dict(atom_ids=src.source_metadata["sample_ids"], target_population=target.tolist(),
                accepted_target=(target*(rad<_cfg.detection.aperture_radius_m)).tolist(),
                scattered_photons=photons.tolist(), radius_m=rad.tolist())


def run(args):
    out = Path(args.output); out.mkdir(parents=True,exist_ok=True)
    cache = out / "blocks"; cache.mkdir(exist_ok=True)
    _worker_init(args.source,CONFIG,args.atoms)
    src = _source
    if src.source_metadata["incident_flux_per_s"] != 1e10:
        raise ValueError("Comparison requires the archived 1e10/s incident normalization")
    protocol = dict(recipes=SETTINGS, source_sha256=_digest(args.source), config_sha256=_digest(CONFIG),
                    solver_sha256=_digest(Path(__file__).with_name("independent_obe.py")), normalization_per_s=1e10,
                    tolerance=.003, floor_per_s=9.5e9, n_incident=args.atoms, slices=args.slices,
                    grating_order=args.order, sigma_p_based_on_measurement=False,
                    field_components=["transfer power multiplier","dark power multiplier","pi power fraction","Jones phase (rad)"],
                    sources=REFS, numerical_method="independent optical Bloch", scenarios=scenarios())
    if args.suite=="nominal":
        protocol["scenarios"]={"nominal":scenarios()["nominal"]}
    if args.suite=="controls":
        protocol["scenarios"]={k:scenarios()[k] for k in ("nominal","T97_D103_30dB","T103_D97_30dB")}
    if args.suite=="power_pair":
        protocol["scenarios"]={k:scenarios()[k] for k in ("T97_D103_ideal","T103_D97_ideal")}
    key = sha256(json.dumps({k:v for k,v in protocol.items() if k!="scenarios"},sort_keys=True).encode()).hexdigest()
    cachedir = cache / key[:16]; cachedir.mkdir(exist_ok=True)
    save(out/"protocol.json",protocol)
    fields_by_recipe = sorted({(role,ff) for pair in protocol["scenarios"].values()
                               for role,ff in zip(SETTINGS, pair)})
    jobs, result = {}, {}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init,
                         initargs=(args.source,CONFIG,args.atoms)) as pool:
        for role,ff in fields_by_recipe:
            for i in range(0,len(src.r_m),args.block):
                inds=list(range(i,min(i+args.block,len(src.r_m))))
                tok=sha256(json.dumps([role,ff,inds],sort_keys=True).encode()).hexdigest()[:20]
                path = cachedir/(tok+".json")
                j=(role,ff,i)
                if path.exists():
                    result[j]=json.loads(path.read_text(encoding="utf-8"))
                else:
                    jobs[pool.submit(_block,(role,ff,args.slices,args.order,inds))]=(j,path)
        print("uncached blocks:",len(jobs),"cached:",len(result),flush=True)
        for i,f in enumerate(as_completed(jobs),1):
            j,path=jobs[f]; result[j]=f.result(); save(path,result[j])
            if i%max(1,len(jobs)//8)==0 or i==len(jobs): print("solved",i,"/",len(jobs),flush=True)
    def assemble(role,ff):
        groups = [result[role,ff,i] for i in range(0,len(src.r_m),args.block)]
        return {k:[v for g in groups for v in g[k]] for k in groups[0]}
    cases = [dict(pair="matched80",scenario=name, baseline=assemble("baseline",f1),
                  candidate=assemble("candidate",f2)) for name,(f1,f2) in protocol["scenarios"].items()]
    save(out/"atom_cases.json",cases)
    metadata=dict(model_note="使用独立的 ⁸⁷Rb D₂ 多能级光学布洛赫方程。沿同一组 MOT 出口弹道推进内态，"
                  "按原探测孔径接收；不反馈泵浦光力或随机反冲。模型取舍和数值加密见配套说明。",
                  uncertainty_note="未找到这两个工作点在原子处的实测功率及偏振记录。±3% 是 780 nm 功率计规格尺度，"
                  "不是漂移标准差；π 功率分数测试 0、1/10001、1/1001。以正交偏振等于 z 向分量为前提。"
                  "T/D 分别指转移光/暗态制备光的功率百分数；reset 表示两次设置偏差方向不同。"
                  "实线偏振泄漏和 ±π/2 相位的椭圆偏振是不同情形。",
                  source_note="2D⁺ MOT 源库沿用上一轮验证的编号前缀。未复制或重抽原子来扩大样本。",
                  numerical_note="区间仅涵盖源抽样；传播分辨率、光场相干性和弹道近似须另外检查。",
                  settings=protocol)
    write_obe_report(out,cases,args.atoms,metadata=metadata)
    print(out/"report.md",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",default=SOURCE)
    p.add_argument("--atoms",type=int,default=32)
    p.add_argument("--slices",type=int,default=48)
    p.add_argument("--order",type=int,default=0)
    p.add_argument("--suite",choices=["nominal","controls","power_pair","all"],default="all")
    p.add_argument("--workers",type=int,default=12)
    p.add_argument("--block",type=int,default=1)
    p.add_argument("--output",default="beam_outputs/obe_80")
    a=p.parse_args()
    if min(a.workers,a.block,a.slices)<1 or a.order<0 or a.atoms<2:
        p.error("Positive atom, time-step, block and worker counts required")
    run(a)


if __name__ == "__main__":
    main()
