"""Re-solve slow atoms at higher spatial order; assemble primary OBE results.

Run after obe_80_study (nominal128, refine96 and full grid at output root).
Keeps the raw and corrected records separate. All checks use paired IDs.
"""
import argparse
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
for var in ("MKL_NUM_THREADS","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(var,"1")
import numpy as np
from . import obe_80_study as s
from .obe_convergence_check import compare_resolutions
from .obe_pair_report import write_obe_report
from .obe_80_summary import make


def key(case,role,source_id):
    return role,s.scenarios()[case][role=="candidate"],int(source_id)


def upgrade(out,n,base_dir,slices,workers,v_cut,higher_order,scope):
    s._worker_init(s.SOURCE,s.CONFIG,n)
    src=s._source
    low_ids=np.flatnonzero(abs(src.v_m_s[:,0])<v_cut)
    keep={"all":list(s.scenarios()), "nominal":["nominal"],
          "power_pair":["T97_D103_ideal","T103_D97_ideal"]}
    combos={name:s.scenarios()[name] for name in keep[scope]}
    todo=sorted(set(key(c,role,i) for c in combos for role in s.SETTINGS for i in low_ids))
    # Scope/report path is distinct; this cache is valid only for this source/solver/config.
    digests=[s._digest(Path(__file__).with_name("independent_obe.py")),s._digest(s.SOURCE),
             s._digest(s.CONFIG)]
    directory=out/"refinement_blocks"
    directory.mkdir(parents=True,exist_ok=True)
    found={};pending={}
    with ProcessPoolExecutor(max_workers=workers,initializer=s._worker_init,initargs=(s.SOURCE,s.CONFIG,n)) as pool:
        for k in todo:
            role,f,i=k
            uid=s.sha256(json.dumps([digests,k,slices,higher_order]).encode()).hexdigest()[:24]
            path=directory/(uid+".json")
            if path.exists():
                found[k]=json.loads(path.read_text(encoding="utf-8"));continue
            pending[pool.submit(s._block,(role,f,slices,higher_order,[i]))]=(k,path)
        print("slow-source recalculations:",len(pending),flush=True)
        for t,fut in enumerate(as_completed(pending),1):
            k,path=pending[fut]; found[k]=fut.result(); s.save(path,found[k])
            if t%12==0:print(t,"/",len(pending),flush=True)
    cases=json.loads((base_dir/"atom_cases.json").read_text(encoding="utf-8"))
    original=json.loads(json.dumps(cases))
    for c in cases:
        for role in s.SETTINGS:
            rr=c[role]; loc={i:j for j,i in enumerate(rr["atom_ids"])}
            for i in low_ids:
                r=found[key(c["scenario"],role,i)]; j=loc[r["atom_ids"][0]]
                for x in ("target_population","accepted_target","scattered_photons"):
                    rr[x][j]=r[x][0]
    s.save(out/"atom_cases.json",cases)
    details=json.loads((base_dir/"obe_summary.json").read_text(encoding="utf-8"))["metadata"]
    details["numerical_note"]=("慢横速原子 |vx| < %g m/s（%d/%d 个）改用驻波傅里叶阶 %d、%d 个时间片；"
        "其余编号使用原始网格。置信区间不含数值误差、反冲与未知两频驻波相位误差。"
        % (v_cut,len(low_ids),n,higher_order,slices))
    details["refinement"]=dict(velocity_cut_m_s=v_cut,source_ids=low_ids.tolist(),order=higher_order,slices=slices)
    s.save(out/"protocol.json",dict(original_protocol=details["settings"],refinement=details["refinement"]))
    compare_resolutions(original,cases,out/"spatial_refinement")
    write_obe_report(out,cases,n,metadata=details)
    make(out)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",required=True);p.add_argument("--output",required=True);p.add_argument("--atoms",type=int,default=32)
    p.add_argument("--workers",type=int,default=12);p.add_argument("--slices",type=int,default=64)
    p.add_argument("--v-cut",type=float,default=.015);p.add_argument("--order",type=int,default=3)
    p.add_argument("--scope",choices=["nominal","power_pair","all"],default="all")
    a=p.parse_args();upgrade(Path(a.output),a.atoms,Path(a.input),a.slices,a.workers,a.v_cut,a.order,a.scope)
