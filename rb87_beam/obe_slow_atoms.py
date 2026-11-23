"""Standing-wave trajectory quadrature to check spatial harmonics for small |vx|.

For each initial standing-wave phase and detuning the explicitly retained
ground, excited and optical density matrix elements follow the time-dependent
Hermitian classical field H(theta). This independent spatial discretization
has no population/coherence Fourier cutoff. It reuses the same D2 atomic
Hamiltonian and Lindblad dissipator as independent_obe. The instantaneous
laser envelope and standing-wave phase are sampled at the step midpoint.
Increase both phase_nodes and slices to check spatial and time convergence.
The optical frequency is rotating-wave transformed, not numerically resolved.
"""
from functools import lru_cache

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from .independent_obe import _design, GF, EF, GAMMA, K


@lru_cache(32)
def _local_design(transfer_uW, dark_uW, length_mm, detuning_MHz, pi_fraction,
                  polarization_phase_rad, secular_hyperfine, bias_G, transfer_pi, dark_pi, group_phase_rad):
    # The m=0 Fourier design has the ordering (n=-1 optical, n=0 same-manifold,
    # n=+1 optical, count). Recombine to obtain the *instantaneous* OBE.
    free,light,orders,start,center,gz,d=_design(transfer_uW,dark_uW,length_mm,
        detuning_MHz,pi_fraction,polarization_phase_rad,0,secular_hyperfine,bias_G,
        transfer_pi,dark_pi,group_phase_rad)
    opt0=np.flatnonzero(orders==-1)
    pop=np.arange(start,start+len(center))
    opt1=np.flatnonzero(orders==1)
    extra=[len(orders)-1]
    indices=np.r_[pop,opt1,extra]
    free=free[indices,:][:,indices].tocsc()
    # exp(+i theta) derivative: output n is sourced by n-1.
    # exp(-i theta) derivative: output n is sourced by n+1.
    pp=sp.csr_matrix((len(pop),len(pop))); oo=sp.csr_matrix((len(opt1),len(opt1)))
    plus=sp.bmat([[pp,light[pop,:][:,opt0]],[light[opt1,:][:,pop],oo]],format="csc")
    minus=sp.bmat([[pp,light[pop,:][:,opt1]],[light[opt0,:][:,pop],oo]],format="csc")
    plus=sp.block_diag((plus,sp.csr_matrix((1,1))),format="csc")
    minus=sp.block_diag((minus,sp.csr_matrix((1,1))),format="csc")
    return free,plus,minus,center,gz,d


def phase_quadrature(source, *, transfer_uW,dark_uW,length_mm,detuning_MHz,
                     pi_fraction=0.,polarization_phase_rad=0.,slices=128,phase_nodes=12,
                     secular_hyperfine=True,bias_G=2.05,transfer_pi=None,dark_pi=None,
                     group_phase_rad=0.):
    """Call as propagate_ensemble(...), replacing grating_order by phase_nodes.

    slices is the total number of steps in each phase trajectory. vx=0 atoms
    require no Doppler phase resolution but fastest nonzero vx may require
    thousands of time steps; callers must assess this method only on slow atoms.
    Output includes midpoint phase spread (max radians/step in any trajectory).
    """
    if any(type(n) is not int or n<2 for n in (slices,phase_nodes)):
        raise ValueError("Need at least two spatial phases and two time steps")
    vals=(transfer_uW,dark_uW,length_mm,detuning_MHz,pi_fraction,polarization_phase_rad,bias_G,group_phase_rad)
    if not np.isfinite(vals).all() or min(transfer_uW,dark_uW)<0 or not 0<=pi_fraction<=1 or not 0<length_mm<28:
        raise ValueError("Invalid light settings")
    free,plus,minus,center,gz,d=_local_design(transfer_uW,dark_uW,length_mm,detuning_MHz,
          pi_fraction,polarization_phase_rad,secular_hyperfine,bias_G,transfer_pi,dark_pi,group_phase_rad)
    # A pi phase shift changes the sign of all optical coherences but not any
    # population. Integrate [0,pi). A half-node offset avoids sampling only
    # standing-wave nodes/antinodes at vx=0.
    phases=(np.arange(phase_nodes)+.5)*np.pi/phase_nodes
    diag=np.searchsorted(center,np.arange(24)*25)
    jf=np.einsum("qeg,qeg->ge",d,d)
    same=(plus-minus).nnz==0
    if same: light=(plus+minus).tocsc()
    nr=len(source.r_m); populations=np.zeros((nr,8)); photons=np.zeros(nr); errors=np.zeros(nr)
    step_phase=np.zeros(nr); peak=np.zeros(nr)
    full=length_mm*1e-3; entry=.014-full/2; g=1-1/np.sqrt(2)
    if source.ground_density is None or entry<=source.reference_plane_z_m:
        raise ValueError("Need stateful MOT-exit ensemble upstream of pump")
    eye=sp.eye(free.shape[0],format="csc")
    for idx,(r,v,rho) in enumerate(zip(source.r_m,source.v_m_s,source.ground_density)):
        if v[2]<=0: raise ValueError("Forward atoms required")
        t0=(entry-source.reference_plane_z_m)/v[2]; dt=full/v[2]/slices
        step=GAMMA*dt
        t=(np.arange(slices)+.5)*dt
        y=r[1]+v[1]*(t+t0)-.5*9.80665*(t+t0)**2
        fld=np.exp(-(4*((np.arange(slices)+.5)/slices-.5))**2-(y/.00625)**2)
        initial=np.zeros(free.shape[0],complex)
        rho0=np.zeros((24,24),complex)
        rho0[:8,:8]=rho*np.exp(-1j*(gz[:,None]-gz[None,:])*GAMMA*t0)*(GF[:,None]==GF[None,:])
        initial[:len(center)]=rho0.ravel()[center]
        const=eye-step*g*free; step_phase[idx]=abs(K*v[0]*dt)
        if same: drive=-step*g*light
        else: lp=-step*g*plus; lm=-step*g*minus
        for phase in phases:
            state=initial.copy(); theta=phase+K*v[0]*t
            if same: factor=fld*np.cos(theta)
            else: factor=fld*np.exp(1j*theta)
            for val in factor:
                if same: mat=const+drive*val
                else: mat=const+lp*val+lm*np.conj(val)
                fac=splu(mat)
                tmp=fac.solve(state)
                state=fac.solve((1/g-1)*tmp+(2-1/g)*state)
                peak[idx]=max(peak[idx],state[diag[8:]].real.sum())
            pop=state[diag].real
            errors[idx]=max(errors[idx],abs(pop.sum()-1))
            populations[idx]+=(pop[:8]+jf@pop[8:])/phase_nodes
            photons[idx]+=(state[-1].real+pop[8:].sum())/phase_nodes
    if np.max(errors)>1e-7 or populations.min() < -1e-6 or photons.min() < -1e-6:
        raise ArithmeticError("Unphysical phase-quadrature result; refine time/phase sampling")
    return dict(target_population=populations[:,1],ground_populations=populations,
                scattered_photons=photons,trace_error=errors,phase_radians_per_step=step_phase,
                peak_excited_population=peak)


def _main():
    """Reproduce the slow-ID quadrature control for the two historical recipes."""
    import argparse
    import json
    from pathlib import Path
    from .io import read_stateful_ensemble
    from .batch import subset_ensemble
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",default="beam_outputs/pumping_optimization/source/source_ensemble.h5")
    p.add_argument("--output",default="beam_outputs/obe_80/slow_phase_check.json")
    p.add_argument("--id",type=int,default=3)
    a=p.parse_args()
    src=read_stateful_ensemble(a.source)
    loc=np.flatnonzero(np.asarray(src.source_metadata["sample_ids"])==a.id)
    if len(loc)!=1:
        raise ValueError("Requested source ID not uniquely present")
    src=subset_ensemble(src,loc)
    result=[]
    for n,m in ((48,4),(96,8),(192,8),(192,16),(384,8)):
        for role,t,d,l,delta in (("1:1",40.,40.,16.,0.),("3:1",60.,20.,4.,1.5)):
            kw=dict(transfer_uW=t,dark_uW=d,length_mm=l,detuning_MHz=delta)
            out=phase_quadrature(src,phase_nodes=m,slices=n,**kw)
            result.append(dict(role=role,slices=n,phase_nodes=m,settings=kw,
                target_population=float(out["target_population"][0]),
                photons=float(out["scattered_photons"][0]),
                max_step_phase_rad=float(out["phase_radians_per_step"][0]),
                trace_error=float(out["trace_error"][0])))
    path=Path(a.output); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(dict(source_id=a.id,vx_m_s=src.v_m_s[0,0],
        method="direct trajectories averaged over uniform entrance phase; no Fourier truncation",
        phase_origin="shared aligned standing waves",results=result),indent=2,
        allow_nan=False),encoding="utf-8")


if __name__=="__main__":
    _main()
