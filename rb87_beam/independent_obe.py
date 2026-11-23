"""Multilevel D2 optical Bloch equations for the 80-uW pump benchmark.

No PyLCP matrices, rate-equation solver or adiabatic elimination. In angular
units the drive is hbar*Omega/2 (cycling Omega=Gamma*sqrt(s/2)); the local
emission counter obeys dN/dt=Gamma*Tr(rho_e). Basis and constants: Steck,
https://steck.us/alkalidata/rubidium87numbers.pdf . The two frequencies drive
all dipole-allowed excited hyperfine components from their respective F=1,2
manifold (discard the other-colour coupling ~6.8 GHz away). The optical
rotating-wave approximation makes the Hamiltonian static apart from beam
profile and spatial phase. The two balanced counterpropagating components are
coherent and co-polarized; the standing-wave origin is uniformly averaged.
The two colours share their standing-wave phase (equal-k approximation).
The Fourier harmonics of the density obey a linear block-tridiagonal ODE.
grating_order=0 keeps the spatial mean GG/EE elements and the first EG/GE
harmonics. Order 1 adds GG/EE +/-2 and EG/GE +/-3, etc. Boundary harmonics
are zero. Unlike a phase-averaged intensity rate equation, the optical
coherences are propagated in time, including saturation and stimulated emission.

With secular_hyperfine=True the equation projects onto GG and EE blocks with
the same F and F' respectively. Zeeman coherences are retained in each such
block, and every allowed optical coherence F,m to F',m' is represented.
Set False to retain inter-hyperfine coherences too. Spontaneous channels are
resolved by (F,F',q) but not m. Both optical frequencies are taken to have
no linewidth during transit. Spatial motion is prescribed (gravity in y,
ballistic x,z); no recoil updates. Do not use as a recoil simulation.
"""
from functools import lru_cache

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from sympy import Rational
from sympy.physics.wigner import wigner_3j, wigner_6j

GAMMA = 2 * np.pi * 6.065e6
K = 2 * np.pi / 780.241209686e-9
ISAT = 16.69                         # W/m^2, cycling two-level convention
GF = np.array([f for f in (1,2) for _ in range(2*f+1)])
GM = np.array([m for f in (1,2) for m in range(-f,f+1)])
EF = np.array([f for f in range(4) for _ in range(2*f+1)])
EM = np.array([m for f in range(4) for m in range(-f,f+1)])
_EHFS = np.cumsum([0.,72.218,156.947,266.651]) * 2*np.pi*1e6
_MU = 2*np.pi*1.39962449e6           # rad/s/G


@lru_cache(None)
def atomic_tensors():
    """Cycling-normalized <exc|d_q|ground>; q order (-1,0,+1)."""
    jg, je, spin = Rational(1,2), Rational(3,2), Rational(3,2)
    d = np.zeros((3,16,8))
    for e, (fe, me) in enumerate(zip(EF.tolist(), EM.tolist())):
        for g, (fg, mg) in enumerate(zip(GF.tolist(), GM.tolist())):
            q = me - mg
            if abs(q) > 1 or abs(fe-fg)>1:
                continue
            d[q+1,e,g] = ((-1)**(fe-me+3+fg+1) * np.sqrt((2*fe+1)*(2*fg+1))
                          * float(wigner_3j(fe,1,fg,-me,q,mg)*wigner_6j(je,fe,spin,fg,jg,1)))
    d /= abs(d[2,-1,-1])
    if not np.allclose(np.einsum("qeg,qeg->e", d,d),1,rtol=0,atol=2e-14):
        raise ArithmeticError("D2 spontaneous strengths are not normalized")
    return d


def _lande(f,j,gj):
    f=np.asarray(f, float)
    a = np.divide(f*(f+1)+j*(j+1)-1.5*2.5, 2*f*(f+1),
                  out=np.zeros_like(f), where=f>0)
    return gj*a - .0009951414*(1-a)


def _kron(a,b):
    return sp.kron(a,b,format="csr")


def _comm(h):
    # Row-major flatten: (h rho)[i,j] - (rho h)[i,j].
    h=sp.csr_matrix(h); eye=sp.eye(24,format="csr")
    return -1j*(_kron(h,eye)-_kron(eye,h.T))


@lru_cache(32)
def _design(transfer_uW, dark_uW, length_mm, detuning_MHz, pi_fraction,
            polarization_phase_rad, grating_order, secular_hyperfine, bias_G,
            transfer_pi, dark_pi, group_phase_rad):
    # Time variable Gamma*t. Uniform-B linear Zeeman approximation; g_I included.
    d=atomic_tensors()
    gz=_lande(GF,.5,2.00233113)*GM*_MU*bias_G/GAMMA
    ez=_lande(EF,1.5,1.3362)*EM*_MU*bias_G/GAMMA
    hdiag=np.r_[gz+(_EHFS[GF-1]+2*np.pi*detuning_MHz*1e6)/GAMMA,
                ez+_EHFS[EF]/GAMMA]
    decay=np.r_[np.zeros(8),np.ones(16)]
    l0=sp.diags((-1j*(hdiag[:,None]-hdiag[None,:])-.5*(decay[:,None]+decay[None,:])).ravel(),
                format="csr")
    for q in range(3):
        for fe in range(4):
            for fg in (1,2):
                jump=np.zeros((24,24))
                jump[:8,8:] = d[q].T * (GF[:,None]==fg) * (EF[None,:]==fe)
                if np.any(jump):
                    l0=l0+_kron(sp.csr_matrix(jump),sp.csr_matrix(jump))
    # Each colour and each direction has half its total power. Spherical
    # field components for Ey (equal imaginary amplitudes) plus complex Ez.
    # Absorption weights are (-1)^q E_{-q}; overall sign of Ey is immaterial
    # for pure Ey, but retained here for general elliptic polarization.
    wz=length_mm*1e-3/4; wy=.00625
    hp=np.zeros((24,24),complex)
    hm=np.zeros_like(hp)
    for fg,pw,pif in ((1,dark_uW,dark_pi),(2,transfer_uW,transfer_pi)):
        pif=pi_fraction if pif is None else pif
        if not np.isfinite(pif) or not 0<=pif<=1:
            raise ValueError("Invalid pi-polarized power fraction")
        pol=np.array([1j*np.sqrt((1-pif)/2),np.sqrt(pif)*np.exp(1j*polarization_phase_rad),
                      1j*np.sqrt((1-pif)/2)])
        sat=pw*1e-6/(np.pi*wy*wz*ISAT)
        coupling=np.einsum("q,qeg->eg",pol,d) * (GF==fg)[None,:]*np.sqrt(sat/2)/2
        # H(theta)=Hplus exp(i theta)+Hminus exp(-i theta)
        offset=group_phase_rad if fg==2 else 0.
        hp[8:,:8] += coupling*np.exp(1j*offset)
        hm[8:,:8] += coupling*np.exp(-1j*offset)
    ap=hp + hm.conj().T
    plus=_comm(ap); minus=_comm(ap.conj().T)
    ii,jj=np.indices((24,24)); isopt=(ii<8)!=(jj<8)
    allowed=np.ones((24,24),bool)
    if secular_hyperfine:
        allowed[:8,:8] = GF[:,None]==GF[None,:]
        allowed[8:,8:] = EF[:,None]==EF[None,:]
        allowed[8:,:8] = np.abs(EF[:,None]-GF[None,:])<=1
        allowed[:8,8:] = allowed[8:,:8].T
    nmax=2*grating_order+1
    ns=np.arange(-nmax,nmax+1)
    ids=[np.flatnonzero(allowed & (isopt if n%2 else ~isopt)) for n in ns]
    chunks=[[None]*len(ns) for _ in ns]; light=[[None]*len(ns) for _ in ns]
    for i,g in enumerate(ids):
        chunks[i][i]=l0[g][:,g]
        light[i][i]=sp.csr_matrix((len(g),len(g)))
        if i: light[i][i-1]=plus[g][:,ids[i-1]]
        if i+1<len(ns): light[i][i+1]=minus[g][:,ids[i+1]]
    free=sp.bmat(chunks,format="csr")
    drive=sp.bmat(light,format="csr");drive.eliminate_zeros()
    ncol=np.repeat(ns, [len(g) for g in ids])
    start=sum(len(g) for g in ids[:nmax]); center=ids[nmax]
    diagat=start+np.searchsorted(center, np.arange(24)*25)
    counter=sp.csr_matrix((np.ones(16),(np.zeros(16,int),diagat[8:])),shape=(1,free.shape[0]))
    free=sp.bmat([[free,None],[counter,sp.csr_matrix((1,1))]],format="csc")
    drive=sp.block_diag((drive,sp.csr_matrix((1,1))),format="csc")
    return free,drive,np.r_[ncol,0],start,center,gz,d


def propagate_ensemble(source, *, transfer_uW, dark_uW, length_mm, detuning_MHz,
                       pi_fraction=0., polarization_phase_rad=0., slices=48,
                       grating_order=0, secular_hyperfine=True, bias_G=2.05,
                       transfer_pi=None, dark_pi=None, group_phase_rad=0.):
    """Return per-source-ID target population, spontaneous photon number and ground populations.

    Input is a MOT-exit Ensemble; retained same-F coherences precess through a
    uniform z bias field from that plane to the pump entrance. Inter-F source
    coherence is averaged over the unknown relative optical phase. Integration
    uses an L-stable second-order SDIRK step with midpoint Gaussian *field*
    held over each of 'slices' equal flight segments. Increase slices and the
    Fourier grating_order independently. For spatial order zero the directions
    interact via the shared density (no initial standing-wave gratings).
    """
    vals=(transfer_uW,dark_uW,length_mm,detuning_MHz,pi_fraction,polarization_phase_rad,bias_G,group_phase_rad)
    if not np.isfinite(vals).all() or min(transfer_uW,dark_uW)<0 or not 0<=pi_fraction<=1:
        raise ValueError("Invalid light settings")
    if not 0<length_mm<28 or type(slices) is not int or slices<2 or type(grating_order) is not int or grating_order<0:
        raise ValueError("Invalid interaction length or integration grid")
    free,drive,orders,offset,center,gz,d=_design(*vals[:6],grating_order,secular_hyperfine,bias_G,
                                                transfer_pi,dark_pi,group_phase_rad)
    nr=len(source.r_m); populations=np.empty((nr,8)); photons=np.empty(nr); errors=np.empty(nr)
    excited_peak=np.zeros(nr); min_eig=np.zeros(nr)
    wz=length_mm*1e-3/4; full=4*wz; entry=.014-full/2; wy=.00625
    if source.ground_density is None or entry<=source.reference_plane_z_m:
        raise ValueError("Need stateful MOT-exit ensemble upstream of pump")
    g=1-1/np.sqrt(2); identity=sp.eye(free.shape[0],format="csc")
    jf=np.einsum("qeg,qeg->ge",d,d)
    for idx,(r,v,rho) in enumerate(zip(source.r_m,source.v_m_s,source.ground_density)):
        if v[2]<=0: raise ValueError("Forward atoms required")
        t0=(entry-source.reference_plane_z_m)/v[2]
        dt=full/v[2]/slices; step=dt*GAMMA
        vel_free=free+sp.diags(-1j*orders*K*v[0]/GAMMA,format="csc")
        state=np.zeros(free.shape[0],complex)
        rho0=np.zeros((24,24),complex)
        rho0[:8,:8]=rho*np.exp(-1j*(gz[:,None]-gz[None,:])*t0*GAMMA)*(GF[:,None]==GF[None,:])
        state[offset:offset+len(center)]=rho0.ravel()[center]
        t=t0+(np.arange(slices)+.5)*dt
        ys=r[1]+v[1]*t-.5*9.80665*t*t
        u=4*((np.arange(slices)+.5)/slices-.5)
        fields=np.exp(-u*u-(ys/wy)**2)
        diagonal=offset+np.searchsorted(center,np.arange(24)*25)
        # For each frozen field: R(z)=(1+(1-2g)z)/(1-gz)^2; exact trace
        # and count integral under the same second-order rational update.
        const=(identity-g*step*vel_free).tocsc(); variable=(-g*step*drive).tocsc()
        for fld in fields:
            fac=splu(const+fld*variable)
            y=fac.solve(state)
            state=fac.solve(y/g-y+(2-1/g)*state)
            # R above: y2=(1/g-1) A^-2 y0 + (2-1/g) A^-1 y0.
            excited_peak[idx]=max(excited_peak[idx],state[diagonal[8:]].real.sum())
        out=np.zeros((24,24),complex)
        out.ravel()[center]=state[offset:offset+len(center)]
        pop=np.diag(out).real
        populations[idx]=pop[:8]+jf@pop[8:]
        errors[idx]=abs(pop.sum()-1)
        photons[idx]=float(state[-1].real+pop[8:].sum())
        min_eig[idx]=np.linalg.eigvalsh((out+out.conj().T)/2).min()
    if np.max(errors)>1e-7 or populations.min() < -1e-6 or photons.min() < -1e-6:
        raise ArithmeticError("Unphysical OBE outcome; refine space/time grid")
    return dict(target_population=populations[:,1], ground_populations=populations,
                scattered_photons=photons, trace_error=errors,
                peak_excited_population=excited_peak, min_exit_eigenvalue=min_eig)
