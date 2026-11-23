"""Shared, uncompressed 87Rb D2 Hamiltonian in Gamma, 1/k units."""

import numpy as np
import pylcp

from .physics import GAMMA, HBAR, K, MASS, MU_B
from .states import GROUND_LABELS, EXCITED_LABELS


def d2_atom():
    atom = pylcp.atom("87Rb")
    g, e = atom.state[0], atom.state[2]
    hg, mug, bg = pylcp.hamiltonians.hyperfine_coupled(
        g.J, atom.I, g.gJ, atom.gI, g.Ahfs * 2 * np.pi / GAMMA,
        muB=MU_B / HBAR / GAMMA, return_basis=True)
    he, mue, be = pylcp.hamiltonians.hyperfine_coupled(
        e.J, atom.I, e.gJ, atom.gI, e.Ahfs * 2 * np.pi / GAMMA,
        Bhfs=e.Bhfs * 2 * np.pi / GAMMA, muB=MU_B / HBAR / GAMMA, return_basis=True)
    if not np.array_equal(bg.T, GROUND_LABELS) or not np.array_equal(be.T, EXCITED_LABELS):
        raise ValueError("Unexpected PyLCP D2 basis order")
    for magnetic, basis in ((mug, bg), (mue, be)):
        magnetic[:, basis[0, :, None] != basis[0, None, :]] = 0
    dipole = pylcp.hamiltonians.dqij_two_hyperfine_manifolds(g.J, e.J, atom.I)
    branching = np.sum(abs(dipole) ** 2, axis=0)
    branching /= branching.sum(axis=0)
    mass = MASS / (HBAR * K ** 2 / GAMMA)
    hamiltonian = pylcp.hamiltonian(hg, he, mug, mue, dipole, mass=mass)
    return hamiltonian, dipole, branching, np.diag(hg), np.diag(he)
