"""Canonical fixed-laboratory-z Rb87 ground basis and state transfer checks."""

import numpy as np

GROUND_LABELS = np.array([(f, m) for f in (1, 2) for m in range(-f, f + 1)], dtype=int)
EXCITED_LABELS = np.array([(f, m) for f in (0, 1, 2, 3) for m in range(-f, f + 1)], dtype=int)
G_CLOCK = int(np.flatnonzero(np.all(GROUND_LABELS == (1, 0), axis=1))[0])
E_CLOCK = int(np.flatnonzero(np.all(GROUND_LABELS == (2, 0), axis=1))[0])


def validate_density(rho, n):
    rho = np.asarray(rho)
    if rho.shape != (n, 8, 8) or not np.all(np.isfinite(rho)):
        raise ValueError("ground_density must have finite shape (N,8,8)")
    if not np.allclose(rho, rho.conj().swapaxes(-1, -2), atol=1e-9):
        raise ValueError("Ground density matrices must be Hermitian")
    if not np.allclose(np.trace(rho, axis1=1, axis2=2), 1, atol=1e-8):
        raise ValueError("Ground density matrices must have unit trace")
    if np.min(np.linalg.eigvalsh(rho)) < -1e-8:
        raise ValueError("Ground density matrices must be positive semidefinite")


def rotation_to_lab_z(axis):
    """D(Rz(phi) Ry(theta)): local quantization z is rotated to the laboratory axis."""
    from scipy.linalg import expm

    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    theta, phi = np.arccos(np.clip(axis[2], -1, 1)), np.arctan2(axis[1], axis[0])
    rotation = np.zeros((8, 8), dtype=complex)
    for f in (1, 2):
        ids = np.flatnonzero(GROUND_LABELS[:, 0] == f)
        ms = GROUND_LABELS[ids, 1]
        plus = np.diag(np.sqrt(f * (f + 1) - ms[:-1] * (ms[:-1] + 1)), -1)
        jy = (plus - plus.T) / (2j)
        rotation[np.ix_(ids, ids)] = np.diag(np.exp(-1j * phi * ms)) @ expm(-1j * theta * jy)
    return rotation


def resolved_state_handling(ensemble, raman):
    mode = raman.state_handling
    if mode == "auto":
        mode = "inherit" if ensemble.ground_density is not None else "ideal_preparation"
    if mode in ("inherit", "project_F1_m0") and ensemble.ground_density is None:
        raise ValueError(mode + " requires source ground-state density matrices")
    return mode


def raman_frame_density(ensemble, raman):
    """Density at the Raman entrance in the model's quantization frame.

    Input checkpoints retain their physical laboratory-z basis at the pumping
    exit. In power mode an explicitly requested, ideal adiabatic rotation first
    rotates the physical spin, then changes coordinates into the Raman frame.
    These operations preserve all F,m populations, rather than projecting onto
    m=0. Transport losses and additional accumulated Zeeman phases are omitted.
    The current native pumping ensemble has no ground-state coherences.
    """
    rho = ensemble.ground_density
    if rho is None or raman.coupling_model == "effective":
        return rho
    if raman.spin_transport != "adiabatic_from_z":
        raise ValueError("Explicit adiabatic spin-transport assumption is required")
    rotation = rotation_to_lab_z(raman.quantization_axis)
    transported_lab = rotation @ rho @ rotation.conj().T
    return rotation.conj().T @ transported_lab @ rotation
