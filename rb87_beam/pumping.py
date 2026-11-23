"""Finite-flight, 24-level PyLCP population optical pumping, not a state reset.

F2->F'1 transfers hyperfine population; F1->F'0 sigma+/- light leaves
F1,m=0 dark on resonance. All off-resonant D2 transitions are retained.
The secular population model assumes ground coherences average out in the
bias field/laser phase ensemble; it does not describe coherent population trapping.
"""

from dataclasses import asdict

import numpy as np
import pylcp

from .physics import Ensemble, GAMMA, HBAR, K, MASS, flight_time, trajectory
from .pylcp_atom import d2_atom
from .states import GROUND_LABELS, EXCITED_LABELS, G_CLOCK

PUMP_ALGORITHM_VERSION = "native-rateeq-quadrature-recoil-v2"


def recoil_noise(rng, absorption, stimulated, directions, spontaneous, mass_scaled, model):
    """Zero-mean optical recoil; compound Poisson is not an internal-state jump solver."""
    if model == "diffusion":
        variance = ((absorption + stimulated)[:, None] * directions ** 2).sum(axis=0)
        return rng.normal(size=3) * np.sqrt(np.maximum(variance + spontaneous / 3, 0)) / mass_scaled
    if model != "compound_poisson":
        raise ValueError("Unknown stochastic pumping recoil model")
    counts = (rng.poisson(absorption) - absorption
              - rng.poisson(stimulated) + stimulated)
    kick = counts @ directions
    emissions = rng.poisson(spontaneous)
    if emissions:
        unit = rng.normal(size=(emissions, 3))
        unit /= np.linalg.norm(unit, axis=1)[:, None]
        kick += unit.sum(axis=0)
    return kick / mass_scaled


class FixedAxisRateEquation(pylcp.rateeq):
    """Same PyLCP generator and inherited motion solver with constant-B coefficients cached.

    Applicable only to this module's static z field, four static-k beams and
    common Gaussian envelope. No level elimination or equilibrium approximation.
    """

    def __init__(self, beams, field, hamiltonian, options, **kwargs):
        self._ready = False
        super().__init__(beams, field, hamiltonian, **kwargs)
        self.options = options
        center = np.array([0., 0., options.center_z_m]) * K
        _, rates = super().construct_evolution_matrix(center, np.zeros(3))
        ham = self.hamiltonian.rotated_hamiltonian
        index = ham.laser_keys["g->e"]
        eg = np.diag(ham.blocks[index[0], index[0]].matrix)
        ee = np.diag(ham.blocks[index[1], index[1]].matrix)
        self._directions = np.array([beam.kvec(center, 0) for beam in beams.beam_vector])
        delta = np.array([beam.delta(0) for beam in beams.beam_vector])
        self._detuning = delta[:, None, None] - (ee[None, :] - eg[:, None])[None, :, :]
        self._gamma = self.hamiltonian.blocks[index].parameters["gamma"]
        self._numerator = rates["g->e"] * (1 + 4 * self._detuning ** 2 / self._gamma ** 2)
        self._ready = True

    def construct_evolution_matrix(self, r, v, t=0., default_axis=np.array([0., 0., 1.])):
        if not self._ready:
            return super().construct_evolution_matrix(r, v, t, default_axis)
        p = self.options
        yy, zz = np.asarray(r)[1:] / K
        profile = np.exp(-2 * ((yy / p.waist_y_m) ** 2 + ((zz - p.center_z_m) / p.waist_z_m) ** 2))
        detuning = self._detuning - (self._directions @ v)[:, None, None]
        rates = profile * self._numerator / (1 + 4 * detuning ** 2 / self._gamma ** 2)
        self.Rijl = {"g->e": rates}
        total = rates.sum(axis=0)
        q = self.Rev_decay.copy()
        q[:8, 8:] += total
        q[8:, :8] += total.T
        q[np.arange(8), np.arange(8)] -= total.sum(axis=1)
        q[8 + np.arange(16), 8 + np.arange(16)] -= total.sum(axis=0)
        self.Rev = q
        return q, self.Rijl

    def force(self, r, t, N, return_details=True):
        net = np.sum(self.Rijl["g->e"] * (N[:8][None, :, None] - N[8:][None, None, :]), axis=(1, 2))
        per_beam = self._directions.T * net[None, :]
        force = per_beam.sum(axis=1)
        return (force, {"g->e": per_beam}, np.zeros(3)) if return_details else force


class PumpBeam(pylcp.fields.laserBeam):
    def __init__(self, sign, peak, delta, options):
        self.peak, self.options = peak, options
        # Both components are transverse to k=x. E_y gives equal sigma+/-;
        # E_z is unwanted pi power relative to the fixed B=z axis.
        polarization = np.array([0., np.sqrt(1 - options.pi_power_fraction),
                                 np.sqrt(options.pi_power_fraction)])
        super().__init__(kvec=np.array([sign, 0., 0.]), s=None,
                         pol=polarization, pol_coord="cartesian", delta=delta)
        self.intensity_sig = "(R)"

    def intensity(self, R=np.zeros(3), t=0):
        p = self.options
        y, z = np.asarray(R)[1:] / K
        u = (z - p.center_z_m) / p.waist_z_m
        return self.peak * np.exp(-2 * (u ** 2 + (y / p.waist_y_m) ** 2))


class OpticalPumping:
    def __init__(self, options, raman):
        self.options, self.raman = options, raman
        self.x0, self.t0, self.v0 = 1 / K, 1 / GAMMA, GAMMA / K
        self.mass_scaled = MASS * GAMMA / (HBAR * K ** 2)
        (self.hamiltonian, self.dipole, self.branching,
         self.ground_energies, self.excited_energies) = d2_atom()
        self.beams = pylcp.laserBeams()
        for f, fp, power, detuning in (
                (2, 1, options.transfer_power_W, options.transfer_detuning_Hz),
                (1, 0, options.dark_power_W, options.dark_detuning_Hz)):
            center = (self.excited_energies[EXCITED_LABELS[:, 0] == fp][0]
                      - self.ground_energies[GROUND_LABELS[:, 0] == f][0])
            peak = 2 * power / (np.pi * options.waist_y_m * options.waist_z_m
                                * options.saturation_intensity_W_m2)
            for sign, fraction in ((1, options.forward_power_fraction),
                                   (-1, 1 - options.forward_power_fraction)):
                # Retain zero-intensity beams: PyLCP needs a nonempty beam container.
                self.beams.add_laser(PumpBeam(sign, peak * fraction,
                    center + 2 * np.pi * detuning / GAMMA, options))
        field = pylcp.magField(np.array([0., 0., options.bias_field_T]))
        kwargs = dict(a=np.asarray(raman.acceleration_m_s2) * self.t0 ** 2 / self.x0,
                      include_mag_forces=False)
        self.eqn = (FixedAxisRateEquation(self.beams, field, self.hamiltonian, options, **kwargs)
                    if options.rate_backend == "fixed_axis" else
                    pylcp.rateeq(self.beams, field, self.hamiltonian, **kwargs))
        self.start_z = options.center_z_m - options.half_widths * options.waist_z_m
        self.end_z = options.center_z_m + options.half_widths * options.waist_z_m
        self.quadrature = np.polynomial.legendre.leggauss(options.diffusion_quadrature_order)

    @staticmethod
    def checked(pop):
        if not np.all(np.isfinite(pop)) or np.min(pop) < -1e-7 or abs(pop.sum() - 1) > 1e-6:
            raise RuntimeError("Optical pumping populations lost positivity or normalization")
        return np.maximum(pop, 0) / np.maximum(pop, 0).sum()

    def evolve_atom(self, r, v, ground, rng):
        p = self.options
        self.eqn.r0, self.eqn.v0 = r / self.x0, v / self.v0
        self.eqn.N0 = np.r_[ground, np.zeros(16)]
        duration = (self.end_z - self.start_z) / v[2] / self.t0
        stop = duration * 2
        chunk = p.recoil_chunk_s / self.t0 if p.recoil_model != "none" else stop
        current, photons, history = 0.0, 0.0, []

        def end_event(t, y):
            return y[-1] - self.end_z / self.x0
        end_event.terminal, end_event.direction = True, 1

        while current < stop:
            sol = self.eqn.evolve_motion((current, min(current + chunk, stop)), method="BDF",
                atol=np.r_[np.full(24, p.population_atol), np.full(3, 1e-8), np.full(3, 1e-3)],
                rtol=p.rtol, max_step=p.max_step_s / self.t0,
                dense_output=True, events=end_event, random_recoil=False)
            if not sol.success:
                raise RuntimeError("Optical pumping integration failed: " + sol.message)
            population = self.checked(sol.N[:, -1])
            rr, vv = sol.r[:, -1].copy(), sol.v[:, -1].copy()
            spontaneous = max(float(np.trapz(self.eqn.decay_rates["g->e"] @ sol.N[8:], sol.t)), 0)
            photons += spontaneous
            history.append(np.column_stack((sol.r[2] * self.x0, sol.N.T)))
            if p.recoil_model != "none":
                midpoint = (sol.t[0] + sol.t[-1]) / 2
                half_duration = (sol.t[-1] - sol.t[0]) / 2
                absorption = np.zeros(len(self.beams.beam_vector))
                stimulated = np.zeros_like(absorption)
                for node, weight in zip(*self.quadrature):
                    mid = sol.sol(midpoint + half_duration * node)
                    _, rates = self.eqn.construct_evolution_matrix(mid[-3:], mid[-6:-3])
                    pop_mid = self.checked(mid[:-6])
                    absorption += half_duration * weight * np.sum(
                        rates["g->e"] * pop_mid[:8][None, :, None], axis=(1, 2))
                    stimulated += half_duration * weight * np.sum(
                        rates["g->e"] * pop_mid[8:][None, None, :], axis=(1, 2))
                directions = np.array([b.kvec(mid[-3:], 0) for b in self.beams.beam_vector])
                vv += recoil_noise(rng, absorption, stimulated, directions, spontaneous,
                                   self.mass_scaled, p.recoil_model)
            self.eqn.N0, self.eqn.r0, self.eqn.v0 = population, rr, vv
            current = float(sol.t[-1])
            if vv[2] <= 0:
                raise RuntimeError("Pumping reversed an atom; forward-beam approximation no longer applies")
            if sol.status == 1:
                break
        else:
            raise RuntimeError("Atom did not reach the pumping exit")
        # Trace over photons emitted after the light is shut off. Include their recoil.
        residual = population[8:].sum()
        ground = population[:8] + self.branching @ population[8:]
        if p.recoil_model == "diffusion":
            vv += rng.normal(size=3) * np.sqrt(residual / 3) / self.mass_scaled
        elif p.recoil_model == "compound_poisson":
            if rng.random() < residual:
                direction = rng.normal(size=3)
                vv += direction / np.linalg.norm(direction) / self.mass_scaled
        rr[2] = self.end_z / self.x0
        return rr * self.x0, vv * self.v0, ground, current * self.t0, photons + residual, np.concatenate(history)

    def prepare(self, ensemble, rng, progress=None):
        ensemble.validate()
        if ensemble.ground_density is None:
            raise ValueError("Optical pumping needs the actual eight-state source density matrices")
        if ensemble.reference_plane_z_m >= self.start_z:
            raise ValueError("Input ensemble must precede the pumping region; use a MOT checkpoint")
        if np.linalg.norm(self.raman.rotation_rad_s) > 0:
            raise ValueError("Pumping motion currently supports gravity but not rotation; set rotation_rad_s=0")
        p = self.options
        times = flight_time(ensemble, self.start_z)
        r0, v0 = trajectory(ensemble, times, self.raman)
        initial = np.real(np.diagonal(ensemble.ground_density, axis1=1, axis2=2))
        n = len(r0)
        rr, vv, populations, photons, durations, histories = [], [], [], [], [], []
        for i in range(n):
            r, v, pop, dt, scat, history = self.evolve_atom(r0[i], v0[i], initial[i], rng)
            rr.append(r); vv.append(v); populations.append(pop)
            photons.append(scat); durations.append(dt); histories.append(history)
            if progress is not None:
                progress(i + 1, n, i + 1)
        populations = np.asarray(populations)
        rho = np.zeros_like(ensemble.ground_density, dtype=complex)
        rho[:, np.arange(8), np.arange(8)] = populations
        if p.transfer_power_W == 0 and p.dark_power_W == 0:
            # With no scattering do not apply the secular population projection.
            rho = ensemble.ground_density.copy()
        coherence = ensemble.ground_density.copy()
        coherence[:, np.arange(8), np.arange(8)] = 0
        diagnostics = {
            "model": "PyLCP 24-level secular rate equations, fixed lab-z bias",
            "laser_transitions": ["F2->Fprime1", "F1->Fprime0"],
            "polarization": "linear E_y (sigma+/-), optional E_z pi leakage; k=+/-x",
            "input_coherence_frobenius_mean": float(np.mean(np.linalg.norm(coherence, axis=(1, 2)))),
            "coherence_assumption": "Bias/phase ensemble averages ground coherences before population pumping",
            "preparation_efficiency": float(populations[:, G_CLOCK].mean()),
            "initial_target_fraction": float(initial[:, G_CLOCK].mean()),
            "target_flux_per_s": float(ensemble.flux_per_s * populations[:, G_CLOCK].mean()),
            "total_flux_per_s": ensemble.flux_per_s,
            "mean_scattered_photons": float(np.mean(photons)),
            "mean_interaction_time_s": float(np.mean(durations)),
            "max_population_norm_error": float(np.max(abs(populations.sum(axis=1) - 1))),
            "config": asdict(p),
        }
        metadata = dict(ensemble.source_metadata, optical_pumping=diagnostics)
        output = Ensemble(np.asarray(rr), np.asarray(vv), ensemble.flux_per_s, metadata,
            ground_density=rho, reference_plane_z_m=self.end_z,
            reference_time_s=times + np.asarray(durations))
        output.validate()
        grid = np.linspace(self.start_z, self.end_z, 121)
        trace = np.mean([np.column_stack([np.interp(grid, h[:, 0], h[:, j])
                                         for j in range(1, 25)]) for h in histories], axis=0)
        return {"ensemble": output, "diagnostics": diagnostics,
                "z_trace_m": grid, "population_trace": trace,
                "scattered_photons": np.asarray(photons), "interaction_time_s": np.asarray(durations),
                "initial_populations": initial, "final_populations": populations}
