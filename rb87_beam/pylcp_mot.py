"""Native PyLCP 24-state 87Rb D2 2D+ MOT with repump and FM sidebands.

Rate-equation populations follow the local magnetic quantization axis (PyLCP's
adiabatic-spin convention). Linear Zeeman shifts retain F,m labels at MOT fields;
the physical hyperfine intervals and all allowed D2 dipole elements are retained.
"""

import time

import numpy as np
import pylcp
from scipy.special import jv

from .physics import Ensemble, GAMMA, HBAR, K, KB, MASS, MU_B, disk_positions
from .states import GROUND_LABELS, EXCITED_LABELS, rotation_to_lab_z
from .pylcp_atom import d2_atom


def fm_spectrum(frequency_Hz, index, order):
    """Phase-modulation powers J_n(beta)^2. Truncated tail is NOT renormalized."""
    if index == 0:
        return np.array([0.0]), np.array([1.0])
    n = np.arange(-order, order + 1)
    return n * 2 * np.pi * frequency_Hz / GAMMA, jv(n, index) ** 2


class MOTBeam(pylcp.fields.laserBeam):
    def __init__(self, axis, sign, helicity, saturation, delta, source, scales, hole_edge_m):
        self.axis, self.sign = axis, sign
        self.peak = saturation
        self.source, self.x0, self.hole_edge_m = source, scales[0], hole_edge_m
        kvec = np.eye(3)[axis] * sign
        # Passing s here would replace the subclass intensity() with a constant
        # instance lambda inside PyLCP, silently removing the Gaussian and hole.
        super().__init__(kvec=kvec, s=None, pol=helicity, delta=delta)
        self.intensity_sig = "(R)"

    def intensity(self, R=np.zeros(3), t=0.0):
        r = np.asarray(R) * self.x0
        s = self.source
        if self.axis < 2:
            other = 1 - self.axis
            profile = np.exp(-2 * (r[other] / s.cooling_waist_m) ** 2
                             -2 * ((r[2] - s.mot_length_m / 2) / s.mot_length_m) ** 2)
        else:
            radial = np.sqrt(r[0] ** 2 + r[1] ** 2)
            profile = np.exp(-2 * (radial / s.cooling_waist_m) ** 2)
            if self.sign < 0:
                if self.hole_edge_m == 0:
                    profile *= radial >= s.exit_radius_m
                else:
                    u = np.clip((radial - s.exit_radius_m + self.hole_edge_m) / (2 * self.hole_edge_m), 0, 1)
                    profile *= u * u * (3 - 2 * u)
        return self.peak * profile


class MOT2DPlus:
    """SI interface to the existing project's PyLCP rate-equation construction."""

    def __init__(self, source, options):
        self.source, self.options = source, options
        self.x0, self.t0 = 1 / K, 1 / GAMMA
        self.v0 = self.x0 / self.t0
        self.mass_scaled = MASS * self.x0 ** 2 / (HBAR * self.t0)
        (self.hamiltonian, self.dipole, self.branching,
         self.ground_energies, self.excited_energies) = d2_atom()
        self.beams = pylcp.laserBeams()
        self.beam_records = []
        cooling = self.excited_energies[EXCITED_LABELS[:, 0] == 3][0] - self.ground_energies[GROUND_LABELS[:, 0] == 2][0]
        repump = self.excited_energies[EXCITED_LABELS[:, 0] == options.repump_Fprime][0] - self.ground_energies[GROUND_LABELS[:, 0] == 1][0]
        self.spectra = {}
        for name, freq, beta, manual, powers in (
                ("cooling", options.cooling_fm_frequency_Hz, options.cooling_fm_index,
                 source.cooling_spectrum_gamma, source.cooling_spectrum_weights),
                ("axial", options.axial_fm_frequency_Hz, options.axial_fm_index,
                 source.axial_spectrum_gamma, source.axial_spectrum_weights),
                ("repump", options.repump_fm_frequency_Hz, options.repump_fm_index, (0,), (1,))):
            offsets, weights = fm_spectrum(freq, beta, options.sideband_order) if beta > 0 else (np.asarray(manual), np.asarray(powers))
            self.spectra[name] = {"offset_gamma": offsets.tolist(), "power_fractions": weights.tolist(),
                                  "retained_power_fraction": float(weights.sum())}
        for axis, sign, helicity in ((0, 1, 1), (0, -1, 1), (1, 1, -1),
                                     (1, -1, -1), (2, 1, -1), (2, -1, -1)):
            family = "cooling" if axis < 2 else "axial"
            cooling_s = source.cooling_saturation if axis < 2 else (source.axial_forward_saturation if sign > 0 else source.axial_backward_saturation)
            det = source.cooling_detuning_gamma if axis < 2 else source.axial_detuning_gamma
            repump_s = options.repump_saturation if axis < 2 else options.repump_axial_saturation
            for label, center, saturation, spectral in (
                    (family, cooling + det, cooling_s, self.spectra[family]),
                    ("repump", repump + options.repump_detuning_gamma, repump_s, self.spectra["repump"])):
                for shift, power in zip(spectral["offset_gamma"], spectral["power_fractions"]):
                    if saturation * power <= 1e-15:
                        continue
                    beam = MOTBeam(axis, sign, helicity, saturation * power, center + shift,
                                   source, (self.x0, self.t0), options.hole_edge_m)
                    self.beams.add_laser(beam)
                    self.beam_records.append({"family": label, "axis": axis, "sign": sign,
                                              "power_saturation": float(saturation * power),
                                              "offset_gamma": float(shift), "detuning_scaled": float(center + shift)})
        # Divergence-free 2D quadrupole; no longitudinal gradient.
        self.mag_field = pylcp.magField(lambda R: self.field_T(R * self.x0))
        self.eqn = pylcp.rateeq(self.beams, self.mag_field, self.hamiltonian,
                                a=np.asarray(options.gravity_m_s2) * self.t0 ** 2 / self.x0,
                                include_mag_forces=options.include_magnetic_force)

    def field_T(self, r_m):
        return (self.source.magnetic_gradient_T_m * np.asarray(r_m) * np.array([1, -1, 0])
                + np.asarray(self.options.bias_field_T))

    def initial_populations(self):
        p = self.options
        ground = (np.asarray(p.initial_ground_populations, dtype=float) if len(p.initial_ground_populations)
                  else np.r_[np.full(3, (1 - p.initial_ground_F2_fraction) / 3),
                              np.full(5, p.initial_ground_F2_fraction / 5)])
        return np.r_[ground, np.zeros(16)]

    def ground_after_decay(self, populations, r_m):
        ground = populations[:8] + self.branching @ populations[8:]
        field = self.field_T(r_m)
        axis = field / np.linalg.norm(field) if np.linalg.norm(field) > 1e-15 else np.array([0., 0., 1.])
        rotation = rotation_to_lab_z(axis)
        rho = (rotation * ground[None, :]) @ rotation.conj().T
        return rho, axis

    def _checked_population(self, population):
        population = np.asarray(population, dtype=float)
        if np.min(population) < -1e-7 or abs(population.sum() - 1) > 1e-6:
            raise RuntimeError("PyLCP populations lost positivity/normalization; reduce integration tolerances")
        population = np.maximum(population, 0)
        return population / population.sum()

    def _diffusion_kick(self, sol, rng):
        """Operator splitting with directional absorption/stimulated and isotropic emission."""
        duration = sol.t[-1] - sol.t[0]
        if duration <= 0:
            return np.zeros(3)
        mid = sol.sol((sol.t[-1] + sol.t[0]) / 2)
        population = self._checked_population(mid[:-6])
        _, rates = self.eqn.construct_evolution_matrix(mid[-3:], mid[-6:-3])
        absorption = np.sum(rates["g->e"] * (population[:8][None, :, None]
                             + population[8:][None, None, :]), axis=(1, 2))
        directions = np.asarray([b.kvec(mid[-3:], 0) for b in self.beams.beam_vector])
        absorption_variance = (absorption[:, None] * directions ** 2).sum(axis=0) * duration
        spontaneous = np.trapz(self.eqn.decay_rates["g->e"] @ sol.N[8:], sol.t)
        return rng.normal(size=3) * np.sqrt(np.maximum(absorption_variance + spontaneous / 3, 0)) / self.mass_scaled

    def evolve_atom(self, r_m, v_m_s, populations, rng):
        s, p = self.source, self.options
        self.eqn.r0, self.eqn.v0 = r_m / self.x0, v_m_s / self.v0
        self.eqn.N0 = self._checked_population(populations)
        end_time = s.max_time_s / self.t0
        chunk = p.recoil_chunk_s / self.t0 if p.recoil_model == "diffusion" else end_time
        current = 0.0
        records = []

        def exit_event(t, y):
            return y[-1] - s.mot_length_m / self.x0
        exit_event.terminal, exit_event.direction = True, 1

        def wall_event(t, y):
            return (s.chamber_radius_m / self.x0) ** 2 - np.sum(y[-3:-1] ** 2)
        wall_event.terminal, wall_event.direction = True, -1

        def backward_event(t, y):
            return y[-1]
        backward_event.terminal, backward_event.direction = True, -1

        status = "timeout"
        atol = np.r_[np.full(24, p.population_atol), np.full(3, 1e-7), np.full(3, 1e-2)]
        while current < end_time:
            sol = self.eqn.evolve_motion((current, min(current + chunk, end_time)), method="BDF",
                rtol=p.rtol, atol=atol, max_step=p.max_motion_step_s / self.t0,
                dense_output=True, events=(exit_event, wall_event, backward_event), random_recoil=False)
            if not sol.success:
                raise RuntimeError("PyLCP integration failed: " + sol.message)
            records.append((sol.t.copy() * self.t0, sol.v.copy().T * self.v0, sol.N.copy().T))
            final_pop = self._checked_population(sol.N[:, -1])
            final_r, final_v = sol.r[:, -1].copy(), sol.v[:, -1].copy()
            if p.recoil_model == "diffusion":
                final_v += self._diffusion_kick(sol, rng)
            self.eqn.r0, self.eqn.v0, self.eqn.N0 = final_r, final_v, final_pop
            current = float(sol.t[-1])
            if sol.status == 1:
                if len(sol.t_events[0]):
                    radius = np.linalg.norm(final_r[:2]) * self.x0
                    status = "extracted" if radius < s.exit_radius_m and final_v[2] > 0 else "missed_aperture"
                else:
                    status = "wall" if len(sol.t_events[1]) else "backward"
                break
        return {"r_m": final_r * self.x0, "v_m_s": final_v * self.v0, "populations": final_pop,
                "time_s": current * self.t0, "status": status,
                "record": tuple(np.concatenate([record[i] for record in records], axis=0) for i in range(3))}

    def simulate(self, rng, progress=None):
        s = self.source
        initial_r = disk_positions(rng, s.n_atoms, s.initial_radius_m)
        initial_r[:, 2] = self.options.initial_z_m
        initial_v = rng.normal(0, np.sqrt(KB * s.initial_temperature_K / MASS), (s.n_atoms, 3))
        initial_v[:, 2] = rng.normal(s.initial_vz_m_s, s.initial_sigma_vz_m_s, s.n_atoms)
        atoms = []
        began = time.perf_counter()
        for i in range(s.n_atoms):
            atoms.append(self.evolve_atom(initial_r[i], initial_v[i], self.initial_populations(), rng))
            if progress:
                progress(i + 1, s.n_atoms, sum(a["status"] == "extracted" for a in atoms))
        extracted = [a for a in atoms if a["status"] == "extracted"]
        if not extracted:
            raise ValueError("No PyLCP atoms extracted; inspect pushing force, aperture and maximum flight time")
        positions, velocities = np.asarray([a["r_m"] for a in extracted]), np.asarray([a["v_m_s"] for a in extracted])
        density, axes = zip(*(self.ground_after_decay(a["populations"], a["r_m"]) for a in extracted))
        positions[:, 2] = 0
        statuses = {name: sum(a["status"] == name for a in atoms)
                    for name in ("extracted", "wall", "backward", "missed_aperture", "timeout")}
        meta = {"mode": "pylcp", "sampled": s.n_atoms, "extracted": len(extracted),
                "status_counts": statuses, "extraction_fraction": len(extracted) / s.n_atoms,
                "absolute_flux_is_calibrated": False, "elapsed_s": time.perf_counter() - began,
                "model": "native pylcp.rateeq, 8 ground + 16 excited, physical hyperfine splittings, linear Zeeman",
                "spin_transport": "adiabatic local magnetic axis; sudden field switch to lab-z at exit",
                "recoil_model": self.options.recoil_model, "spectra": self.spectra,
                "beam_count": len(self.beam_records), "length_unit_m": self.x0, "time_unit_s": self.t0}
        trace, state_trace = self._ensemble_trace(atoms)
        ensemble = Ensemble(positions, velocities, s.incident_flux_per_s * len(extracted) / s.n_atoms,
            meta, ground_density=np.asarray(density), exit_populations=np.asarray([a["populations"] for a in extracted]),
            exit_time_s=np.asarray([a["time_s"] for a in extracted]), quantization_axes=np.asarray(axes),
            population_trace=state_trace)
        ensemble.validate()
        return ensemble, trace

    def _ensemble_trace(self, atoms):
        times = np.linspace(0, max(a["time_s"] for a in atoms), 101)
        trace, states = [], []
        for t in times:
            active = [a for a in atoms if t < a["time_s"]]
            extracted = sum(a["status"] == "extracted" and a["time_s"] <= t for a in atoms)
            velocity = [np.array([np.interp(t, a["record"][0], a["record"][1][:, j]) for j in range(3)]) for a in active]
            populations = [np.array([np.interp(t, a["record"][0], a["record"][2][:, j]) for j in range(24)]) for a in active]
            temperature = MASS / KB * np.mean(np.var(np.asarray(velocity)[:, :2], axis=0)) if len(active) >= 20 else np.nan
            mean_vz = float(np.mean(np.asarray(velocity)[:, 2])) if active else np.nan
            mean_population = np.mean(populations, axis=0) if active else np.full(24, np.nan)
            trace.append([t, len(active), extracted, temperature, mean_vz])
            states.append(np.r_[t, len(active), mean_population])
        return np.asarray(trace), np.asarray(states)
