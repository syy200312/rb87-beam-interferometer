"""All lengths in m, times in s, frequencies explicitly named Hz or rad/s."""

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np


@dataclass
class SourceConfig:
    mode: str = "mot"  # mot: conditional captured reservoir; calibrated: exit beam
    n_atoms: int = 2000
    # Assumed incoming flux of the sampled population, NOT a vapor-loading prediction.
    incident_flux_per_s: float = 1.0e10
    initial_temperature_K: float = 2.0e-3
    initial_radius_m: float = 1.0e-3
    initial_vz_m_s: float = 3.0
    initial_sigma_vz_m_s: float = 0.5
    mot_length_m: float = 0.04
    chamber_radius_m: float = 0.005
    exit_radius_m: float = 0.0005
    cooling_waist_m: float = 0.008
    cooling_saturation: float = 5.0  # per transverse beam
    cooling_detuning_gamma: float = -1.5  # Gamma = 2*pi*6.065 MHz
    magnetic_gradient_T_m: float = 0.10  # 10 G/cm
    effective_magnetic_moment_muB: float = 1.0
    axial_forward_saturation: float = 0.6
    axial_backward_saturation: float = 0.4
    axial_detuning_gamma: float = -1.0
    # Frequency offsets in Gamma units and fractional powers (sum = 1).
    cooling_spectrum_gamma: tuple = (0.0,)
    cooling_spectrum_weights: tuple = (1.0,)
    axial_spectrum_gamma: tuple = (0.0,)
    axial_spectrum_weights: tuple = (1.0,)
    dt_s: float = 2.0e-6
    max_time_s: float = 0.025
    # Empirical EXIT distribution (separate experiment, arXiv:2312.12027).
    calibrated_mean_vz_m_s: float = 10.96
    calibrated_fwhm_vz_m_s: float = 2.92
    calibrated_transverse_temperature_K: float = 16.90e-6
    calibrated_flux_per_s: float = 4.3e9
    calibrated_flux_is_measured: bool = True
    calibration_reference: str = "https://arxiv.org/abs/2312.12027"
    calibrated_point_source: bool = False


@dataclass
class RamanConfig:
    first_zone_z_m: float = 0.030
    zone_spacing_m: float = 0.01045  # illustrative: T=0.95 ms at 11 m/s
    waist_z_m: float = 0.00025  # matched-beam 1/e^2 INTENSITY radii
    waist_y_m: float = 0.006
    envelope: str = "gaussian"
    rectangular_width_m: float = 0.001  # thesis rectangular z-window uses tau=d/vz
    reference_vz_m_s: float = 11.0
    k_direction: tuple = (1.0, 0.0, 0.0)  # horizontal, perpendicular to atom beam
    acceleration_m_s2: tuple = (0.0, -9.80665, 0.0)
    rotation_rad_s: tuple = (0.0, 0.0, 0.0)
    two_photon_detuning_Hz: float = 0.0  # relative to clock + recoil + reference Doppler
    single_photon_detuning_Hz: float = 1.073e9  # power mode: laser 1 minus F=1 -> selected Fprime
    # Zero retains legacy area calibration. Positive: first/third-zone Omega/(2*pi),
    # with twice that peak coupling in the middle zone. Independent of zone width.
    peak_rabi_Hz: float = 0.0
    # Legacy configs remain effective-coupling models. Power mode uses optical
    # power of EACH frequency component at the atoms in the first/third zone.
    # The middle zone doubles both powers; all three have the same beam waists.
    coupling_model: str = "effective"  # effective or rb87_d2_power
    beam1_power_W: float = 0.0  # higher optical frequency, +k, addresses F=1
    beam2_power_W: float = 0.0  # lower optical frequency, -k, addresses F=2
    detuning_reference_Fprime: int = 1
    polarization1: str = "sigma_plus"  # relative to ONE common quantization axis
    polarization2: str = "sigma_plus"
    quantization_axis: tuple = (1.0, 0.0, 0.0)
    spin_transport: str = "none"  # power mode requires explicit adiabatic_from_z
    chirp_rad_s2: float = 0.0
    differential_light_shift_Hz: float = 0.0  # peak differential shift per zone
    phase_offsets_rad: tuple = (0.0, 0.0, 0.0)
    preparation_efficiency: float = 0.8  # unprepared atoms removed
    remove_unprepared: bool = True
    spectator_F2_fraction: float = 0.0  # used only when remove_unprepared=False
    # auto preserves provided ground-state matrices; old stateless sources retain ideal preparation.
    state_handling: str = "auto"  # auto, inherit, project_F1_m0, ideal_preparation
    # Optional velocity-selective preparation; zero disables selection.
    selection_sigma_vx_m_s: float = 0.010
    integration_steps: int = 96
    envelope_half_widths: float = 3.0
    scan_points: int = 41


@dataclass
class DetectionConfig:
    z_m: float = 0.075
    aperture_radius_m: float = 0.006
    interaction_length_m: float = 0.001
    saturation_F1: float = 0.5  # F1 channel assumes ideal repump then detection
    saturation_F2: float = 0.7
    collection_fraction: float = 0.01  # solid angle / (4*pi)
    quantum_efficiency: float = 0.5
    gate_s: float = 0.010  # stationary phase during at least the whole flight
    background_counts: float = 100.0
    electronic_noise_std_counts: float = 5.0
    noise: bool = True


@dataclass
class PylcpMOTConfig:
    """Native 24-state PyLCP rate equations; SI inputs converted at one boundary."""
    repump_saturation: float = 0.5  # per transverse beam, total before FM splitting
    repump_axial_saturation: float = 0.1
    repump_Fprime: int = 2
    repump_detuning_gamma: float = 0.0
    cooling_fm_frequency_Hz: float = 0.0
    cooling_fm_index: float = 0.0
    axial_fm_frequency_Hz: float = 0.0
    axial_fm_index: float = 0.0
    repump_fm_frequency_Hz: float = 0.0
    repump_fm_index: float = 0.0
    sideband_order: int = 3
    bias_field_T: tuple = (0.0, 0.0, 1.0e-5)
    hole_edge_m: float = 25.0e-6
    initial_z_m: float = 10.0e-6
    initial_ground_F2_fraction: float = 0.625
    initial_ground_populations: tuple = ()  # explicit eight populations, otherwise statistical mixture
    gravity_m_s2: tuple = (0.0, -9.80665, 0.0)
    recoil_model: str = "diffusion"  # diffusion splitting or none; not photon-resolved jumps
    recoil_chunk_s: float = 2.0e-4
    max_motion_step_s: float = 2.0e-4
    rtol: float = 1.0e-5
    population_atol: float = 1.0e-9
    include_magnetic_force: bool = True


@dataclass
class OpticalPumpingConfig:
    enabled: bool = False
    center_z_m: float = 0.014
    waist_z_m: float = 0.00625  # 1/e^2 intensity radius; thesis diameter 12.5 mm
    waist_y_m: float = 0.00625
    half_widths: float = 2.0  # truncate light at center +/- 2 waists
    transfer_power_W: float = 0.002  # F=2 -> Fprime=1, total over +/-x beams
    dark_power_W: float = 0.002  # F=1 -> Fprime=0
    saturation_intensity_W_m2: float = 16.69  # D2 cycling transition convention
    transfer_detuning_Hz: float = 0.0
    dark_detuning_Hz: float = 0.0
    bias_field_T: float = 2.05e-4  # +z; weak-field linear Zeeman
    pi_power_fraction: float = 0.0  # E_z leakage; remaining E_y gives sigma+/-
    forward_power_fraction: float = 0.5
    recoil_model: str = "diffusion"
    recoil_chunk_s: float = 5.0e-5
    max_step_s: float = 5.0e-5
    rtol: float = 1.0e-6
    population_atol: float = 1.0e-10
    diffusion_quadrature_order: int = 1  # legacy replay; optimization uses 3
    rate_backend: str = "native"  # fixed_axis caches constant-field PyLCP coefficients


@dataclass
class RabiScanConfig:
    # Scan coupling amplitude at fixed spatial envelope; reference area / pi.
    area_min_pi: float = 0.0
    area_max_pi: float = 6.0
    points: int = 101
    transit_area_pi: float = 4.0


@dataclass
class ChirpScanConfig:
    # Ordinary-frequency slope; RamanConfig.chirp_rad_s2 = 2*pi*alpha_Hz_s.
    center_Hz_s: float = 0.0
    half_range_Hz_s: float = 2.5e6
    points: int = 121


@dataclass
class SimulationConfig:
    seed: int = 87
    source: SourceConfig = field(default_factory=SourceConfig)
    raman: RamanConfig = field(default_factory=RamanConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    pylcp: PylcpMOTConfig = field(default_factory=PylcpMOTConfig)
    pumping: OpticalPumpingConfig = field(default_factory=OpticalPumpingConfig)
    rabi_scan: RabiScanConfig = field(default_factory=RabiScanConfig)
    chirp_scan: ChirpScanConfig = field(default_factory=ChirpScanConfig)

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, data):
        allowed = {item.name for item in fields(cls)}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError("Unknown configuration keys: " + ", ".join(sorted(unknown)))
        data = dict(data)
        for name, typ in (("source", SourceConfig), ("raman", RamanConfig),
                          ("detection", DetectionConfig), ("pylcp", PylcpMOTConfig),
                          ("pumping", OpticalPumpingConfig), ("rabi_scan", RabiScanConfig),
                          ("chirp_scan", ChirpScanConfig)):
            if name in data:
                data[name] = typ(**data[name])
        config = cls(**data)
        config.validate()
        return config

    def validate(self):
        s, r, d = self.source, self.raman, self.detection
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name, section in (("source", s), ("raman", r), ("detection", d),
                              ("pylcp", self.pylcp), ("pumping", self.pumping),
                              ("rabi_scan", self.rabi_scan), ("chirp_scan", self.chirp_scan)):
            for key, value in asdict(section).items():
                if isinstance(value, (int, float, tuple, list)) and not np.all(np.isfinite(value)):
                    raise ValueError(name + "." + key + " must be finite")
        if s.mode not in ("mot", "calibrated", "pylcp"):
            raise ValueError("source.mode must be mot, calibrated or pylcp")
        if r.state_handling not in ("auto", "inherit", "project_F1_m0", "ideal_preparation"):
            raise ValueError("Unknown Raman state_handling")
        if r.coupling_model not in ("effective", "rb87_d2_power"):
            raise ValueError("Unknown Raman coupling_model")
        if min(r.beam1_power_W, r.beam2_power_W) < 0:
            raise ValueError("Raman beam powers must be nonnegative")
        if r.coupling_model == "effective" and (r.beam1_power_W > 0 or r.beam2_power_W > 0):
            raise ValueError("Nonzero optical powers require coupling_model='rb87_d2_power'")
        if (isinstance(r.detuning_reference_Fprime, bool)
                or r.detuning_reference_Fprime not in (0, 1, 2, 3)):
            raise ValueError("Raman detuning reference Fprime must be 0,1,2,3")
        if r.polarization1 not in ("sigma_plus", "sigma_minus", "linear") or r.polarization2 not in ("sigma_plus", "sigma_minus", "linear"):
            raise ValueError("Unknown Raman polarization")
        if r.spin_transport not in ("none", "adiabatic_from_z"):
            raise ValueError("Unknown Raman spin_transport")
        if r.coupling_model == "rb87_d2_power":
            if r.envelope != "gaussian":
                raise ValueError("Actual power calibration currently requires matched Gaussian beams")
            if r.peak_rabi_Hz != 0:
                raise ValueError("Power mode derives peak_rabi_Hz; set the explicit effective peak to zero")
            axis = np.asarray(r.quantization_axis, dtype=float)
            if axis.shape != (3,) or not np.allclose(axis, r.k_direction, atol=1e-12, rtol=0):
                raise ValueError("Power model requires the common quantization axis parallel to beam 1 (+k)")
            if not np.allclose(r.k_direction, (1., 0., 0.), atol=1e-12, rtol=0):
                raise ValueError("Power calibration currently assumes beams along +/-x and Gaussian waists along y,z")
            if r.spin_transport != "adiabatic_from_z":
                raise ValueError("Power model requires explicit ideal adiabatic spin transport from the preparation z axis")
        p = self.pylcp
        if p.repump_Fprime not in (0, 1, 2):
            raise ValueError("F=1 repump must address Fprime=0,1,2")
        if not isinstance(p.sideband_order, int) or p.sideband_order < 0:
            raise ValueError("sideband_order must be a nonnegative integer")
        if p.recoil_model not in ("diffusion", "none"):
            raise ValueError("recoil_model must be diffusion or none")
        if min(p.recoil_chunk_s, p.max_motion_step_s, p.rtol, p.population_atol) <= 0:
            raise ValueError("PyLCP integration tolerances and time steps must be positive")
        if min(p.repump_saturation, p.repump_axial_saturation, p.hole_edge_m) < 0:
            raise ValueError("Repump saturation and hole edge must be nonnegative")
        if s.mode == "pylcp" and (p.hole_edge_m >= s.exit_radius_m or not 0 < p.initial_z_m < s.mot_length_m):
            raise ValueError("Require hole_edge_m < exit_radius and 0 < initial_z_m < mot_length")
        if s.mode == "pylcp" and s.initial_radius_m > s.chamber_radius_m:
            raise ValueError("Initial PyLCP samples must lie inside the source chamber")
        if not 0 <= p.initial_ground_F2_fraction <= 1:
            raise ValueError("initial_ground_F2_fraction must be in [0,1]")
        if len(p.bias_field_T) != 3 or len(p.gravity_m_s2) != 3:
            raise ValueError("PyLCP field and gravity vectors must have three components")
        if len(p.initial_ground_populations):
            pops = np.asarray(p.initial_ground_populations)
            if pops.shape != (8,) or np.min(pops) < 0 or not np.isclose(pops.sum(), 1):
                raise ValueError("initial_ground_populations must have eight nonnegative entries summing to one")
        for frequency, index, offsets, powers in (
                (p.cooling_fm_frequency_Hz, p.cooling_fm_index, s.cooling_spectrum_gamma, s.cooling_spectrum_weights),
                (p.axial_fm_frequency_Hz, p.axial_fm_index, s.axial_spectrum_gamma, s.axial_spectrum_weights),
                (p.repump_fm_frequency_Hz, p.repump_fm_index, (0,), (1,))):
            if frequency < 0 or index < 0 or (index > 0 and frequency <= 0):
                raise ValueError("FM index and frequency must be nonnegative; nonzero index needs positive frequency")
            if index > 0 and (len(offsets) != 1 or offsets[0] != 0 or powers[0] != 1):
                raise ValueError("Choose manual spectral components or Bessel FM, not both for one carrier")
        for name, value in (("n_atoms", s.n_atoms), ("scan_points", r.scan_points),
                            ("integration_steps", r.integration_steps)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(name + " must be a positive integer")
        if r.scan_points < 5:
            raise ValueError("scan_points must be >= 5")
        for name, options in (("rabi_scan", self.rabi_scan), ("chirp_scan", self.chirp_scan)):
            if isinstance(options.points, bool) or not isinstance(options.points, (int, np.integer)) or options.points < 5:
                raise ValueError(name + ".points must be an integer >= 5")
        rs, cs = self.rabi_scan, self.chirp_scan
        if not 0 <= rs.area_min_pi < rs.area_max_pi or rs.transit_area_pi < 0:
            raise ValueError("Rabi areas require 0 <= area_min_pi < area_max_pi and transit_area_pi >= 0")
        if cs.half_range_Hz_s <= 0:
            raise ValueError("Chirp half_range_Hz_s must be positive")
        positive = (s.incident_flux_per_s, s.initial_radius_m, s.mot_length_m,
                    s.chamber_radius_m, s.exit_radius_m, s.cooling_waist_m,
                    s.dt_s, s.max_time_s, s.calibrated_mean_vz_m_s,
                    s.calibrated_flux_per_s, r.first_zone_z_m, r.zone_spacing_m,
                    r.waist_z_m, r.waist_y_m, r.rectangular_width_m, r.reference_vz_m_s,
                    r.envelope_half_widths, d.aperture_radius_m,
                    d.interaction_length_m, d.gate_s, d.saturation_F1, d.saturation_F2)
        if min(positive) <= 0:
            raise ValueError("Lengths, time steps, fluxes and detection saturations must be positive")
        nonnegative = (s.initial_temperature_K, s.initial_sigma_vz_m_s,
                       s.cooling_saturation, s.magnetic_gradient_T_m,
                       s.effective_magnetic_moment_muB, s.axial_forward_saturation,
                       s.axial_backward_saturation, s.calibrated_fwhm_vz_m_s,
                       s.calibrated_transverse_temperature_K, r.selection_sigma_vx_m_s, r.peak_rabi_Hz,
                       d.background_counts, d.electronic_noise_std_counts)
        if min(nonnegative) < 0:
            raise ValueError("Temperatures, saturation parameters and noise levels must be nonnegative")
        for value in (r.preparation_efficiency, d.collection_fraction, d.quantum_efficiency):
            if not 0 < value <= 1:
                raise ValueError("Efficiencies must be in (0, 1]")
        if not 0 <= r.spectator_F2_fraction <= 1:
            raise ValueError("spectator_F2_fraction must be in [0,1]")
        for offsets, powers in ((s.cooling_spectrum_gamma, s.cooling_spectrum_weights),
                                 (s.axial_spectrum_gamma, s.axial_spectrum_weights)):
            if not len(offsets) or len(offsets) != len(powers) or min(powers) < 0 or not np.isclose(sum(powers), 1):
                raise ValueError("Spectral offsets and powers must have equal length, nonnegative powers summing to 1")
        if r.envelope not in ("gaussian", "rectangular"):
            raise ValueError("envelope must be gaussian or rectangular")
        k = np.asarray(r.k_direction, dtype=float)
        a = np.asarray(r.acceleration_m_s2, dtype=float)
        rotation = np.asarray(r.rotation_rad_s, dtype=float)
        if k.shape != (3,) or a.shape != (3,) or rotation.shape != (3,):
            raise ValueError("k_direction, acceleration and rotation must have three components")
        if not np.isclose(np.linalg.norm(k), 1) or k[0] <= 0 or abs(k[1]) > 1e-12 or abs(k[2]) > 0.05:
            raise ValueError("Use a unit k_direction close to +x in the xz plane (|kz| <= 0.05)")
        if a[2] != 0:
            raise ValueError("Constant longitudinal velocity requires acceleration_z=0")
        if len(r.phase_offsets_rad) != 3:
            raise ValueError("Three phase offsets are required")
        span = r.envelope_half_widths * r.waist_z_m if r.envelope == "gaussian" else r.rectangular_width_m / 2
        if r.first_zone_z_m <= span or r.zone_spacing_m <= 2 * span:
            raise ValueError("Raman envelopes must lie downstream and must not overlap")
        if d.z_m <= r.first_zone_z_m + 2 * r.zone_spacing_m + span:
            raise ValueError("Detection must be downstream of the third Raman envelope")
        op = self.pumping
        if min(op.waist_z_m, op.waist_y_m, op.half_widths, op.bias_field_T,
               op.saturation_intensity_W_m2, op.max_step_s, op.recoil_chunk_s,
               op.rtol, op.population_atol) <= 0 or min(op.transfer_power_W, op.dark_power_W) < 0:
            raise ValueError("Pumping widths, bias and tolerances must be positive; powers nonnegative")
        if not 0 <= op.pi_power_fraction <= 1 or not 0 <= op.forward_power_fraction <= 1:
            raise ValueError("Pumping polarization and forward power fractions must lie in [0,1]")
        if op.recoil_model not in ("diffusion", "none", "compound_poisson"):
            raise ValueError("Pumping recoil_model must be diffusion, compound_poisson or none")
        if (isinstance(op.diffusion_quadrature_order, bool)
                or not isinstance(op.diffusion_quadrature_order, int)
                or not 1 <= op.diffusion_quadrature_order <= 8):
            raise ValueError("Pumping diffusion_quadrature_order must be an integer from 1 to 8")
        if op.rate_backend not in ("native", "fixed_axis"):
            raise ValueError("Pumping rate_backend must be native or fixed_axis")
        if op.enabled:
            if op.center_z_m - op.half_widths * op.waist_z_m <= 0:
                raise ValueError("Pumping light must start downstream of the source exit")
            if op.center_z_m + op.half_widths * op.waist_z_m >= r.first_zone_z_m - span:
                raise ValueError("Pumping light must end before the first Raman envelope")
            if r.state_handling == "ideal_preparation":
                raise ValueError("Optical pumping requires inherit, auto or project_F1_m0 state handling")
