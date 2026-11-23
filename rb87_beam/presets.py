"""Documented starting points, not claims of quantitative experimental reproduction."""

import numpy as np

from .config import SimulationConfig
from .physics import KB, MASS, GAMMA


def thesis_preset():
    """Meng 2022, table 2.2 / section 2.4, with explicitly stated extra assumptions."""
    config = SimulationConfig()
    s, r, d = config.source, config.raman, config.detection
    s.mode = "calibrated"
    s.n_atoms = 50000
    s.calibrated_mean_vz_m_s = 10.0
    s.calibrated_fwhm_vz_m_s = 3.0
    s.calibrated_transverse_temperature_K = MASS / KB * (0.13 / np.sqrt(8 * np.log(2))) ** 2
    s.calibrated_point_source = True
    # Table 2.2 does not specify an absolute flux. This is only a noise normalization.
    s.calibrated_flux_per_s = 1.0e9
    s.calibrated_flux_is_measured = False
    s.calibration_reference = "Meng 2022 thesis, table 2.2, printed pp.63-65; flux=1e9/s assumed"
    r.first_zone_z_m = 0.43
    r.zone_spacing_m = 0.0095
    r.reference_vz_m_s = 10.0
    r.envelope = "rectangular"
    r.rectangular_width_m = 0.001
    r.k_direction = (float(np.cos(0.01)), 0.0, float(-np.sin(0.01)))
    r.phase_offsets_rad = (float(np.pi / 2), 0.0, 0.0)
    r.preparation_efficiency = 1.0
    r.selection_sigma_vx_m_s = 0.0  # table assumes no extra velocity-selective preparation
    # Isolate the section 2.4 velocity / Raman comparison from gravity.
    r.acceleration_m_s2 = (0.0, 0.0, 0.0)
    # The y waist is not specified separately by table 2.2. Keep this assumption visible.
    r.waist_y_m = 0.006
    d.z_m = 0.609  # 0.43 + 2*0.0095 + 0.16; illustrative downstream geometry
    d.aperture_radius_m = 0.020
    return config


def broadband_mot_preset():
    """Illustrate the sideband frequencies discussed in thesis section 4.1.3.

    These are chosen power fractions, not a conversion from experimental dBm.
    Total power remains fixed when splitting it over frequency components.
    """
    c = SimulationConfig()
    c.source.cooling_detuning_gamma = -4.0
    c.source.cooling_spectrum_gamma = (-3.0, 0.0, 3.0)
    c.source.cooling_spectrum_weights = (0.1, 0.8, 0.1)
    c.source.axial_detuning_gamma = -4.0
    c.source.axial_spectrum_gamma = (-6.0, 0.0, 6.0)
    c.source.axial_spectrum_weights = (0.1, 0.8, 0.1)
    c.source.cooling_saturation = 5.27
    c.source.axial_forward_saturation = 2.63
    c.source.axial_backward_saturation = 2.63
    return c


def pylcp_mot_preset():
    """Full Rb87 D2 rate equations, repump and phase-modulation sidebands."""
    c = SimulationConfig()
    s, p = c.source, c.pylcp
    s.mode, s.n_atoms = "pylcp", 32
    s.cooling_detuning_gamma = -4.0
    s.axial_detuning_gamma = -4.0
    s.cooling_saturation = 5.27
    s.axial_forward_saturation = 2.63
    s.axial_backward_saturation = 2.63
    p.cooling_fm_frequency_Hz = 3 * GAMMA / (2 * np.pi)
    p.cooling_fm_index = 0.5
    p.axial_fm_frequency_Hz = 6 * GAMMA / (2 * np.pi)
    p.axial_fm_index = 0.5
    c.raman.state_handling = "inherit"
    c.raman.selection_sigma_vx_m_s = 0.0
    # This Doppler-limited source is broader/faster than the empirical source.
    # Broaden Raman acceptance by a shorter zone; keep T_ref=0.95 ms at 15 m/s.
    c.raman.reference_vz_m_s = 15.0
    c.raman.waist_z_m = 25.0e-6
    c.raman.zone_spacing_m = 0.01425
    return c
