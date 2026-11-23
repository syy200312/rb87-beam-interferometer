"""Run from the repository root: python -m rb87_beam --help."""

import argparse
import copy
import json
from pathlib import Path

from .config import SimulationConfig
from .io import export_result, read_legacy_h5, read_stateful_ensemble
from .physics import run_simulation
from .presets import thesis_preset, broadband_mot_preset, pylcp_mot_preset


def main():
    parser = argparse.ArgumentParser(description="87Rb spatial three-zone Raman beam interferometer (SI units)",
        epilog="Compare phase, mirror, magnetic, and intensity noise at fixed flux with "
               "python -m rb87_beam.raman_noise_study --help")
    settings = parser.add_mutually_exclusive_group()
    settings.add_argument("--config", type=Path, help="JSON configuration; omitted fields use defaults")
    settings.add_argument("--preset", choices=("thesis", "broadband-mot", "pylcp-mot"), help="Documented starting points")
    parser.add_argument("--source", choices=("mot", "calibrated", "pylcp"))
    parser.add_argument("--atoms", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path, default=Path("beam_outputs"))
    parser.add_argument("--experiment", choices=("phase", "rabi", "chirp", "all"), default="phase",
                        help="Third-zone phase fringe, single-zone Rabi scan, chirp-rate fringe, or all three")
    parser.add_argument("--rabi-max-area-pi", type=float, help="Largest reference pulse area / pi at fixed zone width")
    parser.add_argument("--rabi-points", type=int)
    parser.add_argument("--rabi-transit-area-pi", type=float, help="Total reference area / pi for the transit-time trace")
    parser.add_argument("--chirp-center-hz-s", type=float, help="Absolute two-photon frequency chirp scan center, Hz/s")
    parser.add_argument("--chirp-half-range-hz-s", type=float)
    parser.add_argument("--chirp-points", type=int)
    parser.add_argument("--no-noise", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--state-handling", choices=("auto", "inherit", "project_F1_m0", "ideal_preparation"))
    pumping_flags = parser.add_mutually_exclusive_group()
    pumping_flags.add_argument("--optical-pumping", action="store_true", help="Finite-flight two-frequency 24-level optical preparation")
    pumping_flags.add_argument("--no-optical-pumping", dest="optical_pumping", action="store_false", help="Replay a prepared checkpoint without pumping it a second time")
    parser.set_defaults(optical_pumping=None)
    parser.add_argument("--compare-preparation", action="store_true", help="Paired comparison: mixed, transfer-only, two-frequency, pi leakage and ideal selectors")
    parser.add_argument("--pump-pi-fraction", type=float, help="Unwanted pi polarization POWER fraction in pumping light")
    parser.add_argument("--pump-recoil-model", choices=("diffusion", "compound_poisson", "none"))
    parser.add_argument("--recoil-model", choices=("diffusion", "none"), help="Native PyLCP source recoil integration")
    parser.add_argument("--no-repump", action="store_true", help="Disable transverse and axial repumping for comparison")
    parser.add_argument("--no-fm", action="store_true", help="Disable Bessel phase-modulation indices for comparison")
    parser.add_argument("--write-default-config", type=Path, help="Write default JSON and exit")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--source-h5", type=Path, help="Legacy sol_i/r,v; no automatic internal-state mapping")
    inputs.add_argument("--source-state-h5", type=Path, help="Self-describing SI source_ensemble.h5, preserving internal states")
    parser.add_argument("--length-scale-m", type=float)
    parser.add_argument("--time-scale-s", type=float)
    parser.add_argument("--exit-z-m", type=float)
    parser.add_argument("--axis-order", default="xyz", help="Input axes used as new x,y,z; see README for handedness")
    args = parser.parse_args()
    try:
        config = (SimulationConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8-sig")))
                  if args.config else SimulationConfig())
        if args.preset:
            config = {"thesis": thesis_preset, "broadband-mot": broadband_mot_preset,
                      "pylcp-mot": pylcp_mot_preset}[args.preset]()
        elif (not args.config and args.source in (None, "pylcp")
              and (args.source == "pylcp" or args.optical_pumping or args.compare_preparation)):
            config = pylcp_mot_preset()
        if args.source is not None:
            config.source.mode = args.source
        if args.atoms is not None:
            config.source.n_atoms = args.atoms
        if args.seed is not None:
            config.seed = args.seed
        if args.no_noise:
            config.detection.noise = False
        for argument, section, key in (
                (args.rabi_max_area_pi, config.rabi_scan, "area_max_pi"),
                (args.rabi_points, config.rabi_scan, "points"),
                (args.rabi_transit_area_pi, config.rabi_scan, "transit_area_pi"),
                (args.chirp_center_hz_s, config.chirp_scan, "center_Hz_s"),
                (args.chirp_half_range_hz_s, config.chirp_scan, "half_range_Hz_s"),
                (args.chirp_points, config.chirp_scan, "points")):
            if argument is not None:
                setattr(section, key, argument)
        if args.state_handling:
            config.raman.state_handling = args.state_handling
        if args.optical_pumping is not None:
            config.pumping.enabled = args.optical_pumping
        if args.pump_pi_fraction is not None:
            config.pumping.pi_power_fraction = args.pump_pi_fraction
        if args.pump_recoil_model:
            config.pumping.recoil_model = args.pump_recoil_model
        if args.recoil_model:
            config.pylcp.recoil_model = args.recoil_model
        if args.no_repump:
            config.pylcp.repump_saturation = config.pylcp.repump_axial_saturation = 0.0
        if args.no_fm:
            config.pylcp.cooling_fm_index = config.pylcp.axial_fm_index = config.pylcp.repump_fm_index = 0.0
        config.validate()
        if args.write_default_config:
            config.save(args.write_default_config)
            print("Configuration saved:", args.write_default_config.resolve())
            return
        ensemble = None
        if args.source_state_h5:
            ensemble = read_stateful_ensemble(args.source_state_h5)
        if args.source_h5:
            if None in (args.length_scale_m, args.time_scale_s, args.exit_z_m):
                parser.error("--source-h5 requires --length-scale-m, --time-scale-s and --exit-z-m")
            ensemble = read_legacy_h5(args.source_h5, args.length_scale_m, args.time_scale_s,
                                      args.exit_z_m, config.source.incident_flux_per_s, args.axis_order)
        if (ensemble is None and (config.pumping.enabled or args.compare_preparation)
                and config.source.mode != "pylcp"):
            raise ValueError("Optical pumping needs actual internal states: use --preset pylcp-mot or --source-state-h5")
        if ensemble is None:
            print("Preparing atoms: generating a fresh source ensemble.")
        elif config.pumping.enabled:
            print("Loaded source checkpoint; computing optical pumping before Raman propagation.")
        else:
            print("Loaded prepared checkpoint; MOT and optical pumping are not recomputed.")
        def progress(done, total, extracted):
            if done == 1 or done == total or done % 8 == 0:
                print("PyLCP atoms: %d/%d; processed/extracted: %d" % (done, total, extracted), flush=True)
        if args.compare_preparation:
            if args.experiment != "phase":
                raise ValueError("--compare-preparation uses the phase experiment; run rabi/chirp separately")
            from .comparison import compare_preparation
            if ensemble is None:
                source_config = copy.deepcopy(config)
                source_config.pumping.enabled = False
                ensemble = run_simulation(source_config, progress=progress)["source_ensemble"]
            rows = compare_preparation(config, ensemble, args.output, plot=not args.no_plot, progress=progress)
            print("Comparison:", (args.output / "comparison.md").resolve())
            for row in rows:
                print("%s: F1,m0=%.3f%%, useful flux=%.6g/s, contrast=%.3f%%" % (
                    row["case"], 100 * row["preparation_efficiency"], row["effective_target_flux_per_s"],
                    100 * row["fringe_contrast"]))
            return
        if args.experiment != "phase":
            from .physics import prepare_ensemble
            from .scans import run_rabi_scan, run_chirp_scan
            from .scan_io import export_scan_result
            prepared = prepare_ensemble(config, ensemble, progress)
            experiments = ("phase", "rabi", "chirp") if args.experiment == "all" else (args.experiment,)
            outputs = {}
            for experiment in experiments:
                directory = args.output / experiment if args.experiment == "all" else args.output
                print("Running %s experiment on the same %d prepared atoms..." % (
                    experiment, len(prepared["ensemble"].r_m)), flush=True)
                if experiment == "phase":
                    experiment_result = run_simulation(prepared=prepared)
                    export_result(experiment_result, directory, plot=not args.no_plot)
                else:
                    def scan_progress(done, total, atoms):
                        if done == 1 or done == total or done % 20 == 0:
                            print("%s scan: %d/%d points; %d atoms" % (experiment, done, total, atoms), flush=True)
                    runner = run_rabi_scan if experiment == "rabi" else run_chirp_scan
                    experiment_result = runner(prepared=prepared, progress=scan_progress)
                    export_scan_result(experiment_result, directory, plot=not args.no_plot)
                outputs[experiment] = str(directory.resolve())
                print("Output:", directory.resolve())
                if experiment == "chirp":
                    print("Reference acceleration-compensating chirp (Hz/s):",
                          experiment_result["reference_chirp_Hz_s"])
                    print("Reference-sinusoid contrast (approximate):", experiment_result["fit"]["contrast"])
                    print("Sampled expected visibility:", experiment_result["sampled_expected_visibility"])
                elif experiment == "rabi":
                    print("Rabi area range / pi:", experiment_result["reference_area_pi"][[0, -1]])
                    print("Reference transit duration (us):", experiment_result["reference_transit_s"] * 1e6)
            if args.experiment == "all":
                (args.output / "suite_summary.json").write_text(json.dumps({
                    "same_prepared_ensemble": True, "atoms": len(prepared["ensemble"].r_m),
                    "experiments": outputs}, indent=2), encoding="utf-8")
            return
        result = run_simulation(config, ensemble, progress=progress)
        summary = export_result(result, args.output, plot=not args.no_plot)
    except (ValueError, OSError, TypeError, RuntimeError, ModuleNotFoundError) as error:
        parser.exit(2, "Simulation error: " + str(error) + "\n")
    print("Output:", args.output.resolve())
    print("Source atoms:", len(result["ensemble"].r_m))
    print("Mean longitudinal velocity: %.5f m/s" % summary["source_exit_mean_vz_m_s"])
    contrast = summary["fit"]["contrast"]
    phase = summary["fit"]["phase_rad"]
    contrast_text = "undefined" if contrast is None else "%.6f (%.2f%%)" % (contrast, 100 * contrast)
    phase_text = "undefined (zero amplitude)" if phase is None else "%.6f" % phase
    print("Fringe contrast (dimensionless):", contrast_text)
    print("Fitted fringe phase (rad):", phase_text)
    print("The fitted phase is signed and may be negative; it is separate from contrast.")
    print("Effective detected Monte Carlo atoms: %.1f" % summary["effective_detected_MC_atoms"])


if __name__ == "__main__":
    main()
