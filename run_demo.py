"""Replay the bundled 256-atom sample and plot the phase, Rabi and chirp scans."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "outputs" / "demo",
                        help="Where to write the three scans (default: outputs/demo)")
    args = parser.parse_args()
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    command = [
        sys.executable, "-m", "rb87_beam", "--experiment", "all",
        "--config", "beam_outputs/raman_power_calibration/config_power_driven.json",
        "--source-state-h5", "beam_outputs/endpoint_comparison/native_validation/baseline/nominal/prepared_0.h5",
        "--no-optical-pumping", "--no-noise", "--output", str(args.output.resolve()),
    ]
    subprocess.run(command, cwd=root, env=env, check=True)
    print("Phase, Rabi and chirp plots:", args.output.resolve())


if __name__ == "__main__":
    main()
