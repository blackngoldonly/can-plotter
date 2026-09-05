"""
gyro_allan_capture.py

Standalone (no GUI) capture + Allan deviation analysis tool for
characterizing gyro noise (ARW / bias instability) on a static IMU.

Usage
-----
1) Capture only:
       python gyro_allan_capture.py capture --duration 7200 --out gyro_static_log.csv

   Run this with the IMU rigidly clamped and NOT moving, ideally for
   several hours. Ctrl+C stops early and still saves what was captured.

2) Analyze only (from an existing log):
       python gyro_allan_capture.py analyze --in gyro_static_log.csv

3) Capture then immediately analyze:
       python gyro_allan_capture.py both --duration 7200 --out gyro_static_log.csv

Notes
-----
- Sample rate for the Allan computation is taken from NOMINAL_RATE_HZ
  (i.e. your CAN timer's designed rate), not from measured wall-clock
  deltas, since you've said the CAN-side timer discipline is good.
  Wall-clock timestamps are still logged so you can verify this
  assumption after the fact (see the printed jitter stats).
- Requires: python-can, numpy, allantools, matplotlib
    pip install python-can numpy allantools matplotlib
"""

import sys
import csv
import time
import struct
import argparse

import numpy as np

try:
    import can
except ImportError:
    can = None


# ============================================================
# Configuration
# ============================================================

CAN_INTERFACE = "vcan0"
GYRO_CAN_ID = 0x693

NOMINAL_RATE_HZ = 100.0          # i.e. 10 ms period -- used for Allan calc
NOMINAL_DT = 1.0 / NOMINAL_RATE_HZ

GYRO_SCALE = 1.0 / 100000.0      # matches decode_gyro() in the live monitor


# ============================================================
# CAN decode (same convention as the live monitor script)
# ============================================================

def decode_gyro(msg):
    gx, gy, gz = struct.unpack_from("<hhh", msg.data)
    return (gx * GYRO_SCALE, gy * GYRO_SCALE, gz * GYRO_SCALE)


# ============================================================
# Capture
# ============================================================

def run_capture(duration_s, out_path):
    if can is None:
        print("python-can is not installed. `pip install python-can`")
        sys.exit(1)

    bus = can.Bus(interface="socketcan", channel=CAN_INTERFACE)

    print(f"Logging gyro (CAN ID 0x{GYRO_CAN_ID:X}) from {CAN_INTERFACE}")
    print(f"Target duration: {duration_s} s   (Ctrl+C to stop early)")
    print(f"Nominal sample rate assumed for Allan calc: {NOMINAL_RATE_HZ} Hz")

    rows = []  # (sample_index, wall_time_s, gx, gy, gz)
    start = time.monotonic()
    sample_idx = 0

    try:
        while True:
            now = time.monotonic() - start
            if duration_s is not None and now >= duration_s:
                break

            msg = bus.recv(timeout=1.0)
            if msg is None:
                continue

            if msg.arbitration_id != GYRO_CAN_ID:
                continue

            try:
                gx, gy, gz = decode_gyro(msg)
            except Exception as e:
                print(f"Decode error: {e}")
                continue

            rows.append((sample_idx, now, gx, gy, gz))
            sample_idx += 1

            if sample_idx % 5000 == 0:
                print(f"  {sample_idx} samples  ({now:.1f} s elapsed)")

    except KeyboardInterrupt:
        print("\nStopped early by user.")

    finally:
        bus.shutdown()

    if not rows:
        print("No samples captured. Nothing written.")
        return

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_idx", "wall_time_s", "gx", "gy", "gz"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} samples to {out_path}")
    _print_timing_diagnostics(rows)


def _print_timing_diagnostics(rows):
    """Sanity-check actual vs nominal sample rate."""
    wall_t = np.array([r[1] for r in rows])
    dt = np.diff(wall_t)

    measured_rate = 1.0 / np.mean(dt)
    print("\n--- Timing diagnostics ---")
    print(f"Nominal rate:  {NOMINAL_RATE_HZ:.4f} Hz  (dt = {NOMINAL_DT*1000:.3f} ms)")
    print(f"Measured rate: {measured_rate:.4f} Hz  (mean dt = {np.mean(dt)*1000:.3f} ms)")
    print(f"dt std dev:    {np.std(dt)*1000:.4f} ms")
    print(f"dt min/max:    {np.min(dt)*1000:.3f} / {np.max(dt)*1000:.3f} ms")

    pct_jitter = 100.0 * np.std(dt) / np.mean(dt)
    if pct_jitter > 5.0:
        print(
            f"WARNING: dt jitter is {pct_jitter:.1f}% of the sample period. "
            f"Using NOMINAL_RATE_HZ for Allan analysis may distort short-tau "
            f"results -- consider resampling to a uniform grid first."
        )
    else:
        print(f"Jitter is {pct_jitter:.1f}% of sample period -- looks fine to "
              f"use nominal rate directly.")


# ============================================================
# Analysis
# ============================================================

def run_analysis(in_path, save_fig=None):
    try:
        import allantools
    except ImportError:
        print("allantools is not installed. `pip install allantools`")
        sys.exit(1)

    import matplotlib.pyplot as plt

    data = np.genfromtxt(in_path, delimiter=",", names=True)

    wall_t = data["wall_time_s"]
    dt = np.diff(wall_t)
    print(f"Loaded {len(wall_t)} samples from {in_path}")
    print(f"Duration: {wall_t[-1] - wall_t[0]:.1f} s "
          f"({(wall_t[-1] - wall_t[0]) / 3600:.2f} hr)")
    _print_timing_diagnostics(
        list(zip(range(len(wall_t)), wall_t))
    )

    axes = ["gx", "gy", "gz"]
    colors = {"gx": "tab:red", "gy": "tab:green", "gz": "tab:blue"}

    results = {}

    fig, ax = plt.subplots(figsize=(8, 6))

    for axis in axes:
        y = data[axis]

        # oadev expects rate data (angular rate), data_type="freq"
        taus, adev, adev_err, n = allantools.oadev(
            y, rate=NOMINAL_RATE_HZ, data_type="freq", taus="octave"
        )

        ax.loglog(taus, adev, "-o", color=colors[axis], label=axis, markersize=3)

        arw, bias_instab = _estimate_arw_and_bias_instability(taus, adev)
        results[axis] = {"taus": taus, "adev": adev, "arw": arw, "bi": bias_instab}

        print(f"\n{axis}:")
        print(f"  ARW (approx, from tau=1 intercept of -1/2 slope region): "
              f"{arw:.6f} deg/sqrt(hr)  [see caveats below]")
        print(f"  Bias instability (approx, from curve minimum):          "
              f"{bias_instab:.6f} deg/hr")

    ax.set_xlabel("Averaging time τ (s)")
    ax.set_ylabel("Allan deviation (deg/s)")
    ax.set_title("Gyro Allan Deviation (static)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    if save_fig:
        fig.savefig(save_fig, dpi=150)
        print(f"\nSaved plot to {save_fig}")

    plt.show()

    return results


def _estimate_arw_and_bias_instability(taus, adev):
    """
    Rough, automated slope-based estimate.

    NOTE: this is a convenience estimate, not a substitute for eyeballing
    the log-log curve yourself (see caveats in chat). For anything you'll
    rely on for filter tuning, also run allantools' noise-identification
    fit (allantools.Dataset + noise_id) and compare.
    """
    taus = np.asarray(taus)
    adev = np.asarray(adev)

    log_tau = np.log10(taus)
    log_adev = np.log10(adev)

    # Bias instability: minimum of the curve, scaled by 0.664
    min_idx = np.argmin(adev)
    bias_instability_deg_per_s = adev[min_idx] * 0.664
    bias_instability_deg_per_hr = bias_instability_deg_per_s * 3600.0

    # ARW: find point(s) with slope closest to -0.5 in the region BEFORE
    # the minimum, then project back to tau=1 using that local slope.
    arw_deg_per_sqrt_hr = np.nan
    if min_idx >= 2:
        slopes = np.diff(log_adev[:min_idx + 1]) / np.diff(log_tau[:min_idx + 1])
        target = -0.5
        best_i = np.argmin(np.abs(slopes - target))
        # local slope segment best_i -> best_i+1
        tau_a, tau_b = taus[best_i], taus[best_i + 1]
        adev_a = adev[best_i]
        slope = slopes[best_i]
        # project adev at tau=1 using this local slope in log-log space
        log_adev_at_1 = np.log10(adev_a) + slope * (np.log10(1.0) - np.log10(tau_a))
        arw_deg_per_sqrt_s = 10 ** log_adev_at_1
        arw_deg_per_sqrt_hr = arw_deg_per_sqrt_s * 60.0  # deg/sqrt(s) -> deg/sqrt(hr)

    return arw_deg_per_sqrt_hr, bias_instability_deg_per_hr


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    p_cap = sub.add_parser("capture", help="Capture raw gyro CAN data to CSV")
    p_cap.add_argument("--duration", type=float, default=None,
                        help="Capture duration in seconds (default: run until Ctrl+C)")
    p_cap.add_argument("--out", type=str, default="gyro_static_log.csv")

    p_ana = sub.add_parser("analyze", help="Run Allan deviation analysis on a CSV log")
    p_ana.add_argument("--in", dest="in_path", type=str, required=True)
    p_ana.add_argument("--save-fig", type=str, default=None,
                        help="Optional path to save the plot as PNG")

    p_both = sub.add_parser("both", help="Capture then analyze")
    p_both.add_argument("--duration", type=float, default=None)
    p_both.add_argument("--out", type=str, default="gyro_static_log.csv")
    p_both.add_argument("--save-fig", type=str, default=None)

    args = parser.parse_args()

    if args.mode == "capture":
        run_capture(args.duration, args.out)
    elif args.mode == "analyze":
        run_analysis(args.in_path, args.save_fig)
    elif args.mode == "both":
        run_capture(args.duration, args.out)
        run_analysis(args.out, args.save_fig)


if __name__ == "__main__":
    main()
