# ============================================================
# NOTES
# ============================================================
"""
CAN frame
   ↓
decoder function
   ↓
{"signal_name": value, ...}
   ↓
signal buffers
   ↓
plots defined by PLOT_SPECS
"""


import sys
import time
import struct
import numpy as np
from dataclasses import dataclass, field
from collections import deque

import can
import pyqtgraph as pg

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
)


# ============================================================
# Configuration
# ============================================================

CAN_INTERFACE = "vcan0"

WINDOW_SECONDS = 300.0
GUI_UPDATE_MS = 20


pg.setConfigOption("background", "#f5f5f5")
pg.setConfigOption("foreground", "#202020")


# ============================================================
# 1. Define signals here
# ============================================================
#
# Add every datapoint you want to record/plot.
#
# Example:
# SIGNAL_SPECS = {
#    "gx": {"color": "#e74c3c", "label": "Gx"},
#    "gy": {"color": "#2ecc71", "label": "Gy"},
#    "gz": {"color": "#3498db", "label": "Gz"},
# }

SIGNAL_SPECS = {
        "ax": {"color": "#e74c3c", "label": "Ax", "stats": True},
        "ay": {"color": "#2ecc71", "label": "Ay", "stats": True},
        "az": {"color": "#3498db", "label": "Az", "stats": True},

    "gx": {"color": "#e74c3c", "label": "Gx", "stats": True},
    "gy": {"color": "#2ecc71", "label": "Gy", "stats": True},
    "gz": {"color": "#3498db", "label": "Gz", "stats": True},

    "qw": {"color": "#202020", "label": "Qw"},
    "qx": {"color": "#e74c3c", "label": "Qx"},
    "qy": {"color": "#2ecc71", "label": "Qy"},
    "qz": {"color": "#3498db", "label": "Qz"},

    "PThetaX": {"color": "#e74c3c", "label": "PThetaX"},
    "PThetaY": {"color": "#2ecc71", "label": "PThetaY"},
    "PThetaZ": {"color": "#3498db", "label": "PThetaZ"},

    "PBiasX": {"color": "#e74c3c", "label": "PBiasX"},
    "PBiasY": {"color": "#2ecc71", "label": "PBiasY"},
    "PBiasZ": {"color": "#3498db", "label": "PBiasZ"},

    "MagX": {"color": "#e74c3c", "label": "MagX"},
    "MagY": {"color": "#2ecc71", "label": "MagY"},
    "MagZ": {"color": "#3498db", "label": "MagZ"},
}


# ============================================================
# 2. Define plots here
# ============================================================
#
# Each entry creates one subplot.
#
# Example:
#
# PLOT_SPECS = [
#     {
#         "title": "Accelerometer",
#         "signals": ["ax", "ay", "az"],
#         "ylabel": "Acceleration",
#     },
#     {
#         "title": "Quaternion",
#         "signals": ["qw", "qx", "qy", "qz"],
#         "ylabel": "Quaternion",
#         "yrange": (-1.1, 1.1),
#     },
# ]
#

PLOT_SPECS = [
     {
         "title": "Gyroscope",
         "signals": ["gx", "gy", "gz"],
         "ylabel": "Angular Rate (Degrees)",
     },

     {
         "title": "PTheta",
         "signals": ["PThetaX", "PThetaY", "PThetaZ"],
         "ylabel": "Covariance",
     },

     {
         "title": "PBias",
         "signals": ["PBiasX", "PBiasY", "PBiasZ"],
         "ylabel": "Covariance",
     },

     {
         "title": "Mag",
         "signals": ["MagX", "MagY", "MagZ"],
         "ylabel": "Magnetometer",
     },
]


# ============================================================
# 3. CAN decoders
# ============================================================
#
# Each decoder:
#
#     input:
#         CAN message
#
#     output:
#         dictionary of signal_name -> value
#
#
# Example:
#
# def decode_accel(msg):
#     ax, ay, az = struct.unpack_from("<hhh", msg.data)
#
#     return {
#         "ax": ax,
#         "ay": ay,
#         "az": az,
#     }
#
#
# Then register it:
#
# CAN_DECODERS = {
#     0x123: decode_accel,
# }
#
def decode_gyro(msg):
    gx, gy, gz = struct.unpack_from("<hhh", msg.data)
    return {
            "gx": gx / 100000.0,
            "gy": gy / 100000.0,
            "gz": gz / 100000.0,
    }


def decode_accel(msg):
    ax, ay, az = struct.unpack_from("<hhh", msg.data)
    return {
            "ax": ax / 100000.0,
            "ay": ay / 100000.0,
            "az": az / 100000.0,
            }

def decode_PTheta(msg):
    pThetax, pThetay, pThetaz = struct.unpack_from("<hhh", msg.data)
    return {
            "PThetaX": pThetax / 1000.0,
            "PThetaY": pThetay / 1000.0,
            "PThetaZ": pThetaz / 1000.0,
    }

def decode_PBias(msg):
    pBiasx, pBiasy, pBiasz = struct.unpack_from("<hhh", msg.data)
    return {
            "PBiasX": pBiasx / 100000.0,
            "PBiasY": pBiasy / 100000.0,
            "PBiasZ": pBiasz / 100000.0,
    }
    
def decode_mag(msg):
    Magx, Magy, Magz = struct.unpack_from("<hhh", msg.data)
    return {
            "MagX": Magx / 100000.0,
            "MagY": Magy / 100000.0,
            "MagZ": Magz / 100000.0,
    }

CAN_DECODERS = {
        0x693: decode_gyro,
        0x593: decode_accel,
        0x793: decode_PTheta,
        0x7A3: decode_PBias,
        0x333: decode_mag,
}

# ============================================================
# Timer Functions
# ============================================================

def print_signal_stats():
    print("\n" + "=" * 70)

    for name, signal in signals.items():
        if not signal.data:
            continue

        x = np.asarray(signal.data, dtype=float)

        print(
            f"{name:12s} "
            f"n={len(x):6d}  "
            f"mean={np.mean(x):+10.6f}  "
            f"std={np.std(x):10.6f}  "
            f"min={np.min(x):+10.6f}  "
            f"max={np.max(x):+10.6f}  "
            f"p2p={np.ptp(x):10.6f}  "
            f"rms={np.sqrt(np.mean(x*x)):10.6f}"
        )

# ============================================================
# Signal storage
# ============================================================

@dataclass
class Signal:
    time: deque = field(default_factory=deque)
    data: deque = field(default_factory=deque)

    def append(self, timestamp, value):
        self.time.append(timestamp)
        self.data.append(value)

    def trim(self, oldest_allowed):
        while self.time and self.time[0] < oldest_allowed:
            self.time.popleft()
            self.data.popleft()


signals = {
    name: Signal()
    for name in SIGNAL_SPECS
}

# ============================================================
# CAN
# ============================================================

bus = can.Bus(
    interface="socketcan",
    channel=CAN_INTERFACE,
)

start_time = time.monotonic()


def receive_can():
    """
    Drain all currently waiting CAN frames.
    """

    while True:
        msg = bus.recv(timeout=0.0)

        if msg is None:
            break

        decoder = CAN_DECODERS.get(msg.arbitration_id)

        if decoder is None:
            continue

        try:
            values = decoder(msg)

        except Exception as e:
            print(
                f"Decode error on CAN ID "
                f"0x{msg.arbitration_id:X}: {e}"
            )
            continue

        timestamp = time.monotonic() - start_time

        for name, value in values.items():

            if name not in signals:
                print(f"Unknown signal: {name}")
                continue

            signals[name].append(
                timestamp,
                value,
            )


def trim_buffers():
    now = time.monotonic() - start_time

    oldest_allowed = now - WINDOW_SECONDS

    for signal in signals.values():
        signal.trim(oldest_allowed)


# ============================================================
# GUI
# ============================================================

app = QApplication(sys.argv)

window = QMainWindow()

central = QWidget()
layout = QVBoxLayout(central)

graphics = pg.GraphicsLayoutWidget()

layout.addWidget(graphics)

window.setCentralWidget(central)

window.resize(1400, 900)
window.setWindowTitle("CAN Monitor")

# ============================================================
# TIMER INIT FUNCTIONS
# ============================================================

stats_timer = QTimer()
stats_timer.timeout.connect(print_signal_stats)
stats_timer.start(5000)

# ============================================================
# Build plots automatically from PLOT_SPECS
# ============================================================

plots = {}
curves = {}


for row, spec in enumerate(PLOT_SPECS):

    plot = graphics.addPlot(
        row=row,
        col=0,
        title=spec["title"],
    )

    plot.showGrid(
        x=True,
        y=True,
        alpha=0.25,
    )

    plot.setLabel(
        "bottom",
        "Time",
        units="s",
    )

    if "ylabel" in spec:
        plot.setLabel(
            "left",
            spec["ylabel"],
        )

    if "yrange" in spec:
        ymin, ymax = spec["yrange"]

        plot.setYRange(
            ymin,
            ymax,
        )

    plot.addLegend()

    plots[spec["title"]] = plot

    curves[spec["title"]] = {}

    for signal_name in spec["signals"]:

        signal_spec = SIGNAL_SPECS[signal_name]

        curve = plot.plot(
            name=signal_spec.get("label", signal_name),
            pen=pg.mkPen(
                signal_spec.get("color", "#202020"),
                width=signal_spec.get("width", 1.5),
            ),
        )

        curves[spec["title"]][signal_name] = curve


# ============================================================
# Link all time axes
# ============================================================

if plots:

    first_plot = next(iter(plots.values()))

    for plot in list(plots.values())[1:]:
        plot.setXLink(first_plot)


# ============================================================
# Plot updating
# ============================================================

def update_plots():

    receive_can()
    trim_buffers()

    for spec in PLOT_SPECS:

        plot_curves = curves[spec["title"]]

        for signal_name in spec["signals"]:

            signal = signals[signal_name]

            plot_curves[signal_name].setData(
                signal.time,
                signal.data,
            )


# ============================================================
# GUI timer
# ============================================================

timer = QTimer()

timer.timeout.connect(update_plots)

timer.start(GUI_UPDATE_MS)


# ============================================================
# Run
# ============================================================

window.show()

sys.exit(app.exec())
