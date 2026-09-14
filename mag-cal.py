#!/usr/bin/env python3
"""
Write config values to the RT-IMU CAN node over SocketCAN (e.g. vcan0).

===============================================================================
WIRE FORMAT  (see handle_config_packet() in rt-imu-espidf.c)
===============================================================================
  CAN ID       : node_id + 0x400   (standard/11-bit frame)
  DLC          : 8  (firmware drops the frame if it's anything else)
  byte 0       : opcode
  bytes 1-3    : reserved, always 0x00
  bytes 4-7    : value, little-endian
                   - float32 for calibration values (hard/soft iron, beta)
                   - uint32  for node id / send rate

===============================================================================
HOW TO ADD A NEW CONFIG OPTION
===============================================================================
This script only wires up hard-iron and soft-iron right now. Everything else
the firmware supports (node id, send rate, beta, reset...) follows the exact
same pattern. To add one:

  1. Add its opcode to the OPCODES section below (copy the value straight out
     of the `#define CAN_CONFIG_...` lines in rt-imu-espidf.c).

  2. Write a small `set_xxx()` function that calls `send_config()` with the
     right opcode + encoded value. Look at set_hard_iron() just below for
     the shape. Use encode_float() for calibration-type values, or
     encode_u32() for node id / send rate.

  3. Add an argparse flag for it in parse_args(), and call your new
     set_xxx() function in main().

That's genuinely the whole recipe - nothing else in the file needs to change.
"""

import argparse
import struct
import sys
import time

import can

# ==============================================================================
# CONSTANTS
# ==============================================================================

CONFIG_ID_OFFSET = 0x400        # config CAN ID = node_id + this
DEFAULT_NODE_ID = 0x11          # APP_CONFIG_DEFAULT_NODE_ID in app_config.h

# --- Opcodes (byte 0 of the config frame) ------------------------------------
# Pulled directly from the #defines at the top of rt-imu-espidf.c.
# Add new ones here as you wire them up.

OPCODE_HARD_IRON_X = 0x10
OPCODE_HARD_IRON_Y = 0x11
OPCODE_HARD_IRON_Z = 0x12
OPCODE_SOFT_IRON_0 = 0x13       # soft_iron[1..8] are 0x14..0x1B, sequential

# Not implemented yet - uncomment / use as you add support for them:
# OPCODE_NODE_ID     = 0x01     # value = uint32 (new node id)
# OPCODE_SEND_RATE   = 0x02     # value = uint32 (CAN TX period, microseconds)
# OPCODE_BETA        = 0x09     # value = float32 (Madgwick filter beta)
OPCODE_RESET       = 0x99     # value = uint32, must be exactly 0x01


# ==============================================================================
# LOW-LEVEL FRAME BUILDING
# ==============================================================================
# You shouldn't need to touch this section when adding a new config option -
# it's generic. It's only here so the encode/send logic isn't duplicated.

def encode_float(value: float) -> bytes:
    """Pack a value into the 4 little-endian payload bytes as float32."""
    return struct.pack("<f", value)


def encode_u32(value: int) -> bytes:
    """Pack a value into the 4 little-endian payload bytes as uint32."""
    return struct.pack("<I", value)


def build_config_frame(node_id: int, opcode: int, payload: bytes) -> can.Message:
    """
    Build one 8-byte config CAN frame.

    payload must be exactly 4 bytes (the output of encode_float/encode_u32) -
    it goes into bytes 4-7. Bytes 1-3 are always reserved/zero.
    """
    assert len(payload) == 4, "config value payload must be 4 bytes"
    data = bytes([opcode, 0x00, 0x00, 0x00]) + payload
    return can.Message(
        arbitration_id=node_id + CONFIG_ID_OFFSET,
        data=data,
        is_extended_id=False,
        dlc=8,
    )


def send_config(bus: can.BusABC, node_id: int, opcode: int, payload: bytes,
                 label: str = "", delay: float = 0.02):
    """
    Build + send one config frame, with a print for visibility.

    This is the function every set_xxx() below should call - it's the one
    place that actually touches the bus.
    """
    msg = build_config_frame(node_id, opcode, payload)
    bus.send(msg)
    print(f"  {label or f'opcode 0x{opcode:02X}'}  "
          f"(id=0x{msg.arbitration_id:03X} data={msg.data.hex()})")
    time.sleep(delay)  # be gentle - each write triggers an nvs_commit() on the ESP32


# ==============================================================================
# CONFIG OPTIONS
# ==============================================================================
# One function per logical config item. This is the section you'll extend.
# Each function just figures out the right opcode(s) + encoding and calls
# send_config(). Keep them small like this - it makes new ones easy to add
# and easy to test individually from a REPL.

def set_hard_iron(bus: can.BusABC, node_id: int, x: float, y: float, z: float,
                   delay: float = 0.02):
    """Write hard iron offsets (uT). Sent as 3 separate frames, X then Y then Z."""
    send_config(bus, node_id, OPCODE_HARD_IRON_X, encode_float(x),
                label=f"hard_iron.x = {x:.6f}", delay=delay)
    send_config(bus, node_id, OPCODE_HARD_IRON_Y, encode_float(y),
                label=f"hard_iron.y = {y:.6f}", delay=delay)
    send_config(bus, node_id, OPCODE_HARD_IRON_Z, encode_float(z),
                label=f"hard_iron.z = {z:.6f}", delay=delay)


def set_soft_iron(bus: can.BusABC, node_id: int, matrix_3x3, delay: float = 0.02):
    """
    Write the soft iron 3x3 matrix. `matrix_3x3` is a flat list/tuple of 9
    floats, row-major (matches soft_iron[9] in app_config.c). Sent as 9
    separate frames, one per element, at OPCODE_SOFT_IRON_0 + index.
    """
    assert len(matrix_3x3) == 9, "soft iron matrix needs exactly 9 values"
    for idx, value in enumerate(matrix_3x3):
        send_config(bus, node_id, OPCODE_SOFT_IRON_0 + idx, encode_float(value),
                    label=f"soft_iron[{idx}] = {value:.6f}", delay=delay)

def set_reset_device(bus: can.BusABC, node_id: int, delay: float = 0.02):
    """Reset the Device"""
    send_config(bus, node_id, OPCODE_RESET, encode_u32(0x01),
                label=f"Sends 01 to 0x99 to CONFIG", delay=delay)

# --- TEMPLATE for your next config option ------------------------------------
# Copy/paste this and fill it in. Example shown for beta, commented out.
#
# def set_beta(bus: can.BusABC, node_id: int, beta: float, delay: float = 0.02):
#     """Write the Madgwick filter beta gain."""
#     send_config(bus, node_id, OPCODE_BETA, encode_float(beta),
#                 label=f"beta = {beta:.6f}", delay=delay)
#
# def set_node_id(bus: can.BusABC, node_id: int, new_node_id: int, delay: float = 0.02):
#     """Change the node's own ID. NOTE: uses uint32 encoding, not float!"""
#     send_config(bus, node_id, OPCODE_NODE_ID, encode_u32(new_node_id),
#                 label=f"node_id -> 0x{new_node_id:02X}", delay=delay)


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--channel", default="vcan0",
                    help="SocketCAN interface (default: vcan0)")
    p.add_argument("--node-id", type=lambda x: int(x, 0), default=DEFAULT_NODE_ID,
                    help="Target node ID (default: 0x%02X)" % DEFAULT_NODE_ID)
    p.add_argument("--delay", type=float, default=0.02,
                    help="Delay between frames in seconds (default: 0.02)")

    p.add_argument("--hard-iron", nargs=3, type=float, metavar=("X", "Y", "Z"),
                    help="Hard iron offsets in uT, e.g. --hard-iron -1.05 -0.10 1.16")
    p.add_argument("--soft-iron", nargs=9, type=float,
                    metavar=("M00", "M01", "M02", "M10", "M11", "M12", "M20", "M21", "M22"),
                    help="Soft iron 3x3 matrix, row-major, 9 values")

    p.add_argument("--reset_device", action="store_true", help="Reset the Device")

    # Add your own flag here as you add new set_xxx() functions, e.g.:
    # p.add_argument("--beta", type=float, help="Madgwick filter beta")

    return p.parse_args()


def main():
    args = parse_args()

    """
    if not args.hard_iron and not args.soft_iron:
        print("Nothing to do: pass --hard-iron and/or --soft-iron", file=sys.stderr)
        sys.exit(1)
        """

    bus = can.Bus(channel=args.channel, bustype="socketcan")

    try:
        print(f"Node ID: 0x{args.node_id:02X}  "
              f"(config CAN ID: 0x{args.node_id + CONFIG_ID_OFFSET:03X})")

        if args.hard_iron:
            print("Hard iron:")
            set_hard_iron(bus, args.node_id, *args.hard_iron, delay=args.delay)

        if args.soft_iron:
            print("Soft iron:")
            set_soft_iron(bus, args.node_id, args.soft_iron, delay=args.delay)

        if args.reset_device:
            print("Reset:")
            set_reset_device(bus, args.node_id, delay=args.delay)

        # Call your new set_xxx() here once wired up, e.g.:
        # if args.beta is not None:
        #     print("Beta:")
        #     set_beta(bus, args.node_id, args.beta, delay=args.delay)

        print("Done.")
    finally:
        bus.shutdown()


if __name__ == "__main__":
    main()
