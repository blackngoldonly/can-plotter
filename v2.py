#!/usr/bin/env python3

import argparse
import math
import struct
import time

import can
import pygame


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

WIDTH = 1000
HEIGHT = 800

BG_COLOR = (20, 20, 24)
LINE_COLOR = (230, 230, 230)
TEXT_COLOR = (220, 220, 220)

AXIS_X_COLOR = (255, 80, 80)
AXIS_Y_COLOR = (80, 255, 80)
AXIS_Z_COLOR = (80, 140, 255)

CUBE_SIZE = 2.0

CAMERA_DISTANCE = 6.0
FOCAL_LENGTH = 500.0


# ---------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------

def normalize_quaternion(q):
    w, x, y, z = q

    mag = math.sqrt(w*w + x*x + y*y + z*z)

    if mag < 1e-9:
        return 1.0, 0.0, 0.0, 0.0

    return (
        w / mag,
        x / mag,
        y / mag,
        z / mag,
    )


def quaternion_rotate_vector(q, v):
    """
    Rotate vector v by quaternion q.

    q = (w, x, y, z)
    """

    w, x, y, z = q
    vx, vy, vz = v

    # Efficient equivalent of:
    #
    #     q * [0, v] * conjugate(q)
    #
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)

    rx = vx + w * tx + (y * tz - z * ty)
    ry = vy + w * ty + (z * tx - x * tz)
    rz = vz + w * tz + (x * ty - y * tx)

    return rx, ry, rz


def quaternion_to_euler(q):
    """
    Debug/display only.

    Returns roll, pitch, yaw in degrees.
    Quaternion itself remains the actual representation used for rendering.
    """

    w, x, y, z = q

    # Roll
    sinr_cosp = 2.0 * (w*x + y*z)
    cosr_cosp = 1.0 - 2.0 * (x*x + y*y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch
    sinp = 2.0 * (w*y - z*x)

    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw
    siny_cosp = 2.0 * (w*z + x*y)
    cosy_cosp = 1.0 - 2.0 * (y*y + z*z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return (
        math.degrees(roll),
        math.degrees(pitch),
        math.degrees(yaw),
    )


# ---------------------------------------------------------------------
# CAN
# ---------------------------------------------------------------------

def decode_quaternion(data):
    """
    CAN payload:

        byte 0..1 = w int16 little endian
        byte 2..3 = x int16 little endian
        byte 4..5 = y int16 little endian
        byte 6..7 = z int16 little endian

    Physical value = int16 / 10000
    """

    if len(data) < 8:
        return None

    w_raw, x_raw, y_raw, z_raw = struct.unpack("<hhhh", bytes(data[:8]))

    q = (
        w_raw / 10000.0,
        x_raw / 10000.0,
        y_raw / 10000.0,
        z_raw / 10000.0,
    )

    return normalize_quaternion(q)


# ---------------------------------------------------------------------
# Cube geometry
# ---------------------------------------------------------------------

S = CUBE_SIZE / 2.0

CUBE_VERTICES = [
    (-S, -S, -S),
    (+S, -S, -S),
    (+S, +S, -S),
    (-S, +S, -S),

    (-S, -S, +S),
    (+S, -S, +S),
    (+S, +S, +S),
    (-S, +S, +S),
]

CUBE_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),

    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),

    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]


# ---------------------------------------------------------------------
# 3D projection
# ---------------------------------------------------------------------

def project(v):
    x, y, z = v

    # Move object away from camera
    depth = z + CAMERA_DISTANCE

    if depth < 0.1:
        depth = 0.1

    scale = FOCAL_LENGTH / depth

    screen_x = WIDTH / 2 + x * scale

    # Pygame Y goes downward, so invert world Y
    screen_y = HEIGHT / 2 - y * scale

    return int(screen_x), int(screen_y)


def draw_axis(screen, q, axis_vector, color, label, font):
    origin = (0.0, 0.0, 0.0)

    rotated_axis = quaternion_rotate_vector(q, axis_vector)

    p0 = project(origin)
    p1 = project(rotated_axis)

    pygame.draw.line(screen, color, p0, p1, 4)

    text = font.render(label, True, color)
    screen.blit(text, (p1[0] + 6, p1[1] + 6))


def draw_cube(screen, q):
    rotated_vertices = [
        quaternion_rotate_vector(q, v)
        for v in CUBE_VERTICES
    ]

    projected_vertices = [
        project(v)
        for v in rotated_vertices
    ]

    for a, b in CUBE_EDGES:
        pygame.draw.line(
            screen,
            LINE_COLOR,
            projected_vertices[a],
            projected_vertices[b],
            3,
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Realtime CAN quaternion cube viewer"
    )

    parser.add_argument(
        "--interface",
        default="vcan0",
        help="SocketCAN interface, e.g. vcan0 or can0",
    )

    parser.add_argument(
        "--can-id",
        type=lambda x: int(x, 0),
        default=0x493,
        help="CAN arbitration ID, e.g. 0x493",
    )

    args = parser.parse_args()

    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(
        f"CAN Quaternion Viewer - {args.interface} - 0x{args.can_id:X}"
    )

    clock = pygame.time.Clock()

    font = pygame.font.SysFont("monospace", 22)
    small_font = pygame.font.SysFont("monospace", 18)

    # SocketCAN bus
    bus = can.Bus(
        interface="socketcan",
        channel=args.interface,
    )

    # Kernel/socket filtering so we mostly receive only the desired ID
    bus.set_filters([
        {
            "can_id": args.can_id,
            "can_mask": 0x7FF,
            "extended": False,
        }
    ])

    quaternion = (1.0, 0.0, 0.0, 0.0)

    last_packet_time = None
    packet_count = 0
    last_raw_data = None

    running = True

    while running:

        # -------------------------------------------------------------
        # Pygame events
        # -------------------------------------------------------------

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # -------------------------------------------------------------
        # Drain CAN RX queue
        #
        # timeout=0 makes this non-blocking. We consume every currently
        # available frame but render only the latest quaternion.
        # -------------------------------------------------------------

        while True:

            msg = bus.recv(timeout=0.0)

            if msg is None:
                break

            if msg.arbitration_id != args.can_id:
                continue

            q = decode_quaternion(msg.data)

            if q is not None:
                quaternion = q
                last_packet_time = time.monotonic()
                packet_count += 1
                last_raw_data = bytes(msg.data)

        # -------------------------------------------------------------
        # Rendering
        # -------------------------------------------------------------

        screen.fill(BG_COLOR)

        draw_cube(screen, quaternion)

        # Body-frame axes
        draw_axis(
            screen,
            quaternion,
            (2.0, 0.0, 0.0),
            AXIS_X_COLOR,
            "+X",
            small_font,
        )

        draw_axis(
            screen,
            quaternion,
            (0.0, 2.0, 0.0),
            AXIS_Y_COLOR,
            "+Y",
            small_font,
        )

        draw_axis(
            screen,
            quaternion,
            (0.0, 0.0, 2.0),
            AXIS_Z_COLOR,
            "+Z",
            small_font,
        )

        # -------------------------------------------------------------
        # Text
        # -------------------------------------------------------------

        w, x, y, z = quaternion
        roll, pitch, yaw = quaternion_to_euler(quaternion)

        lines = [
            f"Interface : {args.interface}",
            f"CAN ID    : 0x{args.can_id:03X}",
            "",
            f"w : {w:+.5f}",
            f"x : {x:+.5f}",
            f"y : {y:+.5f}",
            f"z : {z:+.5f}",
            "",
            f"Roll  : {roll:+8.2f} deg",
            f"Pitch : {pitch:+8.2f} deg",
            f"Yaw   : {yaw:+8.2f} deg",
            "",
            f"Packets: {packet_count}",
        ]

        if last_packet_time is None:
            lines.append("CAN: waiting...")
        else:
            age = time.monotonic() - last_packet_time

            if age < 0.25:
                lines.append(f"CAN: LIVE ({age*1000:.0f} ms)")
            else:
                lines.append(f"CAN: STALE ({age:.2f} s)")

        if last_raw_data is not None:
            raw_string = " ".join(f"{b:02X}" for b in last_raw_data)
            lines.append("")
            lines.append(raw_string)

        y_pos = 20

        for line in lines:
            text = font.render(line, True, TEXT_COLOR)
            screen.blit(text, (20, y_pos))
            y_pos += 26

        pygame.display.flip()

        # Rendering at 120 FPS is independent of CAN update rate.
        clock.tick(120)

    bus.shutdown()
    pygame.quit()


if __name__ == "__main__":
    main()
