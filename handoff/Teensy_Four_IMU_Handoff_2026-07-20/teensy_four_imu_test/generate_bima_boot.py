from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps


WIDTH = 128
HEIGHT = 64
SCALE = 4
FRAME_COUNT = 160
ROOT = Path(__file__).resolve().parent
HEADER_PATH = ROOT / "bima_boot_frames.h"
PREVIEW_PATH = ROOT / "bima_boot_preview.png"
END_LOGO_PREVIEW_PATH = ROOT / "bima_end_logo_preview.png"
LOGO_SOURCE_PATH = ROOT / "bima_logo_source.png"


def ease(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size * SCALE)


FONT_BOLD = r"C:\Windows\Fonts\bahnschrift.ttf"
FONT_UI = r"C:\Windows\Fonts\segoeui.ttf"
FONT_UI_BOLD = r"C:\Windows\Fonts\segoeuib.ttf"


def rotate_point(x: float, y: float, z: float, yaw: float, pitch: float):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    rx = x * cy - z * sy
    rz = x * sy + z * cy
    ry = y * cp - rz * sp
    rz2 = y * sp + rz * cp
    return rx, ry, rz2


def project(point, yaw: float, pitch: float, zoom: float):
    x, y, z = rotate_point(*point, yaw, pitch)
    depth = z + 4.5
    factor = 86.0 * zoom / depth
    return (64.0 + x * factor, 29.0 + y * factor, z)


def scaled(points):
    return [(round(x * SCALE), round(y * SCALE)) for x, y in points]


def draw_curve(draw: ImageDraw.ImageDraw, samples, yaw, pitch, zoom):
    projected = [project(point, yaw, pitch, zoom) for point in samples]
    for index in range(1, len(projected)):
        a = projected[index - 1]
        b = projected[index]
        front = (a[2] + b[2]) * 0.5 <= 0.15
        if front:
            draw.line(scaled([(a[0], a[1]), (b[0], b[1])]), fill=255, width=5)
        elif index % 4 == 0:
            draw.line(scaled([(a[0], a[1]), (b[0], b[1])]), fill=175, width=3)


def sphere_curves(draw, yaw: float, pitch: float, zoom: float):
    steps = 56
    for longitude in (-1.05, -0.35, 0.35, 1.05):
        samples = []
        for step in range(steps + 1):
            latitude = -math.pi / 2 + math.pi * step / steps
            samples.append(
                (
                    math.cos(latitude) * math.cos(longitude),
                    math.sin(latitude),
                    math.cos(latitude) * math.sin(longitude),
                )
            )
        draw_curve(draw, samples, yaw, pitch, zoom)

    for latitude in (-0.65, 0.0, 0.65):
        radius = math.cos(latitude)
        samples = []
        for step in range(steps + 1):
            longitude = math.tau * step / steps
            samples.append(
                (
                    radius * math.cos(longitude),
                    math.sin(latitude),
                    radius * math.sin(longitude),
                )
            )
        draw_curve(draw, samples, yaw, pitch, zoom)


def draw_orbit_nodes(draw, time_value: float, yaw: float, pitch: float, zoom: float):
    nodes = []
    for index in range(4):
        angle = time_value * 1.4 + index * math.pi / 2
        point = (
            math.cos(angle) * 1.48,
            math.sin(angle * 1.7 + index) * 0.35,
            math.sin(angle) * 1.48,
        )
        sx, sy, depth = project(point, yaw * 0.25, pitch, zoom)
        radius = max(1.7, min(4.0, 2.7 - depth * 0.55))
        nodes.append((depth, sx, sy, radius))

    for _, sx, sy, radius in sorted(nodes, reverse=True):
        box = [
            round((sx - radius) * SCALE),
            round((sy - radius) * SCALE),
            round((sx + radius) * SCALE),
            round((sy + radius) * SCALE),
        ]
        draw.ellipse(box, fill=0, outline=255, width=5)
        inner = max(0.8, radius * 0.34)
        draw.ellipse(
            [
                round((sx - inner) * SCALE),
                round((sy - inner) * SCALE),
                round((sx + inner) * SCALE),
                round((sy + inner) * SCALE),
            ],
            fill=255,
        )


def draw_pulse(draw, phase: float, strength: float = 1.0):
    shape = [0, 0, -1, -6, 9, -4, 0, 0, 1, 0]
    points = []
    for index, value in enumerate(shape):
        x = 50 + index * 3.1
        y = 29 + value * 0.55 * strength
        points.append((x, y))
    draw.line(scaled(points), fill=255, width=6, joint="curve")
    pulse_x = 50 + (phase % 1.0) * 28
    draw.ellipse(
        [
            round((pulse_x - 1.5) * SCALE),
            round((29 - 1.5) * SCALE),
            round((pulse_x + 1.5) * SCALE),
            round((29 + 1.5) * SCALE),
        ],
        fill=255,
    )


def draw_stars(draw, frame: int, intensity: float):
    seeds = [
        (9, 12), (21, 41), (34, 8), (48, 51), (78, 7),
        (94, 45), (111, 14), (121, 37), (16, 28), (105, 28),
    ]
    for index, (base_x, base_y) in enumerate(seeds):
        drift = (frame * (1 + index % 3)) % 18
        x = 64 + (base_x - 64) * (1.0 + drift * 0.018)
        y = 30 + (base_y - 30) * (1.0 + drift * 0.018)
        radius = 0.45 + (index % 3) * 0.22
        shade = int(125 + intensity * 130)
        draw.ellipse(
            [
                round((x - radius) * SCALE),
                round((y - radius) * SCALE),
                round((x + radius) * SCALE),
                round((y + radius) * SCALE),
            ],
            fill=shade,
        )


def draw_logo(canvas: Image.Image, reveal: float = 1.0):
    draw = ImageDraw.Draw(canvas)
    mark_x, mark_y = 22, 22
    draw.ellipse(
        scaled([(mark_x - 12, mark_y - 12), (mark_x + 12, mark_y + 12)]),
        outline=255,
        width=7,
    )
    pulse = [
        (mark_x - 11, mark_y), (mark_x - 5, mark_y),
        (mark_x - 2, mark_y - 7), (mark_x + 2, mark_y + 8),
        (mark_x + 6, mark_y - 4), (mark_x + 9, mark_y),
        (mark_x + 12, mark_y),
    ]
    draw.line(scaled(pulse), fill=255, width=6, joint="curve")

    title_font = font(FONT_BOLD, 25)
    draw.text((39 * SCALE, 7 * SCALE), "BIMA", font=title_font, fill=255, stroke_width=0)

    subtitle_font = font(FONT_UI_BOLD, 7)
    draw.text((8 * SCALE, 43 * SCALE), "BIOMETRIC INFANT MOVEMENT", font=subtitle_font, fill=255)
    draw.text((35 * SCALE, 52 * SCALE), "ASSESSMENT", font=subtitle_font, fill=255)

    if reveal < 1.0:
        mask_height = round((1.0 - reveal) * HEIGHT * SCALE)
        draw.rectangle([0, 0, WIDTH * SCALE, mask_height], fill=0)
        draw.line(
            [(0, mask_height), (WIDTH * SCALE, mask_height)],
            fill=255,
            width=3,
        )


@lru_cache(maxsize=1)
def source_end_logo_layer() -> Image.Image:
    """Convert the supplied BIMA artwork into a legible 128x64 OLED lockup."""
    if not LOGO_SOURCE_PATH.exists():
        raise FileNotFoundError(f"Missing supplied BIMA logo: {LOGO_SOURCE_PATH}")

    source = Image.open(LOGO_SOURCE_PATH).convert("RGB")
    source_width, source_height = source.size

    # The top portion of the supplied square contains the infant-and-swoosh
    # emblem. The original wordmark is redrawn below at OLED resolution so it
    # stays readable instead of collapsing into noise when the image is reduced.
    emblem = source.crop((
        round(source_width * 0.19),
        round(source_height * 0.035),
        round(source_width * 0.82),
        round(source_height * 0.655),
    ))
    emblem_mask = ImageOps.invert(ImageOps.grayscale(emblem))
    emblem_mask = emblem_mask.point(lambda pixel: 255 if pixel >= 18 else 0, mode="L")
    content_bounds = emblem_mask.getbbox()
    if content_bounds:
        emblem_mask = emblem_mask.crop(content_bounds)

    max_width = 52 * SCALE
    max_height = 43 * SCALE
    fit = min(max_width / emblem_mask.width, max_height / emblem_mask.height)
    emblem_mask = emblem_mask.resize(
        (max(1, round(emblem_mask.width * fit)), max(1, round(emblem_mask.height * fit))),
        Image.Resampling.LANCZOS,
    )

    layer = Image.new("L", (WIDTH * SCALE, HEIGHT * SCALE), 0)
    emblem_x = (layer.width - emblem_mask.width) // 2
    layer.paste(255, (emblem_x, 0), emblem_mask)

    draw = ImageDraw.Draw(layer)
    title_font = font(FONT_BOLD, 17)
    title_box = draw.textbbox((0, 0), "BIMA", font=title_font)
    title_width = title_box[2] - title_box[0]
    title_x = (layer.width - title_width) // 2 - title_box[0]
    title_y = 44 * SCALE - title_box[1]
    draw.text((title_x, title_y), "BIMA", font=title_font, fill=255)
    return layer


def draw_source_end_logo(canvas: Image.Image, reveal: float = 1.0):
    """Scan-reveal the final logo, then hold the exact converted lockup."""
    reveal = ease(reveal)
    logo = source_end_logo_layer()
    if reveal >= 1.0:
        canvas.paste(ImageChops.lighter(canvas, logo))
        return

    scan_y = round((1.0 - reveal) * HEIGHT * SCALE)
    reveal_mask = Image.new("L", canvas.size, 0)
    ImageDraw.Draw(reveal_mask).rectangle(
        [0, scan_y, WIDTH * SCALE, HEIGHT * SCALE],
        fill=255,
    )
    canvas.paste(ImageChops.lighter(canvas, logo), (0, 0), reveal_mask)
    if 0 <= scan_y < HEIGHT * SCALE:
        ImageDraw.Draw(canvas).line(
            [(0, scan_y), (WIDTH * SCALE, scan_y)],
            fill=255,
            width=3,
        )


def composite_lighten(canvas: Image.Image, layer: Image.Image, x: int, y: int):
    positioned = Image.new("L", canvas.size, 0)
    positioned.paste(layer, (x, y))
    canvas.paste(ImageChops.lighter(canvas, positioned))


def draw_rocket(
    canvas: Image.Image,
    center_x: float,
    center_y: float,
    size: float,
    tilt: float,
    flame_phase: float,
    engines_on: bool = True,
):
    patch_w = 42 * SCALE
    patch_h = 72 * SCALE
    layer = Image.new("L", (patch_w, patch_h), 0)
    draw = ImageDraw.Draw(layer)
    cx = patch_w // 2

    nose_y = 5 * SCALE
    shoulder_y = 17 * SCALE
    base_y = 45 * SCALE
    half_body = 6 * SCALE
    outline = max(3, round(1.3 * SCALE))

    body = [
        (cx, nose_y),
        (cx + half_body, shoulder_y),
        (cx + half_body, base_y),
        (cx - half_body, base_y),
        (cx - half_body, shoulder_y),
    ]
    draw.polygon(body, fill=0, outline=255)
    draw.line(body + [body[0]], fill=255, width=outline, joint="curve")

    left_fin = [
        (cx - half_body, 33 * SCALE),
        (cx - 12 * SCALE, 48 * SCALE),
        (cx - half_body, 44 * SCALE),
    ]
    right_fin = [
        (cx + half_body, 33 * SCALE),
        (cx + 12 * SCALE, 48 * SCALE),
        (cx + half_body, 44 * SCALE),
    ]
    for fin in (left_fin, right_fin):
        draw.polygon(fin, fill=0, outline=255)
        draw.line(fin + [fin[0]], fill=255, width=outline)

    draw.ellipse(
        [cx - 3 * SCALE, 20 * SCALE, cx + 3 * SCALE, 26 * SCALE],
        fill=255,
    )
    draw.line(
        [(cx - half_body, 31 * SCALE), (cx + half_body, 31 * SCALE)],
        fill=255,
        width=max(2, SCALE),
    )

    if engines_on:
        flicker = math.sin(flame_phase * math.tau) * 2.0
        outer_flame = [
            (cx - 5 * SCALE, 47 * SCALE),
            (cx, round((65 + flicker) * SCALE)),
            (cx + 5 * SCALE, 47 * SCALE),
        ]
        inner_flame = [
            (cx - 2 * SCALE, 48 * SCALE),
            (cx, round((57 - flicker * 0.4) * SCALE)),
            (cx + 2 * SCALE, 48 * SCALE),
        ]
        draw.polygon(outer_flame, fill=255)
        draw.polygon(inner_flame, fill=0)

    if abs(size - 1.0) > 0.001:
        layer = layer.resize(
            (max(1, round(layer.width * size)), max(1, round(layer.height * size))),
            Image.Resampling.LANCZOS,
        )
    if abs(tilt) > 0.01:
        layer = layer.rotate(tilt, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=0)

    left = round(center_x * SCALE - layer.width / 2)
    top = round(center_y * SCALE - layer.height * 0.42)
    composite_lighten(canvas, layer, left, top)


def draw_launch_pad(draw: ImageDraw.ImageDraw):
    draw.line(scaled([(0, 55), (128, 55)]), fill=255, width=5)
    draw.line(scaled([(45, 55), (45, 15)]), fill=255, width=4)
    draw.line(scaled([(45, 19), (57, 19)]), fill=255, width=4)
    draw.line(scaled([(45, 33), (57, 33)]), fill=255, width=4)
    draw.line(scaled([(39, 55), (45, 15), (51, 55)]), fill=255, width=3)
    draw.rectangle(scaled([(58, 52), (70, 56)]), fill=255)


def draw_launch_smoke(draw: ImageDraw.ImageDraw, age: float):
    puff_count = min(16, max(0, int(age * 0.55)))
    for index in range(puff_count):
        puff_age = max(0.0, age - index * 1.7)
        spread = 3.5 + puff_age * 0.48
        x = 64 + math.sin(index * 2.17) * spread
        y = 54 + min(8.0, puff_age * 0.28) + math.cos(index * 1.3) * 1.5
        radius = 1.8 + min(7.0, puff_age * 0.22)
        draw.ellipse(
            scaled([(x - radius, y - radius), (x + radius, y + radius)]),
            fill=0,
            outline=255,
            width=max(3, SCALE),
        )


def draw_earth_horizon(draw: ImageDraw.ImageDraw, progress: float):
    top = 47 + progress * 13
    draw.arc(
        scaled([(-72, top), (200, top + 88)]),
        start=188,
        end=352,
        fill=255,
        width=6,
    )
    if progress < 0.75:
        draw.arc(
            scaled([(-53, top + 5), (181, top + 78)]),
            start=197,
            end=343,
            fill=175,
            width=3,
        )


def draw_space_field(draw: ImageDraw.ImageDraw, frame: int, speed: float = 1.0):
    stars = [
        (7, 10), (17, 35), (29, 17), (42, 5), (52, 43), (68, 12),
        (80, 38), (91, 8), (103, 27), (116, 15), (123, 48), (34, 55),
    ]
    for index, (base_x, base_y) in enumerate(stars):
        drift = frame * speed * (0.18 + index % 3 * 0.08)
        x = (base_x - drift) % WIDTH
        radius = 0.45 + (index % 4 == 0) * 0.45
        draw.ellipse(
            scaled([(x - radius, base_y - radius), (x + radius, base_y + radius)]),
            fill=255,
        )


def draw_orbital_gateway(draw: ImageDraw.ImageDraw, frame: int, progress: float):
    cx, cy = 64, 30
    core_radius = 5 + ease(progress) * 13
    angle = frame * 7
    draw.arc(
        scaled([(cx - core_radius, cy - core_radius), (cx + core_radius, cy + core_radius)]),
        start=angle,
        end=angle + 245,
        fill=255,
        width=6,
    )
    draw.arc(
        scaled([(cx - core_radius - 5, cy - core_radius - 5),
                (cx + core_radius + 5, cy + core_radius + 5)]),
        start=-angle * 0.65,
        end=-angle * 0.65 + 210,
        fill=255,
        width=4,
    )

    for sensor in range(4):
        orbit_angle = frame * 0.09 + sensor * math.pi / 2
        orbit_x = cx + math.cos(orbit_angle) * (core_radius + 9)
        orbit_y = cy + math.sin(orbit_angle) * (core_radius * 0.42 + 5)
        node_radius = 2.2 + math.sin(orbit_angle) * 0.5
        draw.ellipse(
            scaled([(orbit_x - node_radius, orbit_y - node_radius),
                    (orbit_x + node_radius, orbit_y + node_radius)]),
            fill=0,
            outline=255,
            width=4,
        )
        draw.ellipse(
            scaled([(orbit_x - 0.8, orbit_y - 0.8),
                    (orbit_x + 0.8, orbit_y + 0.8)]),
            fill=255,
        )

    pulse = [(52, 30), (57, 30), (60, 23), (64, 38), (68, 26), (72, 30), (76, 30)]
    draw.line(scaled(pulse), fill=255, width=5, joint="curve")


def draw_complete_orbital_bima_sequence(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    frame: int,
):
    """The complete earlier 72-frame space/BIMA film, preserved intact."""
    if frame < 16:
        progress = ease(frame / 15)
        draw_stars(draw, frame, progress)
        zoom = 0.18 + progress * 0.70
        yaw = frame * 0.055
        sphere_curves(draw, yaw, -0.20, zoom)
        draw_orbit_nodes(draw, frame * 0.09, yaw, -0.20, zoom)
        if frame > 7:
            draw_pulse(draw, frame * 0.08, progress)

        label_font = font(FONT_UI_BOLD, 8)
        draw.text((8 * SCALE, 51 * SCALE), "BIMA", font=label_font, fill=255)
        draw.rounded_rectangle(
            [42 * SCALE, 54 * SCALE, 120 * SCALE, 60 * SCALE],
            radius=2 * SCALE,
            outline=255,
            width=3,
        )
        progress_width = round(74 * progress)
        if progress_width > 0:
            draw.rounded_rectangle(
                [44 * SCALE, 56 * SCALE, (44 + progress_width) * SCALE, 58 * SCALE],
                radius=SCALE,
                fill=255,
            )

    elif frame < 40:
        local = frame - 16
        draw_stars(draw, frame, 0.65)
        yaw = 0.85 + local * 0.075
        pitch = -0.20 + math.sin(local * 0.11) * 0.08
        zoom = 0.88 + math.sin(local * math.pi / 23) * 0.08
        sphere_curves(draw, yaw, pitch, zoom)
        draw_orbit_nodes(draw, frame * 0.09, yaw, pitch, zoom)
        draw_pulse(draw, frame * 0.075, 1.0)
        label_font = font(FONT_UI_BOLD, 7)
        draw.text((27 * SCALE, 2 * SCALE), "SENSOR FUSION", font=label_font, fill=255)

    elif frame < 50:
        local = frame - 40
        progress = ease(local / 9)
        yaw = 2.65 + local * 0.12
        zoom = 0.95 + progress * 3.2
        sphere_curves(draw, yaw, -0.18, zoom)
        draw_orbit_nodes(draw, frame * 0.09, yaw, -0.18, zoom)
        draw_pulse(draw, frame * 0.075, 1.0 + progress)

    elif frame == 50:
        draw.rectangle([0, 0, WIDTH * SCALE, HEIGHT * SCALE], fill=255)

    elif frame < 61:
        reveal = ease((frame - 51) / 9)
        draw_logo(canvas, reveal)

    else:
        draw_logo(canvas, 1.0)


def cubic_bezier(p0, p1, p2, p3, t):
    inverse = 1.0 - t
    return (
        inverse ** 3 * p0[0]
        + 3 * inverse * inverse * t * p1[0]
        + 3 * inverse * t * t * p2[0]
        + t ** 3 * p3[0],
        inverse ** 3 * p0[1]
        + 3 * inverse * inverse * t * p1[1]
        + 3 * inverse * t * t * p2[1]
        + t ** 3 * p3[1],
    )


def draw_signal_path(draw, start, control_a, control_b, end, progress):
    point_count = max(2, round(34 * progress))
    points = [
        cubic_bezier(start, control_a, control_b, end, progress * i / (point_count - 1))
        for i in range(point_count)
    ]
    draw.line(scaled(points), fill=255, width=5, joint="curve")

    head_x, head_y = points[-1]
    draw.ellipse(
        scaled([(head_x - 2.2, head_y - 2.2), (head_x + 2.2, head_y + 2.2)]),
        fill=0,
        outline=255,
        width=4,
    )
    draw.ellipse(
        scaled([(head_x - 0.75, head_y - 0.75), (head_x + 0.75, head_y + 0.75)]),
        fill=255,
    )


def draw_sensor_seed(draw, x, y, phase, scale=1.0):
    radius = (2.7 + math.sin(phase) * 0.45) * scale
    draw.ellipse(
        scaled([(x - radius, y - radius), (x + radius, y + radius)]),
        fill=0,
        outline=255,
        width=max(3, round(1.1 * SCALE * scale)),
    )
    draw.ellipse(
        scaled([(x - 0.8 * scale, y - 0.8 * scale),
                (x + 0.8 * scale, y + 0.8 * scale)]),
        fill=255,
    )


def petal_points(cx, cy, angle, length, width, completion=1.0):
    count = max(3, round(48 * completion))
    points = []
    radial_x, radial_y = math.cos(angle), math.sin(angle)
    tangent_x, tangent_y = -radial_y, radial_x
    for index in range(count):
        u = completion * index / (count - 1)
        radial = math.sin(math.pi * u) * length
        tangent = math.sin(math.tau * u) * width
        points.append(
            (
                cx + radial_x * radial + tangent_x * tangent,
                cy + radial_y * radial + tangent_y * tangent,
            )
        )
    return points


def draw_protected_motion_glyph(draw, cx, cy, scale, movement_phase):
    # An abstract, curled infant-motion glyph: head, protected spine and limbs.
    head_x = cx - 3.5 * scale
    head_y = cy - 6.0 * scale
    head_r = 3.0 * scale
    draw.ellipse(
        scaled([(head_x - head_r, head_y - head_r),
                (head_x + head_r, head_y + head_r)]),
        fill=0,
        outline=255,
        width=max(3, round(1.25 * SCALE * scale)),
    )

    sway = math.sin(movement_phase) * 1.3 * scale
    spine = [
        (cx - 1.0 * scale, cy - 3.0 * scale),
        (cx + 4.0 * scale + sway, cy + 1.0 * scale),
        (cx + 2.0 * scale, cy + 7.0 * scale),
        (cx - 4.0 * scale, cy + 8.0 * scale),
    ]
    draw.line(scaled(spine), fill=255, width=max(4, round(1.4 * SCALE * scale)), joint="curve")
    draw.line(
        scaled([
            (cx + 2 * scale, cy + 1 * scale),
            (cx + (7 + sway) * scale, cy - 1 * scale),
        ]),
        fill=255,
        width=max(3, round(SCALE * scale)),
    )
    draw.line(
        scaled([
            (cx + 1 * scale, cy + 6 * scale),
            (cx + 7 * scale, cy + (8 - sway) * scale),
        ]),
        fill=255,
        width=max(3, round(SCALE * scale)),
    )


def draw_motion_cradle(
    draw,
    cx,
    cy,
    scale,
    completion,
    rotation,
    frame,
    show_glyph=True,
):
    length = 24 * scale
    petal_width = 7.5 * scale
    for sensor in range(4):
        angle = rotation + sensor * math.pi / 2
        points = petal_points(cx, cy, angle, length, petal_width, completion)
        if len(points) > 1:
            draw.line(
                scaled(points),
                fill=255,
                width=max(3, round(1.3 * SCALE * scale)),
                joint="curve",
            )

        tip_x = cx + math.cos(angle) * length
        tip_y = cy + math.sin(angle) * length
        if completion > 0.88:
            draw_sensor_seed(draw, tip_x, tip_y, frame * 0.22 + sensor, scale=max(0.55, scale))

    pulse_radius = (3.8 + math.sin(frame * 0.18) * 0.8) * scale
    draw.ellipse(
        scaled([(cx - pulse_radius, cy - pulse_radius),
                (cx + pulse_radius, cy + pulse_radius)]),
        outline=255,
        width=max(3, round(SCALE * scale)),
    )

    if show_glyph and completion > 0.45:
        glyph_scale = scale * ease((completion - 0.45) / 0.55)
        draw_protected_motion_glyph(draw, cx, cy, glyph_scale, frame * 0.13)


def draw_bima_cradle_identity(canvas, frame, transition):
    draw = ImageDraw.Draw(canvas)
    amount = ease(transition)
    cx = 64 + (22 - 64) * amount
    cy = 30 + (22 - 30) * amount
    mark_scale = 1.0 + (0.43 - 1.0) * amount
    rotation = math.pi / 4
    draw_motion_cradle(
        draw,
        cx,
        cy,
        mark_scale,
        1.0,
        rotation,
        0,
        show_glyph=amount < 0.72,
    )

    if amount > 0.20:
        title_amount = ease((amount - 0.20) / 0.80)
        title_x = 128 + (39 - 128) * title_amount
        title_font = font(FONT_BOLD, 25)
        draw.text((round(title_x * SCALE), 7 * SCALE), "BIMA", font=title_font, fill=255)

    if amount > 0.64:
        subtitle_amount = ease((amount - 0.64) / 0.36)
        subtitle_font = font(FONT_UI_BOLD, 7)
        subtitle_mask = Image.new("L", canvas.size, 0)
        subtitle_draw = ImageDraw.Draw(subtitle_mask)
        subtitle_draw.text(
            (8 * SCALE, 43 * SCALE),
            "BIOMETRIC INFANT MOVEMENT",
            font=subtitle_font,
            fill=255,
        )
        subtitle_draw.text(
            (35 * SCALE, 52 * SCALE),
            "ASSESSMENT",
            font=subtitle_font,
            fill=255,
        )
        reveal_width = round(WIDTH * SCALE * subtitle_amount)
        canvas.paste(
            ImageChops.lighter(canvas, subtitle_mask),
            (0, 0),
            Image.new("L", canvas.size, 255).crop((0, 0, reveal_width, HEIGHT * SCALE)).resize(
                (reveal_width, HEIGHT * SCALE)
            ) if False else None,
        )
        if reveal_width < WIDTH * SCALE:
            ImageDraw.Draw(canvas).rectangle(
                [reveal_width, 42 * SCALE, WIDTH * SCALE, HEIGHT * SCALE],
                fill=0,
            )


def render_frame(frame: int) -> Image.Image:
    canvas = Image.new("L", (WIDTH * SCALE, HEIGHT * SCALE), 0)
    draw = ImageDraw.Draw(canvas)

    starts = [(8, 8), (120, 8), (8, 56), (120, 56)]
    rotation = math.pi / 4
    target_angles = (-3 * math.pi / 4, -math.pi / 4, 3 * math.pi / 4, math.pi / 4)
    targets = [
        (64 + math.cos(angle) * 24, 30 + math.sin(angle) * 24)
        for angle in target_angles
    ]

    if frame < 20:
        arrival = ease(frame / 19)
        for index, (start_x, start_y) in enumerate(starts):
            x = start_x + (start_x + (64 - start_x) * 0.08 - start_x) * arrival
            y = start_y + (start_y + (30 - start_y) * 0.08 - start_y) * arrival
            draw_sensor_seed(draw, x, y, frame * 0.25 + index)

        ring_radius = 2 + arrival * 10
        draw.ellipse(
            scaled([(64 - ring_radius, 30 - ring_radius),
                    (64 + ring_radius, 30 + ring_radius)]),
            outline=255,
            width=4,
        )

    elif frame < 62:
        local = frame - 20
        base_progress = local / 41
        controls = [
            ((22, 8), (38, 14)),
            ((106, 8), (90, 14)),
            ((22, 56), (38, 46)),
            ((106, 56), (90, 46)),
        ]
        for index in range(4):
            delayed = ease(max(0.0, min(1.0, base_progress * 1.18 - index * 0.06)))
            draw_signal_path(
                draw,
                starts[index],
                controls[index][0],
                controls[index][1],
                targets[index],
                delayed,
            )

        center_radius = 3 + math.sin(frame * 0.20)
        draw.ellipse(
            scaled([(64 - center_radius, 30 - center_radius),
                    (64 + center_radius, 30 + center_radius)]),
            outline=255,
            width=4,
        )

    elif frame < 94:
        local = frame - 62
        assembly = ease(local / 31)
        draw_motion_cradle(
            draw,
            64,
            30,
            1.0,
            assembly,
            rotation,
            frame,
            show_glyph=True,
        )

    elif frame < 124:
        # The visible boot clip starts here. Build the mark cleanly and evenly
        # without rotation, breathing, or wandering geometry.
        local = frame - 94
        assembly = ease(local / 29)
        draw_motion_cradle(
            draw,
            64,
            30,
            1.0,
            assembly,
            rotation,
            0,
            show_glyph=True,
        )

    elif frame < 148:
        draw_source_end_logo(canvas, (frame - 124) / 23)

    else:
        draw_source_end_logo(canvas, 1.0)

    reduced = canvas.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    # A clean threshold preserves antialiased geometry without noisy dithering.
    return reduced.point(lambda pixel: 255 if pixel >= 112 else 0, mode="1")


def frame_bytes(image: Image.Image) -> bytes:
    pixels = image.load()
    output = bytearray()
    for y in range(HEIGHT):
        for byte_x in range(WIDTH // 8):
            value = 0
            for bit in range(8):
                if pixels[byte_x * 8 + bit, y]:
                    value |= 0x80 >> bit
            output.append(value)
    return bytes(output)


def write_header(frames):
    lines = [
        "#pragma once",
        "#include <Arduino.h>",
        "",
        f"constexpr uint8_t BIMA_BOOT_FRAME_COUNT = {len(frames)};",
        f"constexpr uint16_t BIMA_BOOT_FRAME_BYTES = {WIDTH * HEIGHT // 8};",
        "const uint8_t BIMA_BOOT_FRAMES[BIMA_BOOT_FRAME_COUNT][BIMA_BOOT_FRAME_BYTES] PROGMEM = {",
    ]
    for frame in frames:
        data = frame_bytes(frame)
        lines.append("  {")
        for start in range(0, len(data), 16):
            chunk = ", ".join(f"0x{value:02X}" for value in data[start : start + 16])
            lines.append(f"    {chunk},")
        lines.append("  },")
    lines.extend(["};", ""])
    HEADER_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_preview(frames):
    selected = [
        0, 6, 12, 19, 20, 28,
        38, 48, 57, 61, 62, 70,
        82, 94, 101, 102, 108, 116,
        123, 124, 130, 138, 147, 159,
    ]
    preview = Image.new("L", (WIDTH * 6, HEIGHT * 4), 18)
    for index, frame_index in enumerate(selected):
        frame = frames[frame_index].convert("L")
        preview.paste(frame, ((index % 6) * WIDTH, (index // 6) * HEIGHT))
    preview.save(PREVIEW_PATH)
    frames[-1].convert("L").resize(
        (WIDTH * 8, HEIGHT * 8),
        Image.Resampling.NEAREST,
    ).save(END_LOGO_PREVIEW_PATH)


def main():
    frames = [render_frame(index) for index in range(FRAME_COUNT)]
    write_header(frames)
    write_preview(frames)
    print(f"Generated {FRAME_COUNT} BIMA frames")
    print(HEADER_PATH)
    print(PREVIEW_PATH)
    print(END_LOGO_PREVIEW_PATH)


if __name__ == "__main__":
    main()
