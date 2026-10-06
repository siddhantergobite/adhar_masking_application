from __future__ import annotations

import argparse
import io
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
DATASETS_ROOT = ROOT / "datasets"
DOCUMENT_ROOT = DATASETS_ROOT / "aadhaar-document-seg"
NUMBER_ROOT = DATASETS_ROOT / "aadhaar-number"

DOCUMENT_NAMES = {
    0: "aadhaar_front",
    1: "aadhaar_back",
    2: "aadhaar_letter",
}

_D_TABLE = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_P_TABLE = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)
_INVERSE = (0, 4, 3, 2, 1, 5, 6, 7, 8, 9)


@dataclass(frozen=True)
class DocumentTemplate:
    image: np.ndarray
    class_id: int
    number_boxes: tuple[np.ndarray, ...]


def verhoeff_check_digit(prefix: str) -> str:
    checksum = 0
    for index, character in enumerate(reversed(prefix)):
        checksum = _D_TABLE[checksum][_P_TABLE[(index + 1) % 8][int(character)]]
    return str(_INVERSE[checksum])


def synthetic_number(rng: random.Random) -> str:
    prefix = str(rng.randint(2, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(10))
    return prefix + verhoeff_check_digit(prefix)


def _font_path(*, bold: bool) -> Path:
    candidates = (
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("A usable TrueType font was not found.")


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(_font_path(bold=bold)), size=size)


def _text_box(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, text_font: ImageFont.FreeTypeFont) -> np.ndarray:
    left, top, right, bottom = draw.textbbox(xy, text, font=text_font)
    return np.array([left, top, right, bottom], dtype=np.float32)


def _draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    size: int,
    *,
    bold: bool = False,
    fill: tuple[int, int, int] = (24, 31, 28),
) -> np.ndarray:
    text_font = font(size, bold=bold)
    draw.text(xy, text, font=text_font, fill=fill)
    return _text_box(draw, xy, text, text_font)


def _draw_centered(
    draw: ImageDraw.ImageDraw,
    center_x: int,
    y: int,
    text: str,
    size: int,
    *,
    bold: bool = False,
    fill: tuple[int, int, int] = (24, 31, 28),
) -> np.ndarray:
    text_font = font(size, bold=bold)
    bounds = draw.textbbox((0, 0), text, font=text_font)
    width = bounds[2] - bounds[0]
    return _draw_text(draw, (int(center_x - width / 2), y), text, size, bold=bold, fill=fill)


def _draw_qr(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], rng: random.Random) -> None:
    left, top, right, bottom = box
    draw.rectangle(box, fill="white", outline=(40, 40, 40), width=2)
    modules = 25
    cell = max(1, min((right - left) // modules, (bottom - top) // modules))
    ox = left + ((right - left) - cell * modules) // 2
    oy = top + ((bottom - top) - cell * modules) // 2
    for row in range(modules):
        for column in range(modules):
            if rng.random() < 0.43:
                x1 = ox + column * cell
                y1 = oy + row * cell
                draw.rectangle((x1, y1, x1 + cell, y1 + cell), fill=(20, 24, 22))
    for fx, fy in ((1, 1), (modules - 8, 1), (1, modules - 8)):
        x1, y1 = ox + fx * cell, oy + fy * cell
        x2, y2 = x1 + 7 * cell, y1 + 7 * cell
        draw.rectangle((x1, y1, x2, y2), fill="white", outline=(15, 15, 15), width=max(2, cell))
        draw.rectangle(
            (x1 + 2 * cell, y1 + 2 * cell, x2 - 2 * cell, y2 - 2 * cell),
            fill=(15, 15, 15),
        )


def _draw_emblem(draw: ImageDraw.ImageDraw, x: int, y: int, scale: int = 1) -> None:
    colour = (66, 68, 62)
    draw.ellipse((x, y, x + 42 * scale, y + 32 * scale), outline=colour, width=3 * scale)
    draw.line((x + 7 * scale, y + 29 * scale, x + 35 * scale, y + 29 * scale), fill=colour, width=3 * scale)
    draw.line((x + 21 * scale, y + 5 * scale, x + 21 * scale, y + 38 * scale), fill=colour, width=3 * scale)


def _draw_aadhaar_mark(draw: ImageDraw.ImageDraw, x: int, y: int, scale: float = 1.0) -> None:
    orange = (239, 128, 28)
    red = (193, 39, 56)
    radius = int(34 * scale)
    center = (x + radius, y + radius)
    for ring in (radius, int(radius * 0.72), int(radius * 0.44)):
        draw.arc((center[0] - ring, center[1] - ring, center[0] + ring, center[1] + ring), 195, 345, fill=red, width=max(2, int(5 * scale)))
    for angle in range(195, 346, 25):
        radians = math.radians(angle)
        inner = radius + int(5 * scale)
        outer = radius + int(15 * scale)
        draw.line(
            (
                center[0] + math.cos(radians) * inner,
                center[1] + math.sin(radians) * inner,
                center[0] + math.cos(radians) * outer,
                center[1] + math.sin(radians) * outer,
            ),
            fill=orange,
            width=max(2, int(4 * scale)),
        )


def _draw_portrait(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], rng: random.Random) -> None:
    left, top, right, bottom = box
    draw.rectangle(box, fill=(220, 226, 220), outline=(118, 126, 120), width=2)
    skin = rng.choice(((169, 113, 83), (126, 82, 64), (201, 151, 111), (94, 66, 52)))
    cx = (left + right) // 2
    face_radius = max(16, (right - left) // 5)
    draw.ellipse((cx - face_radius, top + 28, cx + face_radius, top + 28 + 2 * face_radius), fill=skin)
    draw.ellipse((left + 18, top + 80, right - 18, bottom + 30), fill=rng.choice(((80, 105, 128), (105, 62, 83), (65, 94, 73))))


def _number_text(number: str, rng: random.Random) -> str:
    style = rng.choice(("groups", "groups", "groups", "compact", "hyphen"))
    if style == "compact":
        return number
    if style == "hyphen":
        return f"{number[:4]}-{number[4:8]}-{number[8:]}"
    return f"{number[:4]} {number[4:8]} {number[8:]}"


def make_front(rng: random.Random) -> DocumentTemplate:
    width, height = 900, 560
    image = Image.new("RGB", (width, height), (249, 247, 239))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((3, 3, width - 4, height - 4), radius=22, outline=(115, 119, 112), width=5)
    draw.rectangle((8, 8, width - 8, 85), fill=(252, 249, 240))
    draw.line((235, 54, 665, 54), fill=(238, 111, 27), width=13)
    draw.line((235, 70, 665, 70), fill=(42, 153, 73), width=13)
    _draw_emblem(draw, 34, 22)
    _draw_aadhaar_mark(draw, 790, 9, 0.85)
    _draw_centered(draw, 450, 18, "GOVERNMENT OF INDIA", 24, bold=True)
    _draw_text(draw, (35, 101), "UNIQUE IDENTIFICATION AUTHORITY OF INDIA", 20, bold=True, fill=(35, 116, 58))
    draw.line((22, 137, 878, 137), fill=(190, 45, 54), width=4)
    _draw_portrait(draw, (45, 164, 235, 405), rng)
    _draw_text(draw, (275, 171), "SYNTHETIC TEST HOLDER", 28, bold=True)
    _draw_text(draw, (275, 220), f"YEAR OF BIRTH: {rng.randint(1970, 2004)}", 24)
    _draw_text(draw, (275, 264), rng.choice(("MALE", "FEMALE", "OTHER")), 24)
    _draw_text(draw, (275, 319), "VID: 9000 1111 2222 3333", 19)
    _draw_qr(draw, (681, 164, 846, 329), rng)
    number = synthetic_number(rng)
    number_text = _number_text(number, rng)
    number_box = _draw_centered(draw, 520, 423, number_text, 46, bold=True)
    draw.line((22, 493, 878, 493), fill=(190, 45, 54), width=4)
    _draw_centered(draw, 450, 504, "AADHAAR — IDENTITY, NOT CITIZENSHIP", 22, bold=True)
    _draw_centered(draw, 450, 535, "SYNTHETIC TRAINING • NOT A REAL ID", 14, fill=(200, 45, 45))
    return DocumentTemplate(np.asarray(image), 0, (number_box,))


def make_back(rng: random.Random, *, include_number: bool = True) -> DocumentTemplate:
    width, height = 900, 560
    image = Image.new("RGB", (width, height), (250, 248, 240))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((3, 3, width - 4, height - 4), radius=22, outline=(115, 119, 112), width=5)
    _draw_emblem(draw, 35, 22)
    _draw_aadhaar_mark(draw, 790, 9, 0.85)
    draw.line((205, 47, 680, 47), fill=(238, 111, 27), width=12)
    draw.line((205, 64, 680, 64), fill=(42, 153, 73), width=12)
    _draw_centered(draw, 450, 91, "UNIQUE IDENTIFICATION AUTHORITY OF INDIA", 22, bold=True, fill=(35, 116, 58))
    draw.line((22, 129, 878, 129), fill=(190, 45, 54), width=4)
    _draw_text(draw, (50, 157), "ADDRESS", 27, bold=True)
    lines = (
        "SYNTHETIC TEST HOLDER",
        "42 SAMPLE ROAD, TEST DISTRICT",
        "DEMO CITY, INDIA — 400000",
        "This training card contains no real identity.",
    )
    for index, line in enumerate(lines):
        _draw_text(draw, (50, 207 + index * 42), line, 21, bold=index == 0)
    _draw_qr(draw, (620, 157, 835, 372), rng)
    number_boxes: tuple[np.ndarray, ...] = ()
    if include_number:
        number = synthetic_number(rng)
        number_box = _draw_centered(draw, 450, 418, _number_text(number, rng), 43, bold=True)
        number_boxes = (number_box,)
    else:
        _draw_centered(draw, 450, 420, "NO PRINTED AADHAAR NUMBER", 26, bold=True, fill=(98, 102, 96))
    draw.line((22, 485, 878, 485), fill=(190, 45, 54), width=4)
    _draw_centered(draw, 450, 500, "help@uidai.gov.in   •   www.uidai.gov.in", 19)
    _draw_centered(draw, 450, 532, "SYNTHETIC TRAINING • NOT A REAL ID", 14, fill=(200, 45, 45))
    return DocumentTemplate(np.asarray(image), 1, number_boxes)


def make_letter(rng: random.Random) -> DocumentTemplate:
    width, height = 900, 1280
    image = Image.new("RGB", (width, height), (253, 253, 249))
    draw = ImageDraw.Draw(image)
    draw.rectangle((4, 4, width - 5, height - 5), outline=(70, 72, 68), width=4)
    draw.line((450, 5, 450, height - 5), fill=(95, 98, 93), width=4)
    _draw_emblem(draw, 30, 24)
    _draw_aadhaar_mark(draw, 354, 12, 0.95)
    _draw_text(draw, (477, 31), "GOVERNMENT OF INDIA", 23, bold=True)
    _draw_aadhaar_mark(draw, 802, 12, 0.95)
    draw.rectangle((8, 105, 442, 168), fill=(231, 92, 16))
    _draw_centered(draw, 225, 117, "GOVERNMENT OF INDIA", 29, bold=True, fill=(255, 255, 255))
    draw.rectangle((8, 216, 442, 273), fill=(38, 151, 62))
    _draw_centered(draw, 225, 226, "UNIQUE IDENTIFICATION", 24, bold=True, fill=(255, 255, 255))
    draw.line((458, 103, 890, 103), fill=(190, 45, 54), width=5)
    _draw_centered(draw, 674, 120, "INFORMATION", 30, bold=True, fill=(181, 39, 49))
    _draw_text(draw, (488, 181), "• Aadhaar is a proof of identity.", 20, bold=True, fill=(181, 39, 49))
    info = (
        "• Verify identity using secure QR code.",
        "• Keep your information updated.",
        "• Aadhaar is unique and secure.",
        "• Use masked copies where possible.",
        "• Never share OTP or biometrics.",
        "• This is synthetic training material.",
        "• No real identity is represented here.",
        "• UIDAI layout signals are simulated.",
    )
    for index, line in enumerate(info):
        _draw_text(draw, (488, 225 + index * 46), line, 18, bold=index in {2, 5})
    _draw_text(draw, (45, 311), "TO", 18, bold=True)
    _draw_text(draw, (45, 348), "SYNTHETIC TEST HOLDER", 22, bold=True)
    _draw_text(draw, (45, 389), "42 SAMPLE ROAD", 20)
    _draw_text(draw, (45, 425), "DEMO CITY, INDIA — 400000", 20)
    _draw_qr(draw, (247, 489, 405, 647), rng)
    _draw_text(draw, (45, 503), "Enrolment: TEST/00000/00000", 16)
    number = synthetic_number(rng)
    display_number = _number_text(number, rng)
    first_box = _draw_centered(draw, 225, 690, display_number, 38, bold=True)
    _draw_centered(draw, 225, 742, "YOUR AADHAAR NUMBER", 17, bold=True, fill=(181, 39, 49))
    draw.line((9, 795, 891, 795), fill=(138, 142, 135), width=3)
    _draw_text(draw, (31, 820), "AADHAAR FRONT", 17, bold=True, fill=(35, 116, 58))
    _draw_portrait(draw, (42, 874, 155, 1040), rng)
    _draw_text(draw, (177, 883), "SYNTHETIC TEST HOLDER", 18, bold=True)
    _draw_text(draw, (177, 919), "DOB: 01/01/1990", 17)
    _draw_text(draw, (177, 953), rng.choice(("MALE", "FEMALE", "OTHER")), 17)
    _draw_text(draw, (480, 820), "AADHAAR BACK", 17, bold=True, fill=(35, 116, 58))
    _draw_text(draw, (480, 870), "ADDRESS: 42 SAMPLE ROAD", 16, bold=True)
    _draw_text(draw, (480, 904), "DEMO CITY — 400000", 16)
    _draw_qr(draw, (716, 847, 855, 986), rng)
    second_box = _draw_centered(draw, 225, 1086, display_number, 30, bold=True)
    third_box = _draw_centered(draw, 676, 1086, display_number, 30, bold=True)
    draw.line((9, 1140, 891, 1140), fill=(190, 45, 54), width=4)
    _draw_centered(draw, 450, 1160, "AADHAAR — IDENTITY, NOT CITIZENSHIP", 19, bold=True)
    _draw_centered(draw, 450, 1212, "SYNTHETIC TRAINING • NOT A REAL ID", 18, bold=True, fill=(210, 42, 42))
    return DocumentTemplate(np.asarray(image), 2, (first_box, second_box, third_box))


def make_negative_card(rng: random.Random, kind: str | None = None) -> np.ndarray:
    kind = kind or rng.choice(("debit", "credit", "pan", "voter", "licence", "visiting"))
    width, height = 900, 560
    if kind in {"debit", "credit"}:
        palette = rng.choice(((22, 34, 49), (28, 63, 76), (82, 26, 56), (39, 61, 35)))
        image = Image.new("RGB", (width, height), palette)
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((3, 3, width - 4, height - 4), radius=30, outline=(190, 198, 191), width=4)
        _draw_text(draw, (48, 42), "SAMPLE BANK", 40, bold=True, fill=(240, 244, 239))
        _draw_text(draw, (690, 48), kind.upper(), 25, bold=True, fill=(240, 244, 239))
        draw.rounded_rectangle((70, 180, 210, 285), radius=14, fill=(211, 173, 77), outline=(244, 218, 145), width=5)
        card_number = " ".join("".join(str(rng.randint(0, 9)) for _ in range(4)) for _ in range(4))
        _draw_centered(draw, 450, 330, card_number, 42, bold=True, fill=(242, 244, 239))
        _draw_text(draw, (58, 430), "TEST CUSTOMER", 29, bold=True, fill=(242, 244, 239))
        _draw_text(draw, (610, 430), "VALID 12/39", 20, fill=(242, 244, 239))
        _draw_centered(draw, 450, 515, "SYNTHETIC • NO PAYMENT VALUE", 14, fill=(245, 197, 75))
        return np.asarray(image)

    image = Image.new("RGB", (width, height), rng.choice(((232, 244, 239), (240, 235, 213), (225, 238, 249))))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((3, 3, width - 4, height - 4), radius=22, outline=(100, 112, 109), width=5)
    title = {
        "pan": "INCOME TAX • PAN TEST",
        "voter": "ELECTION ID • VOTER TEST",
        "licence": "DRIVING LICENCE TEST",
        "visiting": "VISITING CARD",
    }.get(kind, "GENERIC ID TEST")
    _draw_centered(draw, 450, 44, title, 34, bold=True)
    _draw_portrait(draw, (48, 145, 240, 412), rng)
    _draw_text(draw, (290, 162), "SYNTHETIC HOLDER", 29, bold=True)
    _draw_text(draw, (290, 225), "NOT AN AADHAAR DOCUMENT", 23, bold=True, fill=(177, 42, 50))
    fake_id = "ABCDE1234F" if kind == "pan" else "XYZ9876543"
    _draw_text(draw, (290, 298), fake_id, 48, bold=True)
    _draw_text(draw, (290, 380), "TEST DATA ONLY", 25)
    _draw_centered(draw, 450, 508, "SYNTHETIC TRAINING • NOT A REAL ID", 16, fill=(200, 45, 45))
    return np.asarray(image)


def make_background(rng: random.Random, size: int) -> np.ndarray:
    base = np.array(rng.choice(((222, 218, 207), (188, 198, 196), (213, 205, 191), (203, 211, 218), (174, 177, 169))), dtype=np.float32)
    noise = np.random.default_rng(rng.randrange(2**32)).normal(0, rng.uniform(5, 18), (size, size, 1))
    image = np.clip(base.reshape(1, 1, 3) + noise, 0, 255).astype(np.uint8)
    colour = tuple(int(value) for value in np.clip(base * rng.uniform(0.55, 0.9), 0, 255))
    spacing = rng.randint(28, 75)
    for offset in range(-size, size * 2, spacing):
        cv2.line(image, (offset, 0), (offset - size, size), colour, rng.choice((1, 1, 2)), cv2.LINE_AA)
    if rng.random() < 0.55:
        for _ in range(rng.randint(4, 12)):
            center = (rng.randrange(size), rng.randrange(size))
            radius = rng.randint(15, 80)
            cv2.circle(image, center, radius, tuple(int(c) for c in rng.choice(((205, 177, 46), (89, 103, 98), (151, 101, 75)))), -1, cv2.LINE_AA)
    return image


def _quad_for_template(
    rng: random.Random,
    template: np.ndarray,
    canvas_size: int,
    *,
    large: bool,
) -> np.ndarray:
    height, width = template.shape[:2]
    aspect = width / height
    max_side = rng.uniform(canvas_size * (0.58 if large else 0.34), canvas_size * (0.88 if large else 0.53))
    if aspect >= 1:
        target_width = max_side
        target_height = target_width / aspect
    else:
        target_height = max_side
        target_width = target_height * aspect
    angle = rng.uniform(-180, 180)
    margin = canvas_size * 0.08
    center = np.array(
        [rng.uniform(margin + target_width * 0.33, canvas_size - margin - target_width * 0.33),
         rng.uniform(margin + target_height * 0.33, canvas_size - margin - target_height * 0.33)],
        dtype=np.float32,
    )
    local = np.array(
        [
            [-target_width / 2, -target_height / 2],
            [target_width / 2, -target_height / 2],
            [target_width / 2, target_height / 2],
            [-target_width / 2, target_height / 2],
        ],
        dtype=np.float32,
    )
    radians = math.radians(angle)
    rotation = np.array([[math.cos(radians), -math.sin(radians)], [math.sin(radians), math.cos(radians)]], dtype=np.float32)
    quad = local @ rotation.T + center
    jitter = min(target_width, target_height) * rng.uniform(0.015, 0.095)
    quad += np.random.default_rng(rng.randrange(2**32)).uniform(-jitter, jitter, quad.shape).astype(np.float32)
    quad[:, 0] = np.clip(quad[:, 0], 6, canvas_size - 7)
    quad[:, 1] = np.clip(quad[:, 1], 6, canvas_size - 7)
    return quad


def paste_perspective(canvas: np.ndarray, template: np.ndarray, quad: np.ndarray, rng: random.Random) -> np.ndarray:
    canvas_height, canvas_width = canvas.shape[:2]
    template_height, template_width = template.shape[:2]
    source = np.array(
        [[0, 0], [template_width - 1, 0], [template_width - 1, template_height - 1], [0, template_height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, quad.astype(np.float32))
    warped = cv2.warpPerspective(template, matrix, (canvas_width, canvas_height), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT)
    mask_source = np.full((template_height, template_width), 255, dtype=np.uint8)
    mask = cv2.warpPerspective(mask_source, matrix, (canvas_width, canvas_height), flags=cv2.INTER_LINEAR)
    shadow = cv2.GaussianBlur(mask, (0, 0), rng.uniform(5, 14))
    shadow_matrix = np.float32([[1, 0, rng.uniform(5, 15)], [0, 1, rng.uniform(6, 18)]])
    shadow = cv2.warpAffine(shadow, shadow_matrix, (canvas_width, canvas_height), borderValue=0)
    shadow_alpha = (shadow.astype(np.float32) / 255.0 * rng.uniform(0.12, 0.30))[..., None]
    canvas = np.clip(canvas.astype(np.float32) * (1.0 - shadow_alpha), 0, 255).astype(np.uint8)
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    return np.clip(warped.astype(np.float32) * alpha + canvas.astype(np.float32) * (1.0 - alpha), 0, 255).astype(np.uint8)


def photometric_augment(image: np.ndarray, rng: random.Random, *, strong: bool) -> np.ndarray:
    result = image.astype(np.float32)
    contrast = rng.uniform(0.72 if strong else 0.88, 1.22 if strong else 1.12)
    brightness = rng.uniform(-25 if strong else -12, 22 if strong else 12)
    result = np.clip((result - 127.5) * contrast + 127.5 + brightness, 0, 255)
    if rng.random() < (0.70 if strong else 0.42):
        noise = np.random.default_rng(rng.randrange(2**32)).normal(0, rng.uniform(1.5, 9 if strong else 4), result.shape)
        result = np.clip(result + noise, 0, 255)
    result = result.astype(np.uint8)
    if rng.random() < (0.48 if strong else 0.22):
        sigma = rng.uniform(0.35, 1.8 if strong else 0.9)
        result = cv2.GaussianBlur(result, (0, 0), sigma)
    if strong and rng.random() < 0.25:
        kernel_size = rng.choice((3, 5, 7, 9))
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        if rng.random() < 0.5:
            kernel[kernel_size // 2, :] = 1.0 / kernel_size
        else:
            kernel[:, kernel_size // 2] = 1.0 / kernel_size
        result = cv2.filter2D(result, -1, kernel)
    if rng.random() < (0.42 if strong else 0.18):
        overlay = result.copy()
        height, width = result.shape[:2]
        center = (rng.randrange(width), rng.randrange(height))
        axes = (rng.randint(max(20, width // 10), max(30, width // 2)), rng.randint(max(15, height // 20), max(25, height // 5)))
        cv2.ellipse(overlay, center, axes, rng.uniform(0, 180), 0, 360, (255, 255, 245), -1, cv2.LINE_AA)
        result = cv2.addWeighted(overlay, rng.uniform(0.08, 0.28), result, rng.uniform(0.72, 0.92), 0)
    quality = rng.randint(38 if strong else 62, 96)
    success, encoded = cv2.imencode(".jpg", cv2.cvtColor(result, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if success:
        result = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return result


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path, format="JPEG", quality=92, optimize=True)


def _write_document_label(path: Path, labels: list[tuple[int, np.ndarray]], size: int) -> None:
    rows: list[str] = []
    for class_id, polygon in labels:
        coordinates = polygon.astype(np.float64).copy()
        coordinates[:, 0] /= size
        coordinates[:, 1] /= size
        values = " ".join(f"{value:.6f}" for value in coordinates.reshape(-1))
        rows.append(f"{class_id} {values}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def _write_number_label(path: Path, boxes: tuple[np.ndarray, ...], width: int, height: int) -> None:
    rows: list[str] = []
    for box in boxes:
        left, top, right, bottom = [float(value) for value in box]
        center_x = ((left + right) / 2) / width
        center_y = ((top + bottom) / 2) / height
        box_width = (right - left) / width
        box_height = (bottom - top) / height
        rows.append(f"0 {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def _make_template(rng: random.Random, class_id: int | None = None) -> DocumentTemplate:
    class_id = rng.choice(tuple(DOCUMENT_NAMES)) if class_id is None else class_id
    if class_id == 0:
        return make_front(rng)
    if class_id == 1:
        return make_back(rng, include_number=rng.random() < 0.72)
    return make_letter(rng)


def generate_document_example(rng: random.Random, size: int, *, force_negative: bool = False) -> tuple[np.ndarray, list[tuple[int, np.ndarray]]]:
    canvas = make_background(rng, size)
    labels: list[tuple[int, np.ndarray]] = []
    negative = force_negative or rng.random() < 0.28
    negative_count = rng.randint(1, 3)
    for _ in range(negative_count):
        card = make_negative_card(rng)
        quad = _quad_for_template(rng, card, size, large=False)
        canvas = paste_perspective(canvas, card, quad, rng)
    if not negative:
        aadhaar_count = 2 if rng.random() < 0.12 else 1
        for _ in range(aadhaar_count):
            template = _make_template(rng)
            quad = _quad_for_template(rng, template.image, size, large=aadhaar_count == 1)
            canvas = paste_perspective(canvas, template.image, quad, rng)
            labels.append((template.class_id, quad))
    return photometric_augment(canvas, rng, strong=True), labels


def generate_number_example(rng: random.Random, *, force_negative: bool = False) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    negative = force_negative or rng.random() < 0.24
    if negative:
        image = make_negative_card(rng)
        return photometric_augment(image, rng, strong=False), ()
    template = _make_template(rng)
    return photometric_augment(template.image, rng, strong=False), template.number_boxes


def _prepare_root(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def generate(args: argparse.Namespace) -> None:
    _prepare_root(DOCUMENT_ROOT)
    _prepare_root(NUMBER_ROOT)
    split_counts = {"train": args.train, "val": args.val, "test": args.test}
    summary: dict[str, dict[str, dict[str, int]]] = {"document": {}, "number": {}}

    for split_index, (split, count) in enumerate(split_counts.items()):
        document_counts = {"images": count, "negatives": 0, "aadhaar_instances": 0}
        number_counts = {"images": count, "negatives": 0, "number_instances": 0}
        for index in range(count):
            doc_rng = random.Random(args.seed + split_index * 1_000_000 + index)
            force_negative = index < max(2, int(count * 0.25))
            doc_image, doc_labels = generate_document_example(doc_rng, args.scene_size, force_negative=force_negative)
            stem = f"synthetic_{split}_{index:05d}"
            _write_image(DOCUMENT_ROOT / "images" / split / f"{stem}.jpg", doc_image)
            _write_document_label(DOCUMENT_ROOT / "labels" / split / f"{stem}.txt", doc_labels, args.scene_size)
            if not doc_labels:
                document_counts["negatives"] += 1
            document_counts["aadhaar_instances"] += len(doc_labels)

            number_rng = random.Random(args.seed + 500_000 + split_index * 1_000_000 + index)
            number_image, number_boxes = generate_number_example(number_rng, force_negative=force_negative)
            _write_image(NUMBER_ROOT / "images" / split / f"{stem}.jpg", number_image)
            height, width = number_image.shape[:2]
            _write_number_label(NUMBER_ROOT / "labels" / split / f"{stem}.txt", number_boxes, width, height)
            if not number_boxes:
                number_counts["negatives"] += 1
            number_counts["number_instances"] += len(number_boxes)

        summary["document"][split] = document_counts
        summary["number"][split] = number_counts

    manifest = {
        "generator": "privacy-safe procedural Aadhaar bootstrap",
        "seed": args.seed,
        "scene_size": args.scene_size,
        "contains_real_personal_data": False,
        "document_classes": DOCUMENT_NAMES,
        "summary": summary,
        "warning": "Synthetic bootstrap data is not a substitute for an independent, consented production validation set.",
    }
    (DATASETS_ROOT / "synthetic-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate privacy-safe YOLO bootstrap data.")
    parser.add_argument("--train", type=int, default=240)
    parser.add_argument("--val", type=int, default=48)
    parser.add_argument("--test", type=int, default=48)
    parser.add_argument("--scene-size", type=int, default=768)
    parser.add_argument("--seed", type=int, default=20261005)
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
