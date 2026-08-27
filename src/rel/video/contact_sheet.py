"""Contact sheets: many timestamped frames tiled into one image.

A grid of stamped frames lets the model reason about a whole episode at once and,
critically, quote a time back to us. The timestamp is burned into the pixels
because that is the only channel the model reliably reads it from.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont

from .decode import Frame

_LABEL_H = 18
_PAD = 3


def _font(size: int = 13) -> ImageFont.ImageFont:
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


@dataclass(frozen=True)
class Sheet:
    image: Image.Image
    times: list[float]

    @property
    def start(self) -> float:
        return self.times[0]

    @property
    def end(self) -> float:
        return self.times[-1]


def build_sheets(
    frames: list[Frame],
    per_sheet: int = 20,
    columns: int = 5,
) -> list[Sheet]:
    """Tile frames into sheets of `per_sheet`, `columns` wide, each tile stamped."""
    if per_sheet <= 0 or columns <= 0:
        raise ValueError("per_sheet and columns must be positive")
    if not frames:
        return []

    font = _font()
    sheets: list[Sheet] = []
    for chunk_start in range(0, len(frames), per_sheet):
        chunk = frames[chunk_start : chunk_start + per_sheet]
        tw, th = chunk[0].image.size
        cell_w, cell_h = tw + 2 * _PAD, th + _LABEL_H + 2 * _PAD
        rows = (len(chunk) + columns - 1) // columns
        sheet = Image.new("RGB", (cell_w * columns, cell_h * rows), (17, 17, 17))
        draw = ImageDraw.Draw(sheet)

        for i, frame in enumerate(chunk):
            col, row = i % columns, i // columns
            x, y = col * cell_w + _PAD, row * cell_h + _PAD
            sheet.paste(frame.image, (x, y))
            stamp = f"{frame.t:.2f}s"
            ty = y + th + 1
            # Draw on the dark gutter, not over the frame, so nothing is occluded.
            draw.text((x + 1, ty), stamp, fill=(255, 214, 10), font=font)

        sheets.append(Sheet(image=sheet, times=[f.t for f in chunk]))
    return sheets
