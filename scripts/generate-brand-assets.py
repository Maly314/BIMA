"""Generate BIMA application assets from the untouched master logo.

The master artwork is never rewritten. The full logo is copied to ``public``
for the desktop splash, while the infant-and-wave emblem is cropped into the
square icon sizes used by Electron, Windows shortcuts, browsers, and PWAs.
"""

from pathlib import Path
import shutil

from PIL import Image, ImageChops


ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "assets" / "branding" / "bima-logo-master.png"
PUBLIC = ROOT / "public"


def content_bounds(image: Image.Image) -> tuple[int, int, int, int]:
    """Return the non-white artwork bounds, ignoring faint JPEG-like noise."""
    rgb = image.convert("RGB")
    white = Image.new("RGB", rgb.size, "white")
    difference = ImageChops.difference(rgb, white).convert("L")
    mask = difference.point(lambda value: 255 if value > 12 else 0)
    return mask.getbbox() or (0, 0, image.width, image.height)


def make_emblem(master: Image.Image, size: int) -> Image.Image:
    # The supplied artwork places the emblem in the upper 65 percent and the
    # BIMA wordmark below it. Keep only that exact emblem for small app icons.
    upper = master.crop((0, 0, master.width, round(master.height * 0.65)))
    left, top, right, bottom = content_bounds(upper)
    emblem = upper.crop((left, top, right, bottom)).convert("RGB")

    padding = max(1, round(size * 0.055))
    available = size - padding * 2
    scale = min(available / emblem.width, available / emblem.height)
    fitted = emblem.resize(
        (max(1, round(emblem.width * scale)), max(1, round(emblem.height * scale))),
        Image.Resampling.LANCZOS,
    )
    icon = Image.new("RGB", (size, size), "white")
    icon.paste(fitted, ((size - fitted.width) // 2, (size - fitted.height) // 2))
    return icon


def main() -> None:
    if not MASTER.exists():
        raise FileNotFoundError(f"Missing master logo: {MASTER}")

    PUBLIC.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(MASTER, PUBLIC / "bima-logo.png")

    master = Image.open(MASTER).convert("RGB")
    outputs = {
        "bima-icon-1024.png": 1024,
        "bima-icon-512.png": 512,
        "bima-icon-192.png": 192,
        "apple-touch-icon.png": 180,
    }
    rendered: dict[int, Image.Image] = {}
    for filename, size in outputs.items():
        rendered[size] = make_emblem(master, size)
        rendered[size].save(PUBLIC / filename, "PNG", optimize=True)

    icon_sizes = (16, 24, 32, 48, 64, 128, 256)
    ico_base = make_emblem(master, 256)
    ico_base.save(
        PUBLIC / "bima-desktop.ico",
        format="ICO",
        sizes=[(size, size) for size in icon_sizes],
    )
    shutil.copyfile(PUBLIC / "bima-desktop.ico", PUBLIC / "favicon.ico")

    print(f"Master: {MASTER}")
    print(f"Full logo: {PUBLIC / 'bima-logo.png'}")
    print(f"Desktop icon: {PUBLIC / 'bima-desktop.ico'}")


if __name__ == "__main__":
    main()
