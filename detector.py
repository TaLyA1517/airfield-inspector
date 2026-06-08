import base64
import os
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

# Roboflow hosted-inference endpoint.
# Full model reference: road-damage-detection-n2xkq/crack-and-pothole-bftyl/1
ROBOFLOW_API_BASE = "https://detect.roboflow.com"
ROBOFLOW_MODEL = "crack-and-pothole-bftyl/1"

# Russian display names shown in the UI and on bounding-box labels
CLASS_DISPLAY_NAMES: dict[str, str] = {
    "crack":   "Трещина",
    "pothole": "Выбоина",
}

# RGB box colours keyed by the English class name returned by the API
CLASS_COLORS_RGB: dict[str, tuple[int, int, int]] = {
    "crack":          (255, 0, 0),     # red
    "pothole":        (255, 0, 0),     # red
    "spalling":       (255, 165, 0),   # orange
    "patch":          (255, 255, 0),   # yellow
    "fod":            (128, 0, 128),   # purple
    "marking_damage": (255, 255, 0),   # yellow
}
DEFAULT_COLOR_RGB: tuple[int, int, int] = (0, 0, 255)  # blue — all other classes


def _get_api_key() -> str:
    key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not key:
        raise RuntimeError("ROBOFLOW_API_KEY environment variable is not set")
    return key


def _box_color(class_name: str) -> tuple[int, int, int]:
    return CLASS_COLORS_RGB.get(class_name.lower(), DEFAULT_COLOR_RGB)


def _get_font(size: int = 14) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Return a PIL font, trying common system TTF paths before the built-in default."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    # Pillow >= 10 accepts a size parameter on load_default
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def get_condition(detections: list[dict]) -> str:
    """Return overall pavement condition based on English class names from the API."""
    fod_count = sum(1 for d in detections if "fod" in d["class"].lower())
    crack_pothole_count = sum(
        1 for d in detections if d["class"].lower() in {"crack", "pothole"}
    )
    if crack_pothole_count > 3:
        return "CRITICAL"
    if fod_count > 0:
        return "WARNING"
    return "NORM"


def run_detection(image_path: str) -> tuple[list[dict], Image.Image]:
    """Call Roboflow Inference API, draw annotated boxes, return detections + PIL Image.

    Each detection dict contains:
        class       — English class name (used for CSS colour classes and condition logic)
        display     — Localised display name (Russian where mapped)
        confidence  — float 0-100
        bbox        — [x1, y1, x2, y2] in pixel coordinates
    """
    api_key = _get_api_key()
    img = Image.open(image_path).convert("RGB")

    # Roboflow hosted inference accepts raw base64-encoded image bytes
    with open(image_path, "rb") as fh:
        img_b64 = base64.b64encode(fh.read()).decode("utf-8")

    resp = requests.post(
        f"{ROBOFLOW_API_BASE}/{ROBOFLOW_MODEL}",
        params={"api_key": api_key},
        data=img_b64,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    resp.raise_for_status()
    predictions = resp.json().get("predictions", [])

    draw = ImageDraw.Draw(img)
    font = _get_font(14)
    detections: list[dict] = []

    for pred in predictions:
        class_name: str = pred["class"]
        confidence = float(pred["confidence"])

        # Roboflow returns centre-point + dimensions; convert to corner coords
        cx, cy = float(pred["x"]), float(pred["y"])
        w, h = float(pred["width"]), float(pred["height"])
        x1, y1 = int(cx - w / 2), int(cy - h / 2)
        x2, y2 = int(cx + w / 2), int(cy + h / 2)

        color = _box_color(class_name)
        display_name = CLASS_DISPLAY_NAMES.get(class_name.lower(), class_name)
        label = f"{display_name} {confidence:.0%}"

        # Bounding box outline
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)

        # Filled label chip above the box
        lb = draw.textbbox((0, 0), label, font=font)
        tw, th = lb[2] - lb[0], lb[3] - lb[1]
        ly = max(0, y1 - th - 6)
        draw.rectangle([(x1, ly), (x1 + tw + 6, ly + th + 4)], fill=color)
        draw.text((x1 + 3, ly + 2), label, fill=(255, 255, 255), font=font)

        detections.append({
            "class":      class_name,
            "display":    display_name,
            "confidence": round(confidence * 100, 1),
            "bbox":       [x1, y1, x2, y2],
        })

    return detections, img
