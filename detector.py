import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# Primary weights: downloaded from HuggingFace on first run
CUSTOM_WEIGHTS_PATH = "best.pt"
CUSTOM_WEIGHTS_URL = (
    "https://huggingface.co/OpenSistemas/YOLOv8-crack-seg"
    "/resolve/main/yolov8m/weights/best.pt"
)
# Used when the HuggingFace download fails or the file cannot be loaded
FALLBACK_MODEL = "yolo11m.pt"

# Russian display names shown in the UI and on bounding-box labels
CLASS_DISPLAY_NAMES: dict[str, str] = {
    "crack":   "Трещина",
    "pothole": "Выбоина",
}

# RGB box colours keyed by English class name
CLASS_COLORS_RGB: dict[str, tuple[int, int, int]] = {
    "crack":          (255, 0, 0),     # red
    "pothole":        (255, 0, 0),     # red
    "spalling":       (255, 165, 0),   # orange
    "patch":          (255, 255, 0),   # yellow
    "fod":            (128, 0, 128),   # purple
    "marking_damage": (255, 255, 0),   # yellow
}
DEFAULT_COLOR_RGB: tuple[int, int, int] = (0, 0, 255)  # blue — all other classes

_model: YOLO | None = None


def _download_weights() -> bool:
    """Download CUSTOM_WEIGHTS_URL to CUSTOM_WEIGHTS_PATH. Returns True on success."""
    dest = Path(CUSTOM_WEIGHTS_PATH)
    try:
        print(f"Downloading weights from {CUSTOM_WEIGHTS_URL} …")
        urllib.request.urlretrieve(CUSTOM_WEIGHTS_URL, str(dest))
        return True
    except Exception as exc:
        print(f"Weight download failed: {exc}")
        dest.unlink(missing_ok=True)  # remove any partial file
        return False


def _load_model() -> YOLO:
    """Load custom weights, downloading them first if absent; fall back to FALLBACK_MODEL."""
    if not Path(CUSTOM_WEIGHTS_PATH).exists():
        _download_weights()

    if Path(CUSTOM_WEIGHTS_PATH).exists():
        try:
            return YOLO(CUSTOM_WEIGHTS_PATH)
        except Exception as exc:
            print(f"Failed to load {CUSTOM_WEIGHTS_PATH}: {exc}")
            Path(CUSTOM_WEIGHTS_PATH).unlink(missing_ok=True)

    print(f"Falling back to {FALLBACK_MODEL}")
    return YOLO(FALLBACK_MODEL)


def _get_model() -> YOLO:
    global _model
    if _model is None:
        _model = _load_model()
    return _model


def _box_color(class_name: str) -> tuple[int, int, int]:
    return CLASS_COLORS_RGB.get(class_name.lower(), DEFAULT_COLOR_RGB)


def _get_font(size: int = 14) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Return a PIL font, trying common system TTF paths before the built-in default."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
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
    """Return overall pavement condition based on detected classes."""
    fod_count = sum(1 for d in detections if "fod" in d["class"].lower())
    crack_pothole_count = sum(
        1 for d in detections if d["class"].lower() in {"crack", "pothole"}
    )
    if crack_pothole_count == 0 and fod_count == 0:
        return "NORM"
    if crack_pothole_count > 3:
        return "CRITICAL"
    if fod_count > 0:
        return "WARNING"
    return "WARNING"


def run_detection(image_path: str) -> tuple[list[dict], Image.Image]:
    """Run local YOLO inference, draw annotated boxes, return detections + PIL Image.

    Each detection dict contains:
        class       — English class name (used for CSS colour classes and condition logic)
        display     — Localised display name (Russian where mapped)
        confidence  — float 0-100
        bbox        — [x1, y1, x2, y2] in pixel coordinates
    """
    model = _get_model()
    img = Image.open(image_path).convert("RGB")

    results = model(image_path, conf=0.15, verbose=False)[0]
    draw = ImageDraw.Draw(img)
    font = _get_font(28)
    detections: list[dict] = []
    _counter = [0]  # detection counter

    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name: str = model.names[cls_id]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        color = _box_color(class_name)
        display_name = CLASS_DISPLAY_NAMES.get(class_name.lower(), class_name)
        _counter[0] += 1; label = f"#{_counter[0]} {display_name} {confidence:.0%}"

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
