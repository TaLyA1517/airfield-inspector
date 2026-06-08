from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

MODEL_PATH = "yolo11m.pt"

# RGB color mapping for each defect/object class
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


def _get_model() -> YOLO:
    global _model
    if _model is None:
        # YOLO auto-downloads yolo11m.pt on first call if not present locally
        _model = YOLO(MODEL_PATH)
    return _model


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
    """Return overall pavement condition based on detected classes."""
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
    """Run YOLOv11 inference, annotate the image, and return detections + annotated PIL Image."""
    model = _get_model()
    img = Image.open(image_path).convert("RGB")

    # Pass the file path so Ultralytics uses its own loader
    results = model(image_path, verbose=False)[0]
    draw = ImageDraw.Draw(img)
    font = _get_font(14)
    detections: list[dict] = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name: str = model.names[cls_id]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        color = _box_color(class_name)
        label = f"{class_name} {confidence:.0%}"

        # Bounding box outline
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)

        # Filled label chip above the box
        lb = draw.textbbox((0, 0), label, font=font)
        tw, th = lb[2] - lb[0], lb[3] - lb[1]
        ly = max(0, y1 - th - 6)
        draw.rectangle([(x1, ly), (x1 + tw + 6, ly + th + 4)], fill=color)
        draw.text((x1 + 3, ly + 2), label, fill=(255, 255, 255), font=font)

        detections.append({
            "class": class_name,
            "confidence": round(confidence * 100, 1),
            "bbox": [x1, y1, x2, y2],
        })

    return detections, img
