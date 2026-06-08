import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "yolo11m.pt"

# BGR color mapping for each defect/object class
CLASS_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "crack":          (0, 0, 255),       # red
    "pothole":        (0, 0, 255),       # red
    "spalling":       (0, 165, 255),     # orange
    "patch":          (0, 255, 255),     # yellow
    "fod":            (128, 0, 128),     # purple
    "marking_damage": (0, 255, 255),     # yellow
}
DEFAULT_COLOR_BGR: tuple[int, int, int] = (255, 0, 0)  # blue — all other classes

_model: YOLO | None = None


def _get_model() -> YOLO:
    global _model
    if _model is None:
        # YOLO auto-downloads yolo11m.pt on first call if not present locally
        _model = YOLO(MODEL_PATH)
    return _model


def _box_color(class_name: str) -> tuple[int, int, int]:
    return CLASS_COLORS_BGR.get(class_name.lower(), DEFAULT_COLOR_BGR)


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


def run_detection(image_path: str) -> tuple[list[dict], np.ndarray]:
    """Run YOLOv11 inference, annotate the image, and return detections + annotated array."""
    model = _get_model()
    img = cv2.imread(image_path)

    results = model(img, verbose=False)[0]
    detections: list[dict] = []
    annotated = img.copy()

    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name: str = model.names[cls_id]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        color = _box_color(class_name)

        # Draw filled label background then border box
        label = f"{class_name} {confidence:.0%}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated, label, (x1 + 3, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )

        detections.append({
            "class": class_name,
            "confidence": round(confidence * 100, 1),
            "bbox": [x1, y1, x2, y2],
        })

    return detections, annotated
