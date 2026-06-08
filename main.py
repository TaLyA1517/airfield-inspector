import json
import os
import uuid
from datetime import datetime
from pathlib import Path

import aiofiles
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from detector import get_condition, run_detection

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
HISTORY_FILE = Path("history.json")
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_HISTORY = 10

UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# App & static mounts
# ---------------------------------------------------------------------------

app = FastAPI(title="ИСППР-АП — Airfield Inspector")

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/results_img", StaticFiles(directory="results"), name="results_img")

templates = Jinja2Templates(directory="templates")

# ---------------------------------------------------------------------------
# History helpers
# ---------------------------------------------------------------------------


def _load_history() -> list[dict]:
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, IOError):
            return []
    return []


def _save_history(history: list[dict]) -> None:
    HISTORY_FILE.write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload(request: Request, file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "Допустимые форматы: JPG, PNG"},
            status_code=400,
        )

    image_id = str(uuid.uuid4())
    original_path = UPLOAD_DIR / f"{image_id}{ext}"
    result_img_path = RESULTS_DIR / f"{image_id}{ext}"

    async with aiofiles.open(original_path, "wb") as f:
        await f.write(await file.read())

    try:
        detections, annotated_img = run_detection(str(original_path))
    except Exception as exc:
        original_path.unlink(missing_ok=True)
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": f"Ошибка анализа: {exc}"},
            status_code=500,
        )

    annotated_img.save(str(result_img_path))

    condition = get_condition(detections)
    timestamp = datetime.now().isoformat()

    result_data = {
        "id": image_id,
        "original_filename": file.filename,
        "ext": ext,
        "detections": detections,
        "condition": condition,
        "timestamp": timestamp,
    }

    (RESULTS_DIR / f"{image_id}.json").write_text(
        json.dumps(result_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    history = _load_history()
    history.insert(0, {
        "id": image_id,
        "filename": file.filename,
        "ext": ext,
        "condition": condition,
        "timestamp": timestamp,
    })
    _save_history(history[:MAX_HISTORY])

    return RedirectResponse(url=f"/results/{image_id}", status_code=303)


@app.get("/results/{image_id}", response_class=HTMLResponse)
async def results(request: Request, image_id: str):
    result_json = RESULTS_DIR / f"{image_id}.json"
    if not result_json.exists():
        return RedirectResponse(url="/")

    data = json.loads(result_json.read_text(encoding="utf-8"))
    ts = datetime.fromisoformat(data["timestamp"]).strftime("%d.%m.%Y %H:%M:%S")

    class_counts: dict[str, int] = {}
    total_conf = 0.0
    for det in data["detections"]:
        cls = det["class"]
        class_counts[cls] = class_counts.get(cls, 0) + 1
        total_conf += det["confidence"]

    avg_confidence = (
        round(total_conf / len(data["detections"]), 1) if data["detections"] else 0
    )

    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "data": data,
            "timestamp": ts,
            "class_counts": class_counts,
            "avg_confidence": avg_confidence,
        },
    )


@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request):
    hist = _load_history()
    formatted = []
    for item in hist:
        ts = datetime.fromisoformat(item["timestamp"]).strftime("%d.%m.%Y %H:%M")
        formatted.append({**item, "ts_display": ts})
    return templates.TemplateResponse(
        "history.html", {"request": request, "history": formatted}
    )


@app.get("/report/{image_id}")
async def report(image_id: str):
    result_json = RESULTS_DIR / f"{image_id}.json"
    if not result_json.exists():
        return RedirectResponse(url="/")

    data = json.loads(result_json.read_text(encoding="utf-8"))
    pdf_path = _generate_pdf(data)
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"airfield_report_{image_id[:8]}.pdf",
    )


# ---------------------------------------------------------------------------
# PDF generation (ReportLab)
# ---------------------------------------------------------------------------


def _find_cyrillic_font() -> str | None:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]
    return next((p for p in candidates if os.path.exists(p)), None)


def _generate_pdf(data: dict) -> str:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )

    font_name = "Helvetica"
    font_path = _find_cyrillic_font()
    if font_path:
        try:
            pdfmetrics.registerFont(TTFont("CyrFont", font_path))
            font_name = "CyrFont"
        except Exception:
            pass

    pdf_path = str(RESULTS_DIR / f"{data['id']}_report.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "RPTitle", parent=styles["Title"],
        fontName=font_name, fontSize=17, spaceAfter=8,
    )
    h2_style = ParagraphStyle(
        "RPH2", parent=styles["Heading2"],
        fontName=font_name, fontSize=13, spaceAfter=6, spaceBefore=12,
    )
    normal_style = ParagraphStyle(
        "RPNormal", parent=styles["Normal"],
        fontName=font_name, fontSize=11, spaceAfter=5,
    )

    condition_labels = {
        "NORM":     "НОРМА",
        "WARNING":  "ПРЕДУПРЕЖДЕНИЕ",
        "CRITICAL": "КРИТИЧНО",
    }
    ts = datetime.fromisoformat(data["timestamp"]).strftime("%d.%m.%Y %H:%M:%S")
    condition_label = condition_labels.get(data["condition"], data["condition"])

    elements: list = []
    elements.append(Paragraph("ИСППР-АП — Отчёт об инспекции покрытия", title_style))
    elements.append(Spacer(1, 0.3 * cm))
    elements.append(Paragraph(f"Дата анализа: {ts}", normal_style))
    elements.append(Paragraph(f"Файл: {data['original_filename']}", normal_style))
    elements.append(Paragraph(f"Состояние покрытия: {condition_label}", normal_style))
    elements.append(Spacer(1, 0.6 * cm))

    result_img_path = RESULTS_DIR / f"{data['id']}{data['ext']}"
    if result_img_path.exists():
        elements.append(Paragraph("Аннотированное изображение", h2_style))
        elements.append(Image(str(result_img_path), width=16 * cm, height=10 * cm))
        elements.append(Spacer(1, 0.6 * cm))

    elements.append(Paragraph("Таблица обнаружений", h2_style))
    if data["detections"]:
        header_row = ["Класс", "Уверенность", "x1", "y1", "x2", "y2"]
        rows = [header_row]
        for det in data["detections"]:
            b = det["bbox"]
            rows.append([det["class"], f"{det['confidence']}%",
                         str(b[0]), str(b[1]), str(b[2]), str(b[3])])

        col_w = [4.5 * cm, 3.2 * cm, 1.8 * cm, 1.8 * cm, 1.8 * cm, 1.8 * cm]
        table = Table(rows, colWidths=col_w)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a365d")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, -1), font_name),
            ("FONTSIZE",   (0, 0), (-1, -1), 9),
            ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
            ("GRID",       (0, 0), (-1, -1), 0.4, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.Color(0.93, 0.95, 0.98)]),
        ]))
        elements.append(table)
    else:
        elements.append(Paragraph("Дефекты не обнаружены.", normal_style))

    doc.build(elements)
    return pdf_path
