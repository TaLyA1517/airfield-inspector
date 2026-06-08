# ИСППР-АП — Airfield Pavement Inspector

Web application for automated detection of defects and foreign objects (FOD) on airfield pavement using **YOLOv11**.

## Features

- Drag & drop image upload (JPG / PNG)
- YOLOv11 inference with colour-coded bounding boxes per defect class
- Results page: annotated image, detection table, confidence bars, condition badge
- PDF report export (annotated image + detection table + timestamp)
- History page: last 10 analysed images with thumbnails and status badges
- Clean dark UI, Russian language interface, mobile-friendly

## Condition logic

| Badge | Condition |
|---|---|
| 🟢 НОРМА | No detections, or no critical classes found |
| 🟡 ПРЕДУПРЕЖДЕНИЕ | At least one **FOD** object detected |
| 🔴 КРИТИЧНО | More than 3 **crack / pothole** detections |

## Bounding box colours

| Class | Colour |
|---|---|
| `crack`, `pothole` | Red |
| `spalling` | Orange |
| `patch`, `marking_damage` | Yellow |
| `fod` | Purple |
| All other classes | Blue |

> **Note:** the bundled weights are COCO-pretrained (`yolo11m.pt`).  
> Replace with custom airfield-defect weights by swapping `MODEL_PATH` in `detector.py`.

---

## Quick start

### 1. Clone and set up environment

```bash
git clone <repo-url>
cd airfield-inspector
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

Ultralytics pulls in `opencv-python` (the GUI build) as a transitive dependency,
which requires `libGL` and fails in headless/server environments (Codespaces, Docker, CI).
The fix is to install `opencv-python-headless` first, then install ultralytics without
letting pip override it with the GUI build:

```bash
pip install opencv-python-headless==4.10.0.84
pip install -r requirements.txt --no-deps ultralytics==8.3.0
pip install fastapi==0.115.0 "uvicorn[standard]==0.30.6" jinja2==3.1.4 \
    python-multipart==0.0.12 Pillow==10.4.0 reportlab==4.2.2 aiofiles==24.1.0
```

**Simpler one-liner** (works when `opencv-python` is not yet installed):

```bash
pip install -r requirements.txt
# If the above pulled in opencv-python instead of the headless build:
pip uninstall -y opencv-python && pip install opencv-python-headless==4.10.0.84
```

> On first run the `yolo11m.pt` weights (~40 MB) are downloaded automatically.

### 3. (Optional) Cyrillic fonts for PDF reports

ReportLab uses system TTF fonts for Russian text.  
Install DejaVu fonts if not already present:

```bash
# Debian / Ubuntu
sudo apt-get install -y fonts-dejavu-core

# RHEL / Fedora
sudo dnf install -y dejavu-sans-fonts
```

### 4. Run the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000** in your browser.

---

## Project structure

```
airfield-inspector/
├── main.py          # FastAPI app — routes, upload logic, PDF generation
├── detector.py      # YOLOv11 inference, colour coding, condition logic
├── requirements.txt
├── templates/
│   ├── index.html   # Upload page
│   ├── results.html # Detection results
│   └── history.html # Analysis history
├── static/
│   └── style.css    # Dark theme
├── uploads/         # Original uploaded images (created automatically)
└── results/         # Annotated images + JSON metadata (created automatically)
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `fastapi` | Web framework |
| `uvicorn` | ASGI server |
| `jinja2` | HTML templates |
| `python-multipart` | File upload parsing |
| `ultralytics` | YOLOv11 inference |
| `opencv-python-headless` | Image I/O and drawing |
| `Pillow` | Image utilities |
| `reportlab` | PDF generation |
| `aiofiles` | Async file writes |
