FROM python:3.11-slim
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 fonts-dejavu-core --no-install-recommends && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p uploads results
RUN python3 -c "import urllib.request; urllib.request.urlretrieve('https://huggingface.co/OpenSistemas/YOLOv8-crack-seg/resolve/main/yolov8m/weights/best.pt', 'best.pt'); print('Model downloaded')"
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
