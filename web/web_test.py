import os
import cv2
import time
import threading
import numpy as np
from collections import deque
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, StreamingResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from frameprocesser import FrameProcessor
import mediapipe as mp

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

processor = FrameProcessor()
processor.start()

@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>YOLO + EAR Live</title>
<style>
body { margin:0; background:#0b0b0b; color:#fafafa; font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif;}
.wrap { display:flex; min-height:100vh; align-items:center; justify-content:center; flex-direction:column; gap:16px; }
.card { background:#111; border:1px solid #222; border-radius:16px; padding:16px; box-shadow:0 10px 30px rgba(0,0,0,.35);}
img { max-width:min(96vw, 960px); border-radius:12px; display:block; }
.badge { font-size:12px; opacity:.8; }
</style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <div class="badge">ultralytics + mediapipe + fastapi • mjpeg</div>
    <img src="/video_feed" alt="live stream">
  </div>
</div>
</body>
</html>
    """

@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return "ok"

def mjpeg_generator():
    boundary = b"--frame"
    while True:
        with processor.lock:
            frame = processor.latest_jpeg
        if frame is None:
            # 아직 프레임 없으면 살짝 대기
            time.sleep(0.01)
            continue
        yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.on_event("shutdown")
def on_shutdown():
    processor.stop()
