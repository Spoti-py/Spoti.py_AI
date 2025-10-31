from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import StreamingResponse, JSONResponse
from module.streaming import generate, push_frame
from pymongo import MongoClient
from dotenv import load_dotenv
import os

app = FastAPI()
load_dotenv()
MONGODB_URL = os.getenv("MONGODB_URL")
client = MongoClient(MONGODB_URL)
db = client['spotipy']
col = db['sleepy']

@app.get("/")
async def root():
    return "ok"

@app.get("/stream")
async def stream():
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/upload")
async def upload(request: Request, file: UploadFile | None = File(None)):
    if file is not None:
        image_bytes = await file.read()
    else:
        image_bytes = await request.body()
    if not image_bytes:
        return JSONResponse({"ok": False, "error": "empty body"}, status_code=400)
    push_frame(image_bytes)
    return {"ok": True}

@app.post("/statistic")
async def statistic():
    doc = list(col.find().sort("timestamp", -1).limit(50))
    for d in doc:
        d["_id"] = str(d["_id"])
    return {"data": doc}