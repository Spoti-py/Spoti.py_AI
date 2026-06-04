from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
from module.streaming import generate, process_keypoints
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
async def upload(request: Request):
    try:
        payload = await request.json()
        result = process_keypoints(payload)
    except ValueError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

    return {"ok": True, **result}

@app.websocket("/ws/keypoints")
async def keypoints_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            try:
                payload = await websocket.receive_json()
                result = process_keypoints(payload)
                await websocket.send_json({"ok": True, **result})
            except ValueError as e:
                await websocket.send_json({"ok": False, "error": str(e)})
    except WebSocketDisconnect:
        pass

@app.post("/statistic")
async def statistic():
    doc = list(col.find().sort("timestamp", -1).limit(50))
    for d in doc:
        d["_id"] = str(d["_id"])
    return {"data": doc}
