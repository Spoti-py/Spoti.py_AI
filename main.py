from fastapi import FastAPI, File, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse, JSONResponse
from module.streaming import generate, process_keypoints
from module.yolo_detector import decode_base64_image, detect_yawn_bytes, detect_yawn_frame
# from pymongo import MongoClient
# from dotenv import load_dotenv
# import os

app = FastAPI()
# load_dotenv()
# MONGODB_URL = os.getenv("MONGODB_URL")
# client = MongoClient(MONGODB_URL, serverSelectionTimeoutMS=1000)
# db = client['spotipy']
# col = db['sleepy']

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

@app.post("/detect/yawn")
async def detect_yawn(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        result = await run_in_threadpool(detect_yawn_bytes, image_bytes)
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
            except Exception as e:
                print(f"웹소켓 처리 오류남\n{e}")
                await websocket.send_json({"ok": False, "error": "internal server error"})
    except WebSocketDisconnect:
        pass

@app.websocket("/ws/yawn")
async def yawn_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            try:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if "bytes" in message and message["bytes"] is not None:
                    image_bytes = message["bytes"]
                    result = await run_in_threadpool(detect_yawn_bytes, image_bytes)
                elif "text" in message and message["text"] is not None:
                    frame = decode_base64_image(message["text"])
                    result = await run_in_threadpool(detect_yawn_frame, frame)
                else:
                    raise ValueError("empty frame")

                await websocket.send_json({"ok": True, **result})
            except ValueError as e:
                await websocket.send_json({"ok": False, "error": str(e)})
            except Exception as e:
                print(f"하품 웹소켓 처리 오류남\n{e}")
                await websocket.send_json({"ok": False, "error": "internal server error"})
    except WebSocketDisconnect:
        pass

# @app.post("/statistic")
# async def statistic():
#     doc = list(col.find().sort("timestamp", -1).limit(50))
#     for d in doc:
#         d["_id"] = str(d["_id"])
#     return {"data": doc}
