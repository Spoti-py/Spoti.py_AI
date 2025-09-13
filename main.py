import cv2, time, numpy as np
from ultralytics import YOLO

model = YOLO("./models/faceDetecter.pt")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("딴카메라")

prev = time.time()
fps = 0.0
crop_id = 0 

def clamp_box(x1, y1, x2, y2, W, H):
    x1 = max(0, min(int(x1), W-1))
    y1 = max(0, min(int(y1), H-1))
    x2 = max(0, min(int(x2), W-1))
    y2 = max(0, min(int(y2), H-1))
    return x1, y1, x2, y2

while True:
    ok, frame = cap.read()
    if not ok:
        break
    H, W = frame.shape[:2]
    results = model(frame, imgsz=256, conf=0.7)[0]
    annotated = results.plot()
    if results.boxes is not None and len(results.boxes) > 0:
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()

        for i, (b, cf) in enumerate(zip(boxes, confs)):
            x1, y1, x2, y2 = clamp_box(*b, W, H)
            if x2 <= x1 or y2 <= y1:
                continue
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            face_small = cv2.resize(face, (160, 160), interpolation=cv2.INTER_LINEAR)
            cv2.imshow(f"face{i}", face_small)

            # cv2.imwrite(f"face_{crop_id:06d}.jpg", face)
            # crop_id += 1 파일로저장

        # areas = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
        # j = int(np.argmax(areas))
        # x1, y1, x2, y2 = clamp_box(*boxes[j], W, H)
        # main_face = frame[y1:y2, x1:x2]
        # cv2.imshow("main_face", cv2.resize(main_face, (160,160))) 가장 큰 얼굻만

    # FPS 표시
    now = time.time()
    fps = 0.9*fps + 0.1*(1.0/(now - prev))
    prev = now
    cv2.putText(annotated, f"FPS: {fps:.1f}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("YOLO Realtime", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()