import cv2, time
from ultralytics import YOLO
model = YOLO("./models/faceDetecter.pt")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("딴카메라")
prev = time.time()
fps = 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    results = model(frame, imgsz=640, conf=0.8)[0]
    annotated = results.plot()
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