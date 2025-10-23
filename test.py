import cv2
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH,640)
capture.set(cv2.CAP_PROP_FRAME_WIDTH,480)

while cv2.waitKey(33) < 0 :
    # 33ms마다 반복문을 실행
    ret, frame = capture.read()
    from ultralytics import YOLO
    model = YOLO("./models/bestM.pt")
    res = model(frame, conf = 0.306, imgsz = 640)
    frame = res[0].plot()
    cv2.imshow("VideoFrame", frame)
    print(cv2.waitKey(33))

capture.release()
cv2.destroyAllWindows()