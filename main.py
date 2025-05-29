#E:\Minicinda\conda\ObjectDetection\OBJECTDETECTION
from ultralytics import YOLO
import cv2
import numpy as np
import time

PIXEL_TO_METER = 0.05
MOVEMENT_THRESHOLD = 2  

def calculate_distance(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return np.sqrt(dx**2 + dy**2)

def main():
    model = YOLO('yolov8n')
    cap = cv2.VideoCapture('videoplayback (8).mp4')

    positions = {}
    speeds = {}
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (500, 320))
        now = time.time()
        dt = now - prev_time
        prev_time = now

        results = model.track(frame, persist=True, conf=0.5, iou=0.5)
        tracked = results[0]
        out = tracked.plot()

        moving = 0
        stationary = 0

        if tracked.boxes.id is not None:
            for box, obj_id in zip(tracked.boxes.xywh, tracked.boxes.id):
                obj_id = int(obj_id)
                cx, cy, w, h = box
                curr = (cx, cy)

                if obj_id in positions:
                    dist_px = calculate_distance(positions[obj_id], curr)
                    if dist_px > MOVEMENT_THRESHOLD:
                        moving += 1
                    else:
                        stationary += 1

                    # compute speed
                    if dt > 0:
                        speed = dist_px * PIXEL_TO_METER / dt
                        speeds[obj_id] = speed
                        x, y = int(cx - w/2), int(cy - h/2 - 10)
                        cv2.putText(out, f"ID{obj_id} {speed:.2f}m/s",
                                    (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                positions[obj_id] = curr

        cv2.putText(out, f"Moving: {moving}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        cv2.putText(out, f"Stationary: {stationary}", (20,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        cv2.imshow('frame', out)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
