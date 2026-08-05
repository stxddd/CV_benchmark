import cv2
import time
import numpy as np
import onnxruntime as ort

# ==================== НАСТРОЙКИ ====================
MODEL_PATH = "ssd_mobilenet_v2.onnx"
LABELS_PATH = "coco.names"

CONF_THRESHOLD = 0.55          # ← подними порог (0.5–0.65)
NMS_THRESHOLD  = 0.45          # ← Non-Maximum Suppression

INPUT_SIZE = (300, 300)
SOURCE = 0
# ===================================================

with open(LABELS_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

cap = cv2.VideoCapture(SOURCE)
prev_time = time.time()
fps = 0.0
frame_count = 0

print("[INFO] V2 запущен. Нажми 'q' для выхода")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # Подготовка
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, INPUT_SIZE)
    blob = np.expand_dims(img, axis=0).astype(np.uint8)

    outputs = session.run(None, {input_name: blob})

    boxes   = outputs[1][0]      # (100, 4)
    classes = outputs[2][0]      # (100,)
    scores  = outputs[4][0]      # (100,)
    num_det = int(outputs[5][0])

    # Собираем детекции
    det_boxes = []
    det_scores = []
    det_classes = []

    for i in range(num_det):
        score = float(scores[i])
        if score < CONF_THRESHOLD:
            continue

        ymin, xmin, ymax, xmax = boxes[i]
        x1 = int(xmin * w)
        y1 = int(ymin * h)
        x2 = int(xmax * w)
        y2 = int(ymax * h)

        det_boxes.append([x1, y1, x2 - x1, y2 - y1])  # x, y, w, h
        det_scores.append(score)
        det_classes.append(int(classes[i]))

    # ---------- Non-Maximum Suppression ----------
    indices = cv2.dnn.NMSBoxes(det_boxes, det_scores, CONF_THRESHOLD, NMS_THRESHOLD)

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, bw, bh = det_boxes[i]
            score = det_scores[i]
            cls_id = det_classes[i]

            label = class_names[cls_id - 1] if 1 <= cls_id <= len(class_names) else str(cls_id)

            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {score:.2f}", (x, y - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # FPS
    frame_count += 1
    now = time.time()
    if now - prev_time >= 1.0:
        fps = frame_count / (now - prev_time)
        prev_time = now
        frame_count = 0

    cv2.putText(frame, f"FPS: {fps:.1f} | V2", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("SSD MobileNet V2", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()