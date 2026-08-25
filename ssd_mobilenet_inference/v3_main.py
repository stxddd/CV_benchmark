import cv2
import numpy as np
import onnxruntime as ort

MODEL = "ssd_mobilenet_v3.onnx"
LABELS = "coco.names"

INPUT_SIZE = 320
CONF_THRESHOLD = 0.5


with open(LABELS, "r") as f:
    classes = [x.strip() for x in f.readlines()]

session = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])


input_name = session.get_inputs()[0].name


print("INPUT:")
print(session.get_inputs()[0].shape)


print("\nOUTPUTS:")

for o in session.get_outputs():
    print(o.name, o.shape)


cap = cv2.VideoCapture(0)


if not cap.isOpened():
    exit("Camera error")


prev = cv2.getTickCount()


while True:

    ret, frame = cap.read()

    if not ret:
        break

    h, w = frame.shape[:2]

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))

    img = img.astype(np.float32)

    img /= 255.0

    img = np.transpose(img, (2, 0, 1))

    img = np.expand_dims(img, axis=0)

    outputs = session.run(None, {input_name: img})

    boxes = outputs[0]
    scores = outputs[1]
    labels = outputs[2]

    if boxes.ndim == 3:
        boxes = boxes[0]

    if scores.ndim == 2:
        scores = scores[0]

    if labels.ndim == 2:
        labels = labels[0]

    for i in range(len(scores)):

        conf = float(scores[i])

        if conf < CONF_THRESHOLD:
            continue

        cls = int(labels[i]) - 1

        box = boxes[i]

        scale_x = w / INPUT_SIZE
        scale_y = h / INPUT_SIZE

        x1 = int(box[0] * scale_x)

        y1 = int(box[1] * scale_y)

        x2 = int(box[2] * scale_x)

        y2 = int(box[3] * scale_y)

        x1 = max(0, min(x1, w))

        y1 = max(0, min(y1, h))

        x2 = max(0, min(x2, w))

        y2 = max(0, min(y2, h))

        if cls < len(classes):

            name = classes[cls]

        else:

            name = str(cls)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            frame,
            f"{name} {conf:.2f}",
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    now = cv2.getTickCount()

    fps = cv2.getTickFrequency() / (now - prev)

    prev = now

    cv2.putText(
        frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )

    cv2.imshow("SSD MobileNetV3 Lite ONNX", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
