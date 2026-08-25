import os
import time
import cv2
import numpy as np
import onnxruntime as ort

MODEL_DIR = "models"


MODELS = {
    "yolo5n": "yolov5n.onnx",
    "yolo8n": "yolov8n.onnx",
    "yolo10n": "yolov10n.onnx",
    "yolo11n": "yolo11n.onnx",
    "yolo26n": "yolo26n.onnx",
    "LeYOLONano": "LeYOLONano.onnx",
}

INPUT_SIZE = 320

CAMERA_ID = 0

CONF_THRESHOLD = 0.35

NMS_THRESHOLD = 0.45


COCO = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def select_model():

    models = [
        ("YOLOv5n", "yolov5nu.onnx"),
        ("YOLOv8n", "yolov8n.onnx"),
        ("YOLOv10n", "yolov10n.onnx"),
        ("YOLOv11n", "yolo11n.onnx"),
        ("YOLO26n", "yolo26n.onnx"),
        ("LeYOLONano", "LeYOLONano.onnx"),
    ]

    for i, (name, file) in enumerate(models, 1):

        path = os.path.join(MODEL_DIR, file)

        status = "OK" if os.path.exists(path) else "missing"

        print(f"{i}. {name:10} [{status}]")

    print()

    while True:

        choice = input("Enter model number: ").strip()

        if not choice.isdigit():

            print("Enter number")

            continue

        index = int(choice) - 1

        if index < 0 or index >= len(models):

            print("Wrong number")

            continue

        name, file = models[index]

        path = os.path.join(MODEL_DIR, file)

        if not os.path.exists(path):

            print("Model not found:", path)

            continue

        print(f"\nSelected: {name}")

        print("Path:", path)

        return path


def load_model(path):

    options = ort.SessionOptions()

    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1

    session = ort.InferenceSession(
        path, sess_options=options, providers=["CPUExecutionProvider"]
    )

    print("\nINPUT:")
    print(session.get_inputs()[0].shape)

    print("OUTPUT:")
    print(session.get_outputs()[0].shape)

    return session


def letterbox(img, size):

    h, w = img.shape[:2]

    scale = min(size / w, size / h)

    nw = int(w * scale)
    nh = int(h * scale)

    img = cv2.resize(img, (nw, nh))

    canvas = np.full((size, size, 3), 114, dtype=np.uint8)

    dx = (size - nw) // 2
    dy = (size - nh) // 2

    canvas[dy : dy + nh, dx : dx + nw] = img

    return canvas, scale, dx, dy


def preprocess(frame):

    img, scale, dx, dy = letterbox(frame, INPUT_SIZE)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32) / 255.0

    img = np.transpose(img, (2, 0, 1))

    img = np.expand_dims(img, 0)

    return np.ascontiguousarray(img), scale, dx, dy


def nms(boxes, scores):

    ids = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, NMS_THRESHOLD)

    if len(ids) == 0:
        return []

    return ids.flatten()


def postprocess(output, frame_shape, scale, dx, dy):

    h, w = frame_shape[:2]

    output = np.squeeze(output)

    detections = []

    if output.shape[-1] == 6:

        for det in output:

            x1, y1, x2, y2, conf, cls = det

            if conf < CONF_THRESHOLD:
                continue

            cls = int(cls)

            if cls >= 80:
                continue

            x1 = (x1 - dx) / scale
            y1 = (y1 - dy) / scale
            x2 = (x2 - dx) / scale
            y2 = (y2 - dy) / scale

            detections.append(
                ([int(x1), int(y1), int(x2 - x1), int(y2 - y1)], float(conf), cls)
            )

        return detections

    if output.shape[0] in [84, 85]:

        output = output.T

    for det in output:

        x, y, bw, bh = det[:4]

        if len(det) == 85:

            obj = det[4]

            if obj < CONF_THRESHOLD:
                continue

            scores = det[5:] * obj

        else:

            scores = det[4:]

        cls = int(np.argmax(scores))

        conf = float(scores[cls])

        if conf < CONF_THRESHOLD:
            continue

        x1 = x - bw / 2
        y1 = y - bh / 2
        x2 = x + bw / 2
        y2 = y + bh / 2

        x1 = (x1 - dx) / scale
        y1 = (y1 - dy) / scale
        x2 = (x2 - dx) / scale
        y2 = (y2 - dy) / scale

        detections.append(([int(x1), int(y1), int(x2 - x1), int(y2 - y1)], conf, cls))

    if not detections:
        return []

    boxes = [d[0] for d in detections]

    scores = [d[1] for d in detections]

    keep = nms(boxes, scores)

    return [detections[i] for i in keep]


def main():

    model_path = select_model()

    session = load_model(model_path)

    input_name = session.get_inputs()[0].name

    cap = cv2.VideoCapture(CAMERA_ID)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    last = time.time()

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        inp, scale, dx, dy = preprocess(frame)

        output = session.run(None, {input_name: inp})

        detections = postprocess(output[0], frame.shape, scale, dx, dy)

        for box, conf, cls in detections:

            x, y, w, h = box

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(
                frame,
                f"{COCO[cls]} {conf:.2f}",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        fps = 1 / (time.time() - last)

        last = time.time()

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

        cv2.imshow("YOLO ONNX", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
