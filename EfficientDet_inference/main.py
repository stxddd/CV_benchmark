import os
import urllib.request
import cv2
import numpy as np
import onnxruntime as ort
import time

MODEL_URL = "https://github.com/namas191297/efficientdetlite/raw/main/onnx_models/efficientdet_lite1.onnx"
MODEL_PATH = "efficientdet_lite0.onnx"
INPUT_SIZE = 320
SCORE_THRESH = 0.30
CAMERA_ID = 0


LABELS = [
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


def download():
    if not os.path.exists(MODEL_PATH):
        print("Скачиваю модель...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Готово")
    return MODEL_PATH


def preprocess(frame, size):

    resized = cv2.resize(frame, (size, size))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return np.expand_dims(rgb.astype(np.uint8), axis=0)


def postprocess(outputs, orig_h, orig_w, input_size, score_thresh):

    boxes = outputs[0][0]
    class_ids = outputs[1][0]
    scores = outputs[2][0]

    bboxes = boxes * input_size
    bboxes = np.clip(bboxes, 0, input_size)

    scale_w = orig_w / input_size
    scale_h = orig_h / input_size

    results = []
    for i in range(len(scores)):
        if scores[i] < score_thresh:
            continue
        cls = int(class_ids[i])
        if not (0 <= cls < len(LABELS)):
            continue

        ymin, xmin, ymax, xmax = bboxes[i]
        left = int(xmin * scale_w)
        top = int(ymin * scale_h)
        right = int(xmax * scale_w)
        bottom = int(ymax * scale_h)

        results.append(
            {
                "box": (left, top, right, bottom),
                "label": LABELS[cls],
                "score": float(scores[i]),
                "cls": cls,
            }
        )
    return results


def main():
    download()
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        raise RuntimeError("Камера не открылась")

    print("Запуск. q=выход  s=сохранить  +/-=порог")
    print("В консоль пишутся сырые class_id — по ним видно, что реально отдаёт модель")

    threshold = SCORE_THRESH
    prev = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]

        inp = preprocess(frame, INPUT_SIZE)
        outs = session.run(None, {input_name: inp})
        dets = postprocess(outs, h, w, INPUT_SIZE, threshold)

        for d in dets:
            x1, y1, x2, y2 = d["box"]
            text = f"{d['label']} {d['score']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                text,
                (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

        now = time.time()
        fps = 0.9 * fps + 0.1 / (now - prev + 1e-6)
        prev = now
        cv2.putText(
            frame,
            f"FPS {fps:.1f}  thr={threshold:.2f}  n={len(dets)}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        cv2.imshow("EfficientDet-Lite0", frame)

        if int(now * 2) % 2 == 0 and dets:
            top = sorted(dets, key=lambda x: -x["score"])[:5]
            print("top:", [(d["cls"], d["label"], f"{d['score']:.2f}") for d in top])

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            fn = f"cap_{int(time.time())}.jpg"
            cv2.imwrite(fn, frame)
            print("saved", fn)
        elif key in (ord("+"), ord("=")):
            threshold = min(0.9, threshold + 0.05)
        elif key == ord("-"):
            threshold = max(0.1, threshold - 0.05)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
