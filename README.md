# CV Benchmark: ONNX на Orange Pi 4B

Набор минимальных инференсов для ONNX-моделей COCO. 

## Конфигурация теста

- Процессор: AMD Ryzen 7 5700X.
- CPU: 8 ядер / 16 потоков.
- Видео: `test.mp4`, 400 кадров.
- Execution provider: `CPUExecutionProvider`.


## Модели

- EfficientDet-Lite0: `EfficientDet_inference/efficientdet_lite0.onnx`.
- NanoDet-Plus 1.5x: `NanoDet-Plus_inference/nanodet-plus-m-1.5x_320.onnx`.
- PicoDet-S: `PicoDet-s_inference/picodet_s_320_coco.onnx`.
- SSD MobileNet V2: `ssd_mobilenet_inference/ssd_mobilenet_v2.onnx`.
- YOLO: файлы из `yolo_inference/models/`.

## Запуск инференса на видео

Все команды выполняются из корня проекта после активации окружения:

```bash
source .venv/bin/activate
```

Одна модель:

```bash
python benchmark_onnx_models.py --models yolo8n --video test.mp4 --threads 4 --warmup 20
```

Все модели:

```bash
python benchmark_onnx_models.py --video test.mp4 --threads 4 --warmup 20 --output-dir results/all --output-csv results/all.csv
```

Доступные имена: `efficientdet`, `nanodet`, `picodet`, `ssd_v2`, `leyolo_nano`, `yolo5n`, `yolo8n`, `yolo10n`, `yolo11n`, `yolo26n`.

Видео открывается заново для каждой модели. По умолчанию обрабатываются первые 400 кадров; изменить лимит можно через `--max-frames`, а `--max-frames 0` означает всё видео. 

## Вывод

| Модель | Кадры | FPS min | FPS avg | FPS max | Latency avg, мс | Latency min, мс | Latency max, мс |
|---|---:|---:|---:|---:|---:|---:|---:|
| EfficientDet-Lite0 | 400 | 25.96 | 37.33 | 40.69 | 26.88 | 24.58 | 38.51 |
| NanoDet-Plus-M 1.5x | 400 | 23.86 | 40.38 | 44.82 | 24.95 | 22.31 | 41.92 |
| PicoDet-S | 400 | 30.15 | 42.10 | 54.44 | 23.97 | 18.37 | 33.16 |
| SSD MobileNet V2 | 400 | 29.92 | 38.34 | 46.30 | 26.38 | 21.60 | 33.42 |
| LeYOLO Nano | 400 | 60.75 | 92.80 | 104.96 | 10.89 | 9.53 | 16.46 |
| YOLO11n | 400 | 34.45 | 72.20 | 85.23 | 14.07 | 11.73 | 29.03 |
| YOLO26n | 400 | 33.41 | 74.64 | 100.20 | 13.67 | 9.98 | 29.93 |
| YOLOv10n | 400 | 40.08 | 75.59 | 96.30 | 13.63 | 10.38 | 24.95 |
| YOLOv5n-u | 400 | 40.88 | 73.34 | 86.34 | 13.83 | 11.58 | 24.46 |
| YOLOv8n | 400 | 24.11 | 69.56 | 81.33 | 14.69 | 12.30 | 41.47 |

*Результаты получены на AMD Ryzen 7 5700X, 8 ядер / 16 потоков. Для каждой модели обработано 400 кадров.*

## Результаты на Orange Pi 4B

Для каждой модели обработано 200 кадров видео.

| Модель | Кадры | FPS min | FPS avg | FPS max | Latency avg, мс | Latency min, мс | Latency max, мс |
|---|---:|---:|---:|---:|---:|---:|---:|
| EfficientDet-Lite0 | 200 | 2.37 | 2.65 | 2.73 | 377.34 | 366.61 | 421.18 |
| NanoDet-Plus-M 1.5x | 200 | 3.93 | 5.10 | 5.23 | 196.19 | 191.24 | 254.43 |
| PicoDet-S | 200 | 7.22 | 8.08 | 8.24 | 123.75 | 121.40 | 138.59 |
| SSD MobileNet V2 | 200 | 4.19 | 4.33 | 4.38 | 230.71 | 228.16 | 238.65 |
| LeYOLO Nano | 200 | 5.83 | 5.99 | 6.07 | 166.87 | 164.63 | 171.45 |
| YOLO11n | 200 | 5.56 | 6.08 | 6.23 | 164.55 | 160.57 | 179.76 |
| YOLO26n | 200 | 6.72 | 7.39 | 7.48 | 135.27 | 133.68 | 148.72 |
| YOLOv10n | 200 | 6.26 | 6.45 | 6.53 | 155.04 | 153.22 | 159.85 |
| YOLOv5n-u | 200 | 5.69 | 5.94 | 6.09 | 168.49 | 164.31 | 175.79 |
| YOLOv8n | 200 | 5.35 | 5.46 | 5.54 | 183.27 | 180.37 | 186.98 |

*Результаты получены на Orange Pi 4B. Для каждой модели обработано 200 кадров.*