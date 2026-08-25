from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import onnx
    from onnx import numpy_helper, shape_inference
except ImportError as exc:
    raise SystemExit(
        "This script requires the 'onnx' package. Install it with: pip install onnx"
    ) from exc


@dataclass
class ModelRow:
    name: str
    path: Path
    params_m: float | None
    gmacs: float | None
    size_mb: float
    map5095: float | None
    error: str | None = None


DISPLAY_NAMES = {
    "efficientdet_lite0": "EfficientDet-Lite0",
    "efficientdet_lite1": "EfficientDet-Lite1",
    "nanodet-plus-m": "NanoDet-Plus-M",
    "nanodet-plus-m_320": "NanoDet-Plus-M",
    "nanodet-plus-m-1.5x_320": "NanoDet-Plus-M 1.5x",
    "picodet_s_320_coco": "PicoDet-S",
    "ssd_mobilenet_v2": "SSD MobileNet V2",
    "ssd_mobilenet_v3": "SSD MobileNet V3",
    "ssdlite_mobilenet_v3_large": "SSDLite MobileNet V3 Large",
    "LeYOLONano": "LeYOLONano",
    "yolov5nu": "YOLOv5n-u",
    "yolov8n": "YOLOv8n",
    "yolov10n": "YOLOv10n",
    "yolo11n": "YOLO11n",
    "yolo26n": "YOLO26n",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan all ONNX models in the workspace and print a summary table."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Root directory to search for .onnx files.",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        help=(
            "Optional JSON file with mAP50-95 values. Keys may be filenames or relative paths."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Optional CSV output path.",
    )
    return parser.parse_args()


def make_display_name(path: Path, root: Path) -> str:
    stem = path.stem
    if stem in DISPLAY_NAMES:
        return DISPLAY_NAMES[stem]
    relative = path.relative_to(root).with_suffix("").as_posix()
    cleaned = re.sub(r"[_/]+", " ", relative)
    return cleaned.replace("models ", "")


def load_metrics(metrics_path: Path | None) -> dict[str, float]:
    if not metrics_path:
        return {}
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("metrics-json must contain a JSON object")

    metrics: dict[str, float] = {}
    for key, value in data.items():
        if value is None or value == "":
            continue
        metrics[str(key).replace("\\", "/")] = float(value)
    return metrics


def count_parameters(model: Any) -> int:
    total = 0
    for initializer in model.graph.initializer:
        total += math.prod(initializer.dims)
    return total


def tensor_shapes(model: Any) -> dict[str, list[int | None]]:
    inferred = shape_inference.infer_shapes(model)
    shapes: dict[str, list[int | None]] = {}

    def record(value_info: Any) -> None:
        if not value_info.type.HasField("tensor_type"):
            return
        dims: list[int | None] = []
        for dim in value_info.type.tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                dims.append(int(dim.dim_value))
            else:
                dims.append(None)
        shapes[value_info.name] = dims

    for item in inferred.graph.input:
        record(item)
    for item in inferred.graph.value_info:
        record(item)
    for item in inferred.graph.output:
        record(item)
    for initializer in inferred.graph.initializer:
        shapes[initializer.name] = [int(dim) for dim in initializer.dims]

    return shapes


def element_count(shape: list[int | None] | None) -> int | None:
    if not shape:
        return None
    total = 1
    for dim in shape:
        if dim is None or dim <= 0:
            return None
        total *= dim
    return total


def conv_macs(node: Any, shapes: dict[str, list[int | None]]) -> float | None:
    if len(node.input) < 2 or len(node.output) < 1:
        return None
    output_shape = shapes.get(node.output[0])
    weight_shape = shapes.get(node.input[1])
    if not output_shape or not weight_shape:
        return None

    out_elems = element_count(output_shape)
    if out_elems is None or len(weight_shape) < 3:
        return None

    kernel_elems = math.prod(weight_shape[2:])
    in_channels_per_group = weight_shape[1]
    return float(out_elems * in_channels_per_group * kernel_elems)


def gemm_macs(node: Any, shapes: dict[str, list[int | None]]) -> float | None:
    if len(node.input) < 2 or len(node.output) < 1:
        return None
    lhs = shapes.get(node.input[0])
    rhs = shapes.get(node.input[1])
    out = shapes.get(node.output[0])
    if not lhs or not rhs or not out:
        return None

    lhs_elems = element_count(lhs)
    rhs_elems = element_count(rhs)
    out_elems = element_count(out)
    if lhs_elems is None or rhs_elems is None or out_elems is None:
        return None

    if len(lhs) >= 2 and len(rhs) >= 2:
        m = lhs[-2]
        k = lhs[-1]
        n = rhs[-1]
        if m is None or k is None or n is None:
            return None
        return float(m * n * k)

    return None


def estimate_gmacs(model: Any) -> float | None:
    try:
        shapes = tensor_shapes(model)
    except Exception:
        return None

    total_macs = 0.0
    saw_any = False

    for node in model.graph.node:
        op = node.op_type
        macs: float | None = None
        if op in {"Conv", "ConvTranspose"}:
            macs = conv_macs(node, shapes)
        elif op in {"Gemm", "MatMul"}:
            macs = gemm_macs(node, shapes)

        if macs is not None:
            total_macs += macs
            saw_any = True

    if not saw_any:
        return None
    return total_macs / 1_000_000_000.0


def lookup_metric(metrics: dict[str, float], root: Path, path: Path) -> float | None:
    rel = path.relative_to(root).as_posix()
    candidates = [rel, path.name]
    for candidate in candidates:
        if candidate in metrics:
            return metrics[candidate]
    for key, value in metrics.items():
        if Path(key).name == path.name:
            return value
    return None


def metric_to_percent(value: float | None) -> float | None:
    if value is None:
        return None
    return value * 100.0 if value <= 1.5 else value


def collect_rows(root: Path, metrics: dict[str, float]) -> list[ModelRow]:
    rows: list[ModelRow] = []
    for path in sorted(root.rglob("*.onnx")):
        if "venv" in path.parts:
            continue

        size_mb = path.stat().st_size / (1024 * 1024)
        map5095 = lookup_metric(metrics, root, path)

        try:
            model = onnx.load(str(path), load_external_data=False)
            params = count_parameters(model)
            gmacs = estimate_gmacs(model)
            params_m: float | None = params / 1_000_000.0
            error: str | None = None
        except Exception as exc:
            params_m = None
            gmacs = None
            error = f"{type(exc).__name__}: {exc}"

        rows.append(
            ModelRow(
                name=path.stem,
                path=path,
                params_m=params_m,
                gmacs=gmacs,
                size_mb=size_mb,
                map5095=map5095,
                error=error,
            )
        )
    return rows


def fmt_number(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def print_markdown(rows: list[ModelRow]) -> None:
    print("| Модель | Параметры, млн | КОПТс | Размер, МБ | mAP50-95, % |")
    print("|---|---:|---:|---:|---:|")
    for row in rows:
        map_percent = metric_to_percent(row.map5095)
        print(
            f"| {row.path.parent.name}/{row.path.name} | {fmt_number(row.params_m)} | "
            f"{fmt_number(row.gmacs)} | {fmt_number(row.size_mb)} | {fmt_number(map_percent)} |"
        )

    failures = [row for row in rows if row.error]
    if failures:
        print("\nПроблемные файлы:")
        for row in failures:
            print(f"- {row.path}: {row.error}")


def print_pretty_table(rows: list[ModelRow], root: Path) -> None:
    headers = [
        "Модель",
        "Параметры, млн",
        "КОПТс",
        "Размер, МБ",
        "mAP50-95, %",
        "Статус",
    ]

    display_rows: list[list[str]] = []
    for row in rows:
        map_percent = metric_to_percent(row.map5095)
        display_rows.append(
            [
                make_display_name(row.path, root),
                fmt_number(row.params_m),
                fmt_number(row.gmacs),
                fmt_number(row.size_mb),
                fmt_number(map_percent),
                "OK" if not row.error else "ERR",
            ]
        )

    widths = [len(header) for header in headers]
    for row in display_rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def line(left: str, fill: str, sep: str, right: str) -> str:
        parts = [fill * (width + 2) for width in widths]
        return left + sep.join(parts) + right

    def render_row(row: list[str]) -> str:
        cells = [f" {cell.ljust(widths[index])} " for index, cell in enumerate(row)]
        return "|" + "|".join(cells) + "|"

    print(line("+", "-", "+", "+"))
    print(render_row(headers))
    print(line("+", "=", "+", "+"))
    for row in display_rows:
        print(render_row(row))
    print(line("+", "-", "+", "+"))

    failures = [row for row in rows if row.error]
    if failures:
        print("\nПроблемные файлы:")
        for row in failures:
            print(f"- {make_display_name(row.path, root)}: {row.error}")


def write_csv(rows: list[ModelRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "params_m", "gmacs", "size_mb", "map50_95_pct"])
        for row in rows:
            writer.writerow(
                [
                    f"{row.path.parent.name}/{row.path.name}",
                    f"{row.params_m:.6f}",
                    "" if row.gmacs is None else f"{row.gmacs:.6f}",
                    f"{row.size_mb:.6f}",
                    (
                        ""
                        if row.map5095 is None
                        else f"{metric_to_percent(row.map5095):.4f}"
                    ),
                ]
            )


def main() -> None:
    args = parse_args()
    metrics = load_metrics(args.metrics_json)
    rows = collect_rows(args.root, metrics)

    if not rows:
        raise SystemExit(f"No .onnx files found under {args.root}")

    print_pretty_table(rows, args.root)

    if args.output_csv:
        write_csv(rows, args.output_csv)
        print(f"\nCSV written to: {args.output_csv}")


if __name__ == "__main__":
    main()
