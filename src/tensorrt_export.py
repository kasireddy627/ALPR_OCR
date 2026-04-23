"""
TensorRT FP16 Export Script
----------------------------
NOTE: TensorRT requires an NVIDIA GPU with CUDA and TensorRT installed.
This machine is CPU-only, so export is skipped.
"""

import os
import sys
import time
import yaml
import numpy as np


def load_config():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base, "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def benchmark_pt_model(model, n=20):
    """Run n inference passes on a dummy frame and return avg latency ms."""
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    for _ in range(2):
        model(dummy, verbose=False)

    start = time.perf_counter()
    for _ in range(n):
        model(dummy, verbose=False)
    elapsed = (time.perf_counter() - start) * 1000
    return elapsed / n


def export_tensorrt():
    cfg = load_config()

    import torch
    has_cuda = torch.cuda.is_available()

    print("=" * 60)
    print("ALPR TensorRT Export Utility")
    print("=" * 60)
    print(f"  CUDA available : {has_cuda}")
    print(f"  use_tensorrt   : {cfg['model']['use_tensorrt']}")
    print("=" * 60)

    # Always run CPU benchmark
    print("\nRunning CPU benchmark (PT model)...")
    _run_cpu_benchmark(cfg)

    if not has_cuda:
        print("\n[SKIPPED] TensorRT export requires NVIDIA GPU.")
        print("  To export on a GPU machine:")
        print("  1. Install CUDA + TensorRT")
        print("  2. Set use_tensorrt: true in config.yaml")
        print("  3. Run: python src/tensorrt_export.py")
        print("\n[OK] CPU-only mode confirmed. Use .pt model directly.")
        return

    # GPU export path
    try:
        from ultralytics import YOLO

        base         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        weights_path = os.path.join(base, cfg["model"]["weights_path"])
        engine_path  = os.path.join(base, cfg["model"]["engine_path"])

        model = YOLO(weights_path)

        print("\nExporting to TensorRT FP16...")
        model.export(format="engine", half=True, device=0, workspace=4)

        if os.path.exists(engine_path):
            size_mb = os.path.getsize(engine_path) / (1024 * 1024)
            print(f"Engine saved: {engine_path} ({size_mb:.1f} MB)")

        print("\nBenchmarking TensorRT engine...")
        trt_model   = YOLO(engine_path)
        trt_latency = benchmark_pt_model(trt_model, n=20)
        print(f"  TensorRT avg latency: {trt_latency:.2f} ms/frame")

    except Exception as e:
        print(f"[ERROR] TensorRT export failed: {e}")


def _run_cpu_benchmark(cfg):
    try:
        from ultralytics import YOLO
        base         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        weights_path = os.path.join(base, cfg["model"]["weights_path"])

        model   = YOLO(weights_path)
        latency = benchmark_pt_model(model, n=20)

        print(f"  CPU PT model avg latency : {latency:.2f} ms/frame")
        print(f"  Expected TensorRT on GPU : ~11 ms/frame")
        print(f"  Estimated GPU speedup    : ~{latency/11:.1f}x")

    except Exception as e:
        print(f"  Benchmark error: {e}")


if __name__ == "__main__":
    export_tensorrt()