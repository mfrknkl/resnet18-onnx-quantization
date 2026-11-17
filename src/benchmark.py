import csv
import time
import numpy as np
import onnxruntime as ort
import torch
from data import get_loaders
from model import ROOT, load_model

def pytorch_inference(model, images):
    with torch.inference_mode():
        return model(images).argmax(dim=1)

def onnx_inference(session, images):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    outputs = session.run(
        [output_name], {input_name: images.numpy().astype(np.float32)}
    )[0]
    return outputs.argmax(axis=1)

def measure_latency(fn, model, images, warmup=50, iterations=1000):
    images = images[:1].cpu()
    for _ in range(warmup):
        fn(model, images)
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn(model, images)
        times.append((time.perf_counter() - start) * 1000)
    return [np.mean(times), np.median(times), np.percentile(times, 95)]

def measure_throughput(fn, model, images, batch_sizes=(1, 16, 32, 64),
                       warmup=20, iterations=100):
    rows = []
    for batch_size in batch_sizes:
        batch = images[:batch_size].cpu()
        for _ in range(warmup):
            fn(model, batch)
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            fn(model, batch)
            times.append(time.perf_counter() - start)
        mean_seconds = float(np.mean(times))
        rows.append([batch_size, mean_seconds * 1000, batch_size / mean_seconds])
    return rows

def main():
    model = load_model()
    val_loader, _ = get_loaders()
    images, _ = next(iter(val_loader))
    fp32 = ort.InferenceSession(
        str(ROOT / "models/resnet18_finetuned_fp32.onnx"),
        providers=["CPUExecutionProvider"],
    )
    int8 = ort.InferenceSession(
        str(ROOT / "models/resnet18_finetuned_int8.onnx"),
        providers=["CPUExecutionProvider"],
    )
    runners = [
        ("PyTorch FP32", pytorch_inference, model),
        ("ONNX FP32", onnx_inference, fp32),
        ("ONNX INT8", onnx_inference, int8),
    ]

    latency_rows = []
    throughput_rows = []
    for name, fn, runner_model in runners:
        mean, median, p95 = measure_latency(fn, runner_model, images)
        latency_rows.append([name, 1, mean, median, p95])
        for batch_size, mean_ms, throughput in measure_throughput(
            fn, runner_model, images
        ):
            throughput_rows.append([name, batch_size, mean_ms, throughput])

    with (ROOT / "results/latency.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "batch_size", "mean_ms", "median_ms", "p95_ms"])
        writer.writerows(latency_rows)

    with (ROOT / "results/throughput.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "batch_size", "mean_latency_ms", "throughput_img_sec"])
        writer.writerows(throughput_rows)

    print("Saved latency.csv and throughput.csv")

if __name__ == "__main__":
    main()
