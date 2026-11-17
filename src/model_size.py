import csv
from model import ROOT

def main():
    files = {
        "PyTorch FP32": ["best_resnet18_cifar100.pth"],
        "ONNX FP32": [
            "resnet18_finetuned_fp32.onnx",
            "resnet18_finetuned_fp32.onnx.data",
        ],
        "ONNX INT8": ["resnet18_finetuned_int8.onnx"],
    }
    rows = []
    for name, filenames in files.items():
        total = sum(
            (ROOT / "models" / filename).stat().st_size
            for filename in filenames
        )
        size_mb = total / (1024 ** 2)
        rows.append([name, size_mb])
        print(f"{name}: {size_mb:.2f} MB")

    out = ROOT / "results" / "model_size.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "size_mb"])
        writer.writerows(rows)

if __name__ == "__main__":
    main()
