import csv
import numpy as np
import onnxruntime as ort
import torch
from sklearn.metrics import accuracy_score, f1_score
from data import get_loaders
from model import ROOT, load_model

def evaluate_pytorch(model, loader):
    preds, labels_all = [], []
    with torch.inference_mode():
        for images, labels in loader:
            preds.extend(model(images).argmax(1).tolist())
            labels_all.extend(labels.tolist())
    return accuracy_score(labels_all, preds), f1_score(
        labels_all, preds, average="macro"
    )

def evaluate_onnx(session, loader):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    preds, labels_all = [], []
    for images, labels in loader:
        outputs = session.run(
            [output_name], {input_name: images.numpy().astype(np.float32)}
        )[0]
        preds.extend(outputs.argmax(axis=1).tolist())
        labels_all.extend(labels.tolist())
    return accuracy_score(labels_all, preds), f1_score(
        labels_all, preds, average="macro"
    )

def main():
    model = load_model()
    val_loader, test_loader = get_loaders()
    fp32 = ort.InferenceSession(
        str(ROOT / "models/resnet18_finetuned_fp32.onnx"),
        providers=["CPUExecutionProvider"],
    )
    int8 = ort.InferenceSession(
        str(ROOT / "models/resnet18_finetuned_int8.onnx"),
        providers=["CPUExecutionProvider"],
    )

    rows = []
    for split, loader in [("validation", val_loader), ("test", test_loader)]:
        results = [
            ("PyTorch FP32", evaluate_pytorch(model, loader)),
            ("ONNX FP32", evaluate_onnx(fp32, loader)),
            ("ONNX INT8", evaluate_onnx(int8, loader)),
        ]
        for name, (acc, f1) in results:
            rows.append([split, name, acc, f1])
            print(f"{split:10s} | {name:12s} | Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")

    out = ROOT / "results" / "metrics.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "model", "accuracy", "macro_f1"])
        writer.writerows(rows)
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
