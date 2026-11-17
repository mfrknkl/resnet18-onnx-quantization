
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Project directories
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# Load results
training = pd.read_csv(RESULTS_DIR / "training_history.csv")
metrics = pd.read_csv(RESULTS_DIR / "metrics.csv")
latency = pd.read_csv(RESULTS_DIR / "latency.csv")
throughput = pd.read_csv(RESULTS_DIR / "throughput.csv")
model_size = pd.read_csv(RESULTS_DIR / "model_size.csv")


# 1. Training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(training["epoch"], training["train_loss"], label="Train Loss")
axes[0].plot(training["epoch"], training["val_loss"], label="Validation Loss")
axes[0].set_title("Training and Validation Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(
    training["epoch"],
    training["train_accuracy"] * 100,
    label="Train Accuracy"
)
axes[1].plot(
    training["epoch"],
    training["val_accuracy"] * 100,
    label="Validation Accuracy"
)
axes[1].set_title("Training and Validation Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "training_curves.png", dpi=300)
plt.close()


# 2. Accuracy and Macro F1 comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

models = metrics["model"].unique()
splits = ["validation", "test"]
x = np.arange(len(models))
width = 0.35

for i, split in enumerate(splits):
    subset = metrics[metrics["split"] == split]
    subset = subset.set_index("model").reindex(models)

    axes[0].bar(
        x + (i - 0.5) * width,
        subset["accuracy"] * 100,
        width,
        label=split.capitalize()
    )

    axes[1].bar(
        x + (i - 0.5) * width,
        subset["macro_f1"],
        width,
        label=split.capitalize()
    )

axes[0].set_title("Accuracy Comparison")
axes[0].set_ylabel("Accuracy (%)")
axes[0].set_ylim(75, 85)

axes[1].set_title("Macro F1 Comparison")
axes[1].set_ylabel("Macro F1")
axes[1].set_ylim(0.75, 0.85)

for ax in axes:
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    # Move legend above the plot
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False
    )

# Leave enough space for legends
plt.tight_layout()
plt.savefig(PLOTS_DIR / "accuracy_comparison.png", dpi=300)
plt.close()


# 3. Latency comparison
fig, ax = plt.subplots(figsize=(9, 5))

x = np.arange(len(latency))
width = 0.25

for i, metric in enumerate(["mean_ms", "median_ms", "p95_ms"]):
    ax.bar(
        x + (i - 1) * width,
        latency[metric],
        width,
        label=metric.replace("_ms", "").capitalize()
    )

ax.set_title("CPU Inference Latency (Batch Size = 1)")
ax.set_ylabel("Latency (ms)")
ax.set_xticks(x)
ax.set_xticklabels(latency["model"])
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "latency_comparison.png", dpi=300)
plt.close()


# 4. Throughput comparison
fig, ax = plt.subplots(figsize=(9, 5))

for model in throughput["model"].unique():
    subset = throughput[throughput["model"] == model]

    ax.plot(
        subset["batch_size"],
        subset["throughput_img_sec"],
        marker="o",
        label=model
    )

ax.set_title("CPU Inference Throughput")
ax.set_xlabel("Batch Size")
ax.set_ylabel("Images per Second")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "throughput_comparison.png", dpi=300)
plt.close()


# 5. Model size comparison
fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(model_size["model"], model_size["size_mb"])

ax.set_title("Model Size Comparison")
ax.set_ylabel("Model Size (MB)")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "model_size_comparison.png", dpi=300)
plt.close()


print(f"All plots saved to: {PLOTS_DIR}")
