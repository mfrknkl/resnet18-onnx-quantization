import torch
from model import ROOT, load_model

def main():
    model = load_model()
    dummy_input = torch.randn(1, 3, 224, 224)

    output_path = ROOT / "models" / "resnet18_finetuned_fp32.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=18,
    )
    print(f"Exported: {output_path}")

if __name__ == "__main__":
    main()
