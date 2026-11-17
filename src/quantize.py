import numpy as np
from onnxruntime.quantization import (
    CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
    quantize_static,
)
from data import get_loaders
from model import ROOT

class CIFAR100CalibrationReader(CalibrationDataReader):
    def __init__(self, loader, input_name, max_batches=16):
        self.loader = loader
        self.input_name = input_name
        self.max_batches = max_batches
        self.rewind()

    def get_next(self):
        if self.batch_count >= self.max_batches:
            return None
        try:
            images, _ = next(self.iterator)
        except StopIteration:
            return None
        self.batch_count += 1
        return {self.input_name: images.numpy().astype(np.float32)}

    def rewind(self):
        self.iterator = iter(self.loader)
        self.batch_count = 0

def main():
    import onnxruntime as ort

    fp32_path = ROOT / "models" / "resnet18_finetuned_fp32.onnx"
    int8_path = ROOT / "models" / "resnet18_finetuned_int8.onnx"

    val_loader, _ = get_loaders()
    session = ort.InferenceSession(
        str(fp32_path), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    reader = CIFAR100CalibrationReader(val_loader, input_name, max_batches=16)

    quantize_static(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=True,
    )
    print(f"Quantized model saved: {int8_path}")

if __name__ == "__main__":
    main()
