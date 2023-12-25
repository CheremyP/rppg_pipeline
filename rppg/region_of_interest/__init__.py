import onnxruntime
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic
from segment_anything import SamAutomaticMaskGenerator,sam_model_registry, SamPredictor
from segment_anything.utils.onnx import SamOnnxModel


sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
onnx_model_path = "sam_onnx_example.onnx"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)
ort_session = onnxruntime.InferenceSession(onnx_model_path, providers =" CUDAExecutionProvider")
onnxruntime.get_device()