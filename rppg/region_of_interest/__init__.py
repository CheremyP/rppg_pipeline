# import onnxruntime
# from onnxruntime.quantization import QuantType
# from onnxruntime.quantization.quantize import quantize_dynamic
# from segment_anything import SamAutomaticMaskGenerator,sam_model_registry, SamPredictor
# from segment_anything.utils.onnx import SamOnnxModel


# sam_checkpoint = "model_checkpoints/sam_vit_b_01ec64.pth"
# model_type = "vit_b"
# onnx_model_path = "model_checkpoints/sam_onnx_example.onnx"
# device = "cuda"


# sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# sam.to(device)
# predictor = SamPredictor(sam)
# ort_session = onnxruntime.InferenceSession(onnx_model_path, providers =" CUDAExecutionProvider")
# onnxruntime.get_device()