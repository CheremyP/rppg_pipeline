import numpy as np
import torch
import torchvision
import cv2
import onnxruntime
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
from segment_anything.utils.onnx import SamOnnxModel

class SegmentAnything():
    def __init__(self, model_path, model_type, device):      
        sam_checkpoint = model_path

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device)
        self.predictor = SamPredictor(sam)

        onnx_model_path = "model_checkpoints/sam_onnx_example.onnx"
        onnx_model = SamOnnxModel(sam, return_single_mask=True)
        self.ort_session = onnxruntime.InferenceSession(onnx_model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    def segment_anything_onnx(self, image, input_point, input_label):
        image_embedding = self.predictor.get_image_embedding().cpu().numpy()
        onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
        onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)

        onnx_coord = self.predictor.transform.apply_coords(onnx_coord, image.shape[:2]).astype(np.float32)
        onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        onnx_has_mask_input = np.zeros(1, dtype=np.float32)

        ort_inputs = {
            "image_embeddings": image_embedding,
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
            "mask_input": onnx_mask_input,
            "has_mask_input": onnx_has_mask_input,
            "orig_im_size": np.array(image.shape[:2], dtype=np.float32)
        }

        masks, _, low_res_logits = self.ort_session.run(None, ort_inputs)
        masks = masks > self.predictor.model.mask_threshold
        return masks
    

def extract_eyes_mouth(image, landmarks, dilation_factor=5):
    left_eye_points = landmarks[0][0][36:42]  # Extract left eye landmarks
    right_eye_points = landmarks[0][0][42:48]  # Extract right eye landmarks
    mouth_points = landmarks[0][0][48:68]  # Extract mouth landmarks

    # Create masks for eyes and mouth
    eye_mask = np.zeros_like(image, dtype=np.uint8)
    mouth_mask = np.zeros_like(image, dtype=np.uint8)

    # Draw polygons for eyes and mouth regions on masks
    cv2.fillPoly(eye_mask, np.int32([left_eye_points]), (255, 255, 255))
    cv2.fillPoly(eye_mask, np.int32([right_eye_points]), (255, 255, 255))
    cv2.fillPoly(mouth_mask, np.int32([mouth_points]), (255, 255, 255))

    # Apply dilation to the masks
    eye_mask = cv2.dilate(eye_mask, None, iterations=int(dilation_factor))
    mouth_mask = cv2.dilate(mouth_mask, None, iterations=int(dilation_factor))

    # Apply masks to extract eyes and mouth regions
    extracted_eyes = cv2.bitwise_and(image, eye_mask)
    extracted_mouth = cv2.bitwise_and(image, mouth_mask)

    return extracted_eyes, extracted_mouth
