import cv2
import numpy as np
import gc

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
landmark_model = cv2.face.createFacemarkLBF()

def main(self, video_path):
    sig = []
    frame_counter = 14
    input_label = np.array([1,1,1,1,1])

    for frame in self.extract_frames_yield(video_path):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            frame_counter += 1
            _, landmarks = landmark_model.fit(gray, faces)
            eyes, mouth = extract_eyes_mouth(image, landmarks)

            if frame_counter % 15 == 0:
                input_point = self.get_point(image)
                self.predictor_model.set_image(image)

                # Start
                image_embedding = predictor.get_image_embedding().cpu().numpy()
                onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
                onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)

                onnx_coord = predictor.transform.apply_coords(onnx_coord, image.shape[:2]).astype(np.float32)
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

                masks, _, low_res_logits = ort_session.run(None, ort_inputs)
                masks = masks > predictor.model.mask_threshold
                # End

                h, w = masks[0].shape[-2:]
                mask_image = masks[0].reshape(h, w, 1)
                masked_video = np.multiply(mask_image,  image)

                eyes[eyes > 0] = 1
                mouth[mouth > 0] = 1

                eyes = eyes.astype(np.uint8)
                mouth = mouth.astype(np.uint8)

                complement_eyes = 1 - eyes
                complement_mouth = 1 - mouth

                # Remove the eyes and mouth from the masked image
                final_mask = masked_video * complement_eyes * complement_mouth
                try:
                    eyes[eyes > 0] = 1
                    mouth[mouth > 0] = 1
                    eyes = eyes.astype(np.uint8)
                    mouth = mouth.astype(np.uint8)
                    complement_eyes = 1 - eyes
                    complement_mouth = 1 - mouth

                    if frame_counter % 5 == 0:
                        final_mask = masked_video * complement_eyes * complement_mouth
                        previous_mask = final_mask  # Update the previous mask
                    else:
                        previous_mask_gray = (previous_mask > 0).astype(np.uint8)
                        final_mask = np.multiply(previous_mask_gray, image) * complement_eyes * complement_mouth

                    sig.append(final_mask)

                except IndexError:
                    continue

    # Process the video
    unprocessed_signal = self.spatial_averaging(sig)

    np.savetxt("/content/drive/MyDrive/UCLAsignal" + "/_" + str(self.subject ) +
    str(self.sub_video) + "_signal.txt", unprocessed_signal)

    # Delete video from RAM
    del sig, masked_video, final_mask
    gc.collect()

if __name__ == "__main__":
    main()