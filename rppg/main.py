import cv2
import numpy as np
import gc
from rppg.region_of_interest.onnx_segment_anything import SegmentAnything, extract_eyes_mouth

class rppg_pipeline():
    def __init__(self):
        self.segment_anything = SegmentAnything()
    
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

                    # Start ONNX
                    masks = self.segment_anything.segment_anything_onnx(image, input_point, input_label)
                    # End MASKS

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

        # Save the signal
        np.savetxt("/content/drive/MyDrive/UCLAsignal" + "/_" + str(self.subject ) +
        str(self.sub_video) + "_signal.txt", unprocessed_signal)

        #TODO: Start processing the signal

        # Delete video from RAM
        del sig, masked_video, final_mask
        gc.collect()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
landmark_model = cv2.face.createFacemarkLBF()
video_path = "UCLA/subject_1/video_1.mp4"    

if __name__ == "__main__":
    rppg_pipeline().main(video_path)