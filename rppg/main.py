import cv2
import numpy as np
import gc
from rppg.region_of_interest.onnx_segment_anything import SegmentAnything, extract_eyes_mouth
from rppg.signal_processing import post_processing
from rppg.rppg_algorithms import chormicance, independent_component_analysis, plane_orthogonal_to_skin
from rppg.hearrate_estimation import heartrate_estimation

from tqdm import tqdm

class rppg_pipeline():
    def __init__(self, model_path, model_type, device):
        self.segment_anything = SegmentAnything(model_path, model_type, device)
        self.predictor_model = self.segment_anything.predictor

    def extract_frames_yield(self, videoFileName):
        vidcap = cv2.VideoCapture(videoFileName)
        success, image = vidcap.read()

        while success:
            yield image
            success, image = vidcap.read()
    
    def spatial_averaging(self, frames):
        frames_array = np.array(frames)
        num_frames = frames_array.shape[0]
        RGB = np.zeros((num_frames, 3))

        for i in range(num_frames):
            R = np.sum(np.sum(frames_array[i, :, :, 0]))
            G = np.sum(np.sum(frames_array[i, :, :, 1]))
            B = np.sum(np.sum(frames_array[i, :, :, 2]))
            num_pixels = np.count_nonzero(frames_array[i, :, :, 0])
            RGB[i] = [R / num_pixels, G / num_pixels, B / num_pixels]

        return RGB
    
    def get_point(self, frame):
      face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

      gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

      # Detect faces using Haar cascade
      faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

      if len(faces) == 0:
          return frame

      # Get the first detected face
      x, y, w, h = faces[0]
      center_x = x + w/2
      center_y = y + h/2
      top_x = center_x
      top_y = y*1.4
      bottom_x = center_x
      bottom_y = y + h
      left_x = x*1.25
      left_y = center_y
      right_x = x*0.75 + w
      right_y = center_y

      return np.array([[center_x, center_y],[ top_x, top_y], [bottom_x, bottom_y], [left_x, left_y], [right_x, right_y]])

    def process_signal(self, signal, post_processing, heartrate_estimation):
        filtered_signal = post_processing.detrend(signal)
        filtered_signal = post_processing.standardisation(filtered_signal)
        filtered_signal = post_processing.cheby2_bandpass_filter(filtered_signal, 0.7, 3, 30, order=4)

        hr_fft = heartrate_estimation.calculate_fft_hr(filtered_signal)
        hr_peak = heartrate_estimation.calculate_peak_hr(filtered_signal)

        return filtered_signal, hr_fft, hr_peak


    def main(self, video_path, methods= "all", show=True, save=True):
        sig = []
        frame_counter = 14
        input_label = np.array([1,1,1,1,1])
        mask_image, final_mask, previous_mask = None, None, None

        for frame in tqdm(self.extract_frames_yield(video_path)):
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

                    masks = self.segment_anything.segment_anything_onnx(image, input_point, input_label)

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
        
        # load a signal
        # unprocessed_signal = np.loadtxt("_11_signal.txt")

        # Method HR estimation
        if methods == "all":
            chrom = chormicance.chormicance(unprocessed_signal)
            ica = independent_component_analysis.independent_component_analysis(unprocessed_signal)
            pos = plane_orthogonal_to_skin.plane_orthoganal_to_skin(unprocessed_signal)

            filtered_signal_chrom, hr_chrom_fft, hr_chrom_peak = self.process_signal(chrom, post_processing, heartrate_estimation)
            filtered_signal_ica, hr_ica_fft, hr_ica_peak = self.process_signal(ica[0,:], post_processing, heartrate_estimation)
            filtered_signal_pos, hr_pos_fft, hr_pos_peak = self.process_signal(pos[0,:], post_processing, heartrate_estimation)

        elif methods == "chromicance":
            chrom = chormicance.chormicance(unprocessed_signal)
            filtered_signal_chrom, hr_chrom_fft, hr_chrom_peak = self.process_signal(chrom, post_processing, heartrate_estimation)

        elif methods == "ica":
            ica = independent_component_analysis.independent_component_analysis(unprocessed_signal)
            filtered_signal_ica, hr_ica_fft, hr_ica_peak = self.process_signal(ica[0,:], post_processing, heartrate_estimation)

        elif methods == "pos":
            pos = plane_orthogonal_to_skin.plane_orthoganal_to_skin(unprocessed_signal)
            filtered_signal_pos, hr_pos_fft, hr_pos_peak = self.process_signal(pos[0,:], post_processing, heartrate_estimation)

        else:
            raise ValueError("Invalid method")

        # Show the results
        if show:
            print("Fast Fourier Transform")
            print("Chromicance FFT HR: ", hr_chrom_fft)
            print("ICA FFT HR: ", hr_ica_fft)
            print("POS FFT HR: ", hr_pos_fft)
            print('\n')
            print("Peak Detection")
            print("Chromicance Peak HR: ", hr_chrom_peak)
            print("ICA Peak HR: ", hr_ica_peak)
            print("POS Peak HR: ", hr_pos_peak)

        # Save the signal
        if save:
            np.savetxt(str(video_path) + "unprocessed_signal.txt", unprocessed_signal)
            np.savetxt(str(video_path) + "filtered_chrom.txt", filtered_signal_chrom)
            np.savetxt(str(video_path) + "filtered_ica.txt", filtered_signal_ica)
            np.savetxt(str(video_path) + "filtered_pos.txt", filtered_signal_pos)

        # Delete video from RAM
        if mask_image is not None:
            del mask_image
        if sig is not None:
            del sig
        if final_mask is not None:
            del final_mask
        gc.collect() 

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
landmark_model = cv2.face.createFacemarkLBF()
landmark_model.loadModel("model_checkpoints/lbfmodel.yaml")

if __name__ == "__main__":

    # Set the parameters
    video_path = "vid.mp4"
    model_path = "model_checkpoints/sam_vit_b_01ec64.pth"
    model_type = "vit_b" # "vit_h" "vit_l"
    device = "cuda" # "cpu"

    pipepline = rppg_pipeline(model_path, model_type, device)
    pipepline.main(video_path, methods="all", show=True, save=True)