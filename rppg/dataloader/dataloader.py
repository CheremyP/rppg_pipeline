import cv2
import numpy as np
import os
from tqdm import tqdm

class DataLoader:
    """ A class used to load videos and extract frames from them.	"""

    def __init__(self, main_directory, subjects, sam_model, predictor_model):
        self.main_directory = main_directory
        self.subjects = subjects
        self.n_sub_videos = range(1, 6)
        self.sam_model = sam_model
        self.predictor_model = predictor_model
        self.sub_video = None
        self.subject = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def extract_frames_yield(self, videoFileName):
        vidcap = cv2.VideoCapture(videoFileName)
        success, image = vidcap.read()

        while success:
            yield image
            success, image = vidcap.read()

    def load_videos(self):
      for subject in tqdm(self.subjects, desc="Subjects"):
          self.subject = subject
          sub_dir = self.main_directory + str(subject) + "/" + str(subject)
          for sub_video in tqdm(self.n_sub_videos, desc="Sub Video"):
              self.sub_video = sub_video
              sub_dir_videos = sub_dir + "_" + str(sub_video) + "/vid.avi"
              if os.path.exists(sub_dir_videos):
                  self.process_video(sub_dir_videos)
              else:
                  return NameError("Directory does not exist")

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

      gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      # Detect faces using Haar cascade
      faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

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
