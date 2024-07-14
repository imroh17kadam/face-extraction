import cv2
import numpy as np
from facenet_pytorch import MTCNN
import streamlit as st


class FaceValidator:
    def __init__(self):
        # Initialize MTCNN for face detection
        self.mtcnn = MTCNN(keep_all=True)

    def process_image(self, image):
        browsed_path = 'images/output/browsed_face.png'

        # Convert the image to numpy array
        img_array = np.array(image)

        # Convert RGB to BGR (OpenCV uses BGR format)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

        # Detect faces
        boxes, _ = self.mtcnn.detect(img_bgr)

        if boxes is not None:
            if len(boxes) > 1:
                st.warning("More than one face detected. Please upload an image with only one face.")
                return False, []
            else:
                cv2.imwrite(browsed_path, img_array)
                return True, boxes[0]
        else:
            st.warning("No face detected. Please upload another image.")
            return None