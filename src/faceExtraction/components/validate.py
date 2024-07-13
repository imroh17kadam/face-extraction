import cv2
import numpy as np
from facenet_pytorch import MTCNN
import io
import streamlit as st
from PIL import Image  # Import Image module from PIL


class FaceExtractor:
    def __init__(self):
        # Initialize MTCNN for face detection
        self.mtcnn = MTCNN(keep_all=True)

    def process_image(self, image):
        browsed_path = 'c:\\Users\\DELL\\OneDrive\\Desktop\\MachineLearning\\face-extraction\\images\\output\\browsed_face.png'

        # Convert the image to numpy array
        img_array = np.array(image)

        # Convert RGB to BGR (OpenCV uses BGR format)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Detect faces
        boxes, _ = self.mtcnn.detect(img_bgr)

        if boxes is not None:
            if len(boxes) > 1:
                st.warning("More than one face detected. Please upload an image with only one face.")
                return False
            else:
                cv2.imwrite(browsed_path, img_array)
                return True
                # Extract the face with highest confidence score
                x1, y1, x2, y2 = boxes[0]
                face = img_bgr[int(y1):int(y2), int(x1):int(x2)]

                # Convert BGR to RGB (PIL uses RGB format)
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face_pil = Image.fromarray(face_rgb)

                # Convert the face to PNG format with transparent background
                face_pil = face_pil.convert("RGBA")
                datas = face_pil.getdata()

                new_data = []
                for item in datas:
                    if item[:3] == (0, 0, 0):
                        new_data.append((255, 255, 255, 0))
                    else:
                        new_data.append(item)

                face_pil.putdata(new_data)

                # Save the final output
                output_buffer = io.BytesIO()
                face_pil.save(output_buffer, format='PNG')
                output_buffer.seek(0)

                return face_pil, output_buffer
        else:
            st.warning("No face detected. Please upload another image.")
            return None
