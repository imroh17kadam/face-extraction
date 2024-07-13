import cv2
import numpy as np


class FaceDetector:
    def __init__(self, prototxt_path, model_path):
        # Load pre-trained face detection model from OpenCV's DNN module
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    
    def detect_and_save_face(self, image_path, output_path, confidence_threshold=0.5, padding=80):
        # Read the image
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error: Image not found at {image_path}")
            return False
        
        # Resize image to a fixed width and height and prepare it for face detection
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        
        # Pass the blob through the network and obtain the face detections
        self.net.setInput(blob)
        detections = self.net.forward()
        
        # Loop over the detections
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # Filter out weak detections by ensuring the confidence is greater than the minimum confidence
            if confidence > confidence_threshold:
                # Compute the (x, y)-coordinates of the bounding box for the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Expand the bounding box by a certain number of pixels (e.g., 20 pixels)
                startX = max(0, startX - padding)
                startY = max(0, startY - padding)
                endX = min(w, endX + padding)
                endY = min(h, endY)
                
                # Crop the face region from the image
                face_region = image[startY:endY, startX:endX]
                
                # Save the cropped face region as a new image
                cv2.imwrite(output_path, face_region)
                print(f"Detected face saved as {output_path}")
                return face_region
        
        # If no face detected, return False
        print("No face detected in the image.")
        return False