import streamlit as st
from PIL import Image
import cv2
from src.faceExtraction.components.validate import FaceExtractor
from src.faceExtraction.components.detection import FaceDetector
from src.faceExtraction.components.background import BackgroundRemover

prototxt_path = 'c:\\Users\\DELL\\OneDrive\\Desktop\\MachineLearning\\face-extraction\\models\\deploy.prototxt.txt'
model_path = 'c:\\Users\\DELL\\OneDrive\\Desktop\\MachineLearning\\face-extraction\\models\\res10_300x300_ssd_iter_140000.caffemodel'
    
output_path = 'c:\\Users\\DELL\\OneDrive\\Desktop\\MachineLearning\\face-extraction\\images\\output\\detected_face.png'
browsed_path = 'c:\\Users\\DELL\\OneDrive\\Desktop\\MachineLearning\\face-extraction\\images\\output\\browsed_face.png'
transparent_path = 'c:\\Users\\DELL\\OneDrive\\Desktop\\MachineLearning\\face-extraction\\images\\output\\transparent.png'

st.title("Face Extraction from Image")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Process Image"):
        face_extractor = FaceExtractor()
        face_detector = FaceDetector(prototxt_path, model_path)
        background_remover = BackgroundRemover(output_path, transparent_path)

        input_image = face_extractor.process_image(image)

        if input_image:
            cropped_image = face_detector.detect_and_save_face(browsed_path, output_path)
            output_image = background_remover.remove_background()
        else:
            output_image = None

        if output_image is not None:
            st.image(output_image, caption="Extracted Face", use_column_width=True)

            # Provide a download button for the user
            st.markdown("### Download Extracted Face")
            st.download_button(
                label="Click here to download",
                data=output_image,
                file_name="extracted_face.png",
                mime="image/png"
            )

            st.success("Face extracted and ready for download.")