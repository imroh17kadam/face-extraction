import streamlit as st
from PIL import Image
import cv2
from src.faceExtraction.components.validate import FaceValidator
from src.faceExtraction.components.detection import FaceDetector
from src.faceExtraction.components.background import BackgroundRemover

prototxt_path = 'models/deploy.prototxt.txt'
model_path = 'models/res10_300x300_ssd_iter_140000.caffemodel'
    
browsed_path = 'images/output/browsed_face.png'
output_path = 'images/output/detected_face.png'
transparent_path = 'images/output/transparent_face.png'

st.title("Face Extraction from Image")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Process Image"):
        face_validator = FaceValidator()
        face_detector = FaceDetector(prototxt_path, model_path)

        validation, input_image = face_validator.process_image(image)

        background_remover = BackgroundRemover(output_path, transparent_path, input_image)

        if validation:
            cropped_image = face_detector.detect_and_save_face(browsed_path, output_path)
            output_image, output_buffer = background_remover.remove_background()
        else:
            output_image, output_buffer = None, None

        if output_image is not None:
            st.image(output_image, caption="Extracted Face", use_column_width=True)

            # Provide a download button for the user
            st.markdown("### Download Extracted Face")
            st.download_button(
                label="Click here to download",
                data=output_buffer,
                file_name="extracted_face.png",
                mime="image/png"
            )

            st.success("Face extracted and ready for download.")