import streamlit as st
from PIL import Image
import cv2
from src.faceExtraction.components.validate import FaceValidator
from src.faceExtraction.components.detection import FaceDetector
from src.faceExtraction.components.background import BackgroundRemover
from src.faceExtraction.components.extraction import FaceExtractor

prototxt_path = 'models/deploy.prototxt.txt'
model_path = 'models/res10_300x300_ssd_iter_140000.caffemodel'
    
browsed_path = 'images/output/browsed_face.png'
detected_path = 'images/output/detected_face.png'
extracted_path = 'images/output/extracted_face.png'
transparent_path = 'images/output/transparent_face.png'

st.title("Face Extraction from Image")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Process Image"):
        face_validate = FaceValidator()
        validation, input_image = face_validate.process_image(image)

        face_detector = FaceDetector(prototxt_path, model_path)

        face_extractor = FaceExtractor()

        if validation:
            cropped_image = face_detector.detect_and_save_face(browsed_path, detected_path)

            face_extractor.read_image(detected_path)
            face_extractor.predict()
            extracted_face_image = face_extractor.extract_faces()
            face_extractor.save_image(extracted_face_image, extracted_path)
            print(f"Extracted face saved to {extracted_path}")

            background_remover = BackgroundRemover(extracted_path, transparent_path, input_image)
            output_image, output_buffer = background_remover.remove_background()
        else:
            output_image, output_buffer = None

        if output_image is not None:
            st.image(output_image, caption="Extracted Face", use_column_width=True)

            # Provide a download button for the user
            st.markdown("### Download Extracted Face")
            st.download_button(
                label="Click here to download",
                data=output_buffer,
                file_name="transparent_face.png",
                mime="image/png"
            )

            st.success("Face extracted and ready for download.")