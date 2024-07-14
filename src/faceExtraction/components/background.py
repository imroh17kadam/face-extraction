from rembg import remove
import cv2
from PIL import Image
import io
import numpy as np


class BackgroundRemover:
    def __init__(self, input_path, output_path, image_array):
        self.input_path = input_path
        self.output_path = output_path
        self.image_array = image_array

    def remove_background(self):
        # Load the image
        input_image = cv2.imread(self.input_path)
        
        resized_image = self.resize_to_square(input_image)

        # Remove the background
        output_image = remove(resized_image)

        face_pil = Image.fromarray(output_image)
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

        # Save the image
        cv2.imwrite(self.output_path, output_image)
        print(f"Image saved to {self.output_path}")

        return output_image, output_buffer
    
    def resize_to_square(self, image):
        # Get dimensions of the image
        h, w = image.shape[:2]

        # Determine the size of the square
        size = max(h, w)

        # Create a square background
        square_bg = np.zeros((size, size, 3), dtype=np.uint8)
        start_h = (size - h) // 2
        start_w = (size - w) // 2
        square_bg[start_h:start_h + h, start_w:start_w + w] = image

        return square_bg