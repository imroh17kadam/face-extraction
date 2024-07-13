from rembg import remove
import cv2


class BackgroundRemover:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def remove_background(self):
        # Load the image
        input_image = cv2.imread(self.input_path)
        
        # Remove the background
        output_image = remove(input_image)
        
        # Save the image
        cv2.imwrite(self.output_path, output_image)
        print(f"Image saved to {self.output_path}")
        return output_image