# RetinaFace from https://pypi.org/project/retina-face/
import cv2
import matplotlib.pyplot as plt
from retinaface import RetinaFace
import tensorflow as tf

def detect_and_extract_face_retinaface(img_path, output_face_path=None, show_result=True, threshold=0.9):
    # Read the image with OpenCV
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"Error: Could not load image {img_path}")
        return None

    # Detect faces
    faces = RetinaFace.detect_faces(img_path, threshold=threshold)
    if not faces:
        print("No faces detected.")
        return None

    # Extract the first detected face
    face_key = list(faces.keys())[0]
    face = faces[face_key]
    facial_area = face['facial_area']
    cropped_face = img_bgr[facial_area[1]:facial_area[3], facial_area[0]:facial_area[2]]

    # Save the cropped face if output path is provided
    if output_face_path:
        cv2.imwrite(output_face_path, cropped_face)

    # Show the result if requested
    if show_result:
        plt.imshow(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    return cropped_face

# Example usage
if __name__ == "__main__":
    img_path = "images/CI_Specimen3.jpg"
    output_face_path = "output/RetinaFace_cropped_face3.jpg"
    detect_and_extract_face_retinaface(img_path, output_face_path)