# TBD

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from facenet_pytorch import MTCNN

# Initialize MTCNN
mtcnn = MTCNN(keep_all=True, device='cpu') #device='cuda' for GPU

def detect_and_extract_face(img_path, output_face_path=None, show_result=True):
    # Read the image with OpenCV
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"Error: Could not load image {img_path}")
        return None
    
    # Convert BGR to RGB for MTCNN
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    boxes, probs = mtcnn.detect(img_rgb)

    # If no faces are detected, return
    if boxes is None or len(boxes) == 0:
        print("No face detected.")
        return None
    
    # In case there are multiple faces, we select the one with the highest probability
    max_prob_idx = np.argmax(probs)
    box = boxes[max_prob_idx]  # [x1, y1, x2, y2]
    conf = probs[max_prob_idx]
    
    print(f"Detected face with confidence: {conf:.3f}")
    
    # Convert coordinates to integers
    x1, y1, x2, y2 = [int(coord) for coord in box]
    
    # Crop the face
    face_cropped_bgr = img_bgr[y1:y2, x1:x2]
    
    # Optionally save the face
    if output_face_path:
        cv2.imwrite(output_face_path, face_cropped_bgr)
    
    # Show the result if requested
    if show_result:
        # Draw a rectangle on the original image (for visualization only)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Convert back to RGB for display
        img_bgr_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12, 6))
        
        # Show original image with bounding box
        plt.subplot(1, 2, 1)
        plt.imshow(img_bgr_rgb)
        plt.title("ID Image with Detected Face")
        plt.axis("off")
        
        # Show extracted face
        face_cropped_rgb = cv2.cvtColor(face_cropped_bgr, cv2.COLOR_BGR2RGB)
        plt.subplot(1, 2, 2)
        plt.imshow(face_cropped_rgb)
        plt.title("Extracted Face")
        plt.axis("off")
        
        plt.tight_layout()
        plt.show()

    return face_cropped_bgr

# Example usage
if __name__ == "__main__":
    img_path = "images/CI_Specimen7.png"
    output_face_path = "output/DeepFace_cropped_face7.jpg"
    
    cropped_face = detect_and_extract_face(
        img_path=img_path,
        output_face_path=output_face_path,
        show_result=True
    )
    if cropped_face is not None:
        print("Face extraction successful.")
    else:
        print("Face extraction failed or no face found.")
