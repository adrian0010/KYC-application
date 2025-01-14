import cv2
import numpy as np
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN

def detect_and_extract_face_mtcnn(img_path, output_face_path=None, show_result=False, device='cpu'):
    """
    Detects a face in the given image using MTCNN and extracts it.
    Optionally saves the cropped face to 'output_face_path'.
    If show_result=True, displays the result via matplotlib.
    Returns the cropped face as a NumPy array (BGR) or None if no face is found.
    """
    # Initialize MTCNN (you can do this once outside the function if preferred)
    mtcnn = MTCNN(keep_all=True, device=device)

    # Read image with OpenCV (BGR format)
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"[ERROR] Could not load image {img_path}")
        return None

    # Convert BGR to RGB for MTCNN
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Detect faces
    boxes, probs = mtcnn.detect(img_rgb)

    if boxes is None or len(boxes) == 0:
        print(f"[INFO] No face detected in {img_path}")
        return None

    # Find the face with the highest probability
    max_prob_idx = np.argmax(probs)
    box = boxes[max_prob_idx]  # [x1, y1, x2, y2]
    conf = probs[max_prob_idx]

    print(f"[INFO] Detected face in {img_path} with confidence: {conf:.3f}")

    # Convert bounding box to integers
    x1, y1, x2, y2 = [int(coord) for coord in box]
    
    # Clamp coordinates to image boundaries
    h, w, _ = img_bgr.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    # Crop the face region
    face_cropped_bgr = img_bgr[y1:y2, x1:x2]

    # Optionally save the cropped face
    if output_face_path:
        cv2.imwrite(output_face_path, face_cropped_bgr)

    # Display the detection if requested
    if show_result:
        # Draw a rectangle on the original image (for visualization)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Convert images for matplotlib
        img_bgr_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        face_cropped_rgb = cv2.cvtColor(face_cropped_bgr, cv2.COLOR_BGR2RGB)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        ax1.imshow(img_bgr_rgb)
        ax1.set_title("ID Image with Detected Face (MTCNN)")
        ax1.axis("off")

        ax2.imshow(face_cropped_rgb)
        ax2.set_title("Cropped Face")
        ax2.axis("off")

        plt.tight_layout()
        plt.show()

    return face_cropped_bgr
