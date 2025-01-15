import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
from deepface import DeepFace

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

def verify_faces(face1_bgr, face2_bgr, model_name='Facenet', distance_metric='cosine'):
    """
    Verifies whether face1 and face2 belong to the same person using DeepFace.
    Returns a dict with keys: { 'verified', 'distance', 'threshold', ... } 
    or None if something went wrong.
    """
    if face1_bgr is None or face2_bgr is None:
        print("[ERROR] One of the input faces is None.")
        return None

    # DeepFace expects RGB arrays (or file paths)
    face1_rgb = cv2.cvtColor(face1_bgr, cv2.COLOR_BGR2RGB)
    face2_rgb = cv2.cvtColor(face2_bgr, cv2.COLOR_BGR2RGB)

    try:
        result = DeepFace.verify(
            img1_path=face1_rgb,  
            img2_path=face2_rgb,
            model_name=model_name,
            distance_metric=distance_metric,
            enforce_detection=False  # because we already have cropped faces
        )
        return result
    except Exception as e:
        print(f"[ERROR] DeepFace verification failed: {e}")
        return None

if __name__ == "__main__":
    # Example file paths (replace with your own)
    id_image_path = "private/CI_Specimen0.jpeg" 
    selfie_image_path = "private/CI_Selfie0.jpeg"

    # Step 1: Detect and extract face from ID
    face_id = detect_and_extract_face_mtcnn(
        img_path=id_image_path,
        output_face_path=None,  # e.g., "cropped_id_face.jpg"
        show_result=True,
        device='cpu'            # or 'cuda' if you have a GPU
    )

    # Step 2: Detect and extract face from Selfie
    face_selfie = detect_and_extract_face_mtcnn(
        img_path=selfie_image_path,
        output_face_path=None,  # e.g., "cropped_selfie_face.jpg"
        show_result=True,
        device='cpu'            # or 'cuda' if you have a GPU
    )

    # Step 3: Verify the faces
    if face_id is not None and face_selfie is not None:
        verification_result = verify_faces(face_id, face_selfie, model_name='Facenet', distance_metric='cosine')
        
        if verification_result is not None:
            if verification_result['verified']:
                print(
                    f"[MATCH] The faces match! "
                    f"Distance: {verification_result['distance']:.3f}, "
                    f"Threshold: {verification_result['threshold']:.3f}"
                )
            else:
                print(
                    f"[NO MATCH] The faces do NOT match. "
                    f"Distance: {verification_result['distance']:.3f}, "
                    f"Threshold: {verification_result['threshold']:.3f}"
                )
        else:
            print("[INFO] Verification could not be performed.")
    else:
        print("[ERROR] One of the input faces is None.")