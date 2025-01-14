import os
from face_detection_mtcnn import detect_and_extract_face_mtcnn  # Import the function
from face_verification_deepface import verify_faces  # Import the verify_faces function

if __name__ == "__main__":
    # Example file paths (replace with your own)
    id_image_path = "images/CI_Specimen4.jpg"
    selfie_image_path = "images/CI_Selfie0.jpeg"

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