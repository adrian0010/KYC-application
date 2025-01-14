import cv2
from deepface import DeepFace

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