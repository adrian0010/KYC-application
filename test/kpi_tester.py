import os
import json
import cv2
import numpy as np
from datetime import datetime

############################################
# 1. SAMPLE TEST DATA (manually labeled)
############################################
# Each entry should match your local file paths and ground-truth
# "same_person" indicates whether the ID image and selfie
# truly belong to the same individual.

test_data = [
    {
        "id_image": "test_images/test_id_1.jpg",
        "selfie_image": "test_images/test_selfie_1.jpg",
        "expected_ocr_fields": {
            "first_name": "ELENA"
            ,"last_name": "VASILESCU"
            ,"birth_date": "15-Apr-41"
            ,"CNP": "2410415400342"
        },
        "same_person": True
    },
    {
        "id_image": "test_images/test_id_2.jpg",
        "selfie_image": "test_images/test_selfie_2.jpg",
        "expected_ocr_fields": {
            "first_name": "CRISTIAN CONSTANTIN"
            ,"last_name": "BALEA"
            ,"birth_date": "01-May-02"
            ,"CNP": "5020501015564"
        },
        "same_person": True
    },
    {
        "id_image": "test_images/test_id_3.jpg",
        "selfie_image": "test_images/test_selfie_3.jpg",
        "expected_ocr_fields": {
            "first_name": "GICU ROMEO"
            ,"last_name": "TITA"
            ,"birth_date": "22-Feb-62"
            ,"CNP": "1620222400084"
        },
        "same_person": False
    }

]

############################################
# 2. LABELS & OCR / MRZ FUNCTIONS
############################################

# For OCR
from paddleocr import PaddleOCR, draw_ocr

# Initialize PaddleOCR (adjust version & language if needed)
ocr = PaddleOCR(use_angle_cls=True, lang='ro')

# Labels to exclude
labels_to_exclude = [
    "ROMANIA", "ROUMANIE", "CARTE", "CARTE DE IDENTITATE",
    "IDENTITY", "D'IDENTITE", "SERIA", "NR", "CARD", "CNP",
    "Nume/Nom/Lastname", "Prenume/Prenom/Firstname",
    "Cetatenie/Nationalite/Nationality", "Sex/Sexe/Sex",
    "Romanä\\ROU", "Loc nastere/Lieu de naissance/Placeof birth",
    "Jud", "Jud.", "SPCLEP", "Domiciliu/Adresse/Address",
    "717", "evo", "Emisäde/Delivree par/lssued by",
    "Valabilitate/Validite/Validity"
]

import re

def is_mrz_line(text):
    """Checks if a text line matches the pattern for an MRZ line."""
    return bool(re.match(r'^[A-Z0-9<]{30,44}$', text))

def parse_mrz(mrz_lines):
    if len(mrz_lines) == 2 and all(len(line) == 36 for line in mrz_lines):
        line1, line2 = mrz_lines
        document_type = line1[0:2]
        issuing_country = line1[2:5]
        names = line1[5:].split('<<', 1)
        last_name = names[0].replace('<', ' ').strip()
        first_name = names[1].replace('<', ' ').strip() if len(names) > 1 else ''
        id_number = line2[0:9].replace('<', '')
        id_number_check_digit = line2[9]
        nationality = line2[10:13]
        birth_date_raw = line2[13:19]
        birth_date = datetime.strptime(birth_date_raw, '%y%m%d').strftime('%d-%b-%y') if birth_date_raw.isdigit() else birth_date_raw
        birth_date_check_digit = line2[19]
        sex = line2[20]
        expiration_date_raw = line2[21:27]
        expiration_date = datetime.strptime(expiration_date_raw, '%y%m%d').strftime('%d-%b-%y') if expiration_date_raw.isdigit() else expiration_date_raw
        expiration_date_check_digit = line2[27]
        cnp_series = line2[28]
        cnp_number = line2[29:35]
        final_check_digit = line2[35]
        cnp = f"{cnp_series}{birth_date_raw}{cnp_number}"
        

        return {
            "document_type": document_type,
            "issuing_country": issuing_country,
            "last_name": last_name,
            "first_name": first_name,
            "nationality": nationality,
            "birth_date": birth_date,
            "birth_date_check_digit": birth_date_check_digit,
            "sex": sex,
            "CNP": cnp,
            "id_number": id_number,
            "id_number_check_digit": id_number_check_digit,
            "expiration_date": expiration_date,
            "expiration_date_check_digit": expiration_date_check_digit,
            "cnp_series": cnp_series,
            "cnp_number": cnp_number,
            "final_check_digit": final_check_digit
        }
    return {}

############################################
# 3. FACE DETECTION (MTCNN) & VERIFICATION (DeepFace)
############################################

from facenet_pytorch import MTCNN
from deepface import DeepFace

def detect_and_extract_face_mtcnn(img_path, show_result=False, device='cpu'):
    """
    Detects a face in the given image using MTCNN and extracts it.
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

    # # Optional visualization
    # if show_result:
    #     # Draw bounding box
    #     cv2.rectangle(img_bgr, (x1,y1), (x2,y2), (0,255,0), 2)
    #     # Show original + cropped face
    #     import matplotlib.pyplot as plt
    #     img_bgr_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    #     face_cropped_rgb = cv2.cvtColor(face_cropped_bgr, cv2.COLOR_BGR2RGB)

    #     fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,4))
    #     ax1.imshow(img_bgr_rgb)
    #     ax1.set_title("Face Detected")
    #     ax1.axis("off")

    #     ax2.imshow(face_cropped_rgb)
    #     ax2.set_title("Cropped Face")
    #     ax2.axis("off")

    #     plt.show()

    return face_cropped_bgr

def verify_faces(face1_bgr, face2_bgr, model_name='Facenet', distance_metric='cosine'):
    """Verifies whether face1 and face2 belong to the same person using DeepFace."""
    if face1_bgr is None or face2_bgr is None:
        print("[ERROR] One of the faces is None.")
        return None

    face1_rgb = cv2.cvtColor(face1_bgr, cv2.COLOR_BGR2RGB)
    face2_rgb = cv2.cvtColor(face2_bgr, cv2.COLOR_BGR2RGB)

    try:
        result = DeepFace.verify(
            img1_path=face1_rgb,
            img2_path=face2_rgb,
            model_name=model_name,
            distance_metric=distance_metric,
            enforce_detection=False
        )
        return result
    except Exception as e:
        print(f"[ERROR] DeepFace verification failed: {e}")
        return None

############################################
# 4. METRIC FUNCTIONS (CER & Verification Stats)
############################################

def character_error_rate(expected, actual):
    """
    Computes character-level Levenshtein distance between the
    expected string and actual string, normalized by length of expected.
    """
    import numpy as np

    if not expected and not actual:
        return 0.0
    if not expected:
        return len(actual)
    if not actual:
        return len(expected)

    m, n = len(expected), len(actual)
    dp = np.zeros((m + 1, n + 1), dtype=int)

    for i in range(m + 1):
        dp[i, 0] = i
    for j in range(n + 1):
        dp[0, j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if expected[i-1] == actual[j-1] else 1
            dp[i, j] = min(
                dp[i-1, j] + 1,      # deletion
                dp[i, j-1] + 1,      # insertion
                dp[i-1, j-1] + cost  # substitution
            )

    dist = dp[m, n]
    return dist / float(len(expected))

class VerificationStats:
    """Tracks face verification outcomes to compute FAR and FRR."""
    def __init__(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def update(self, same_person_gt, verified):
        if same_person_gt and verified:
            self.tp += 1
        elif same_person_gt and not verified:
            self.fn += 1
        elif not same_person_gt and verified:
            self.fp += 1
        else:
            self.tn += 1

    def far(self):
        # False Acceptance Rate = FP / (FP + TN)
        denom = self.fp + self.tn
        return self.fp / denom if denom else 0.0

    def frr(self):
        # False Rejection Rate = FN / (TP + FN)
        denom = self.tp + self.fn
        return self.fn / denom if denom else 0.0

    def summary(self):
        return {
            "TP": self.tp,
            "TN": self.tn,
            "FP": self.fp,
            "FN": self.fn,
            "FAR": self.far(),
            "FRR": self.frr()
        }

############################################
# 5. MAIN TESTING FUNCTION
############################################
def run_kpi_tests(test_data, output_json_path="kpi_results.json"):
    """
    Iterates over test_data, performs OCR, MRZ parsing, face detection, face verification,
    and collects metrics (CER for OCR fields, FAR/FRR for verification).
    Stores results in a JSON file.
    """
    verification_stats = VerificationStats()
    results = []

    for i, sample in enumerate(test_data, start=1):
        print(f"\n=== Processing Sample {i} ===")
        print(f"ID Image: {sample['id_image']}")
        print(f"Selfie Image: {sample['selfie_image']}")
        
        # Ground truth data
        expected_ocr = sample["expected_ocr_fields"]
        same_person_gt = sample["same_person"]

        # OCR with PaddleOCR
        ocr_result = ocr.ocr(sample["id_image"], cls=True)
        lines = [(line[0], line[1][0], line[1][1]) for line in ocr_result[0]]
        filtered_lines = [(bbox, text, conf) for (bbox, text, conf) in lines if text not in labels_to_exclude]
        filtered_text = [text for (_, text, _) in filtered_lines]

        # Parse MRZ
        mrz_candidates = [t for t in filtered_text if is_mrz_line(t)]
        mrz_data = parse_mrz(mrz_candidates)

        recognized_first_name = mrz_data.get("first_name", "")
        recognized_last_name = mrz_data.get("last_name", "")
        recognized_birth_date = mrz_data.get("birth_date", "")
        recognized_cnp = mrz_data.get("CNP", "")

        # Compute CER for each field
        first_name_cer = character_error_rate(expected_ocr.get("first_name", ""), recognized_first_name)
        last_name_cer = character_error_rate(expected_ocr.get("last_name", ""), recognized_last_name)
        birth_date_cer = character_error_rate(expected_ocr.get("birth_date", ""), recognized_birth_date)
        cnp_cer = character_error_rate(expected_ocr.get("CNP", ""), recognized_cnp)

        # Face Detection on ID & Selfie
        face_id = detect_and_extract_face_mtcnn(sample["id_image"], show_result=False)
        face_selfie = detect_and_extract_face_mtcnn(sample["selfie_image"], show_result=False)

        # Face Verification
        verified = False
        verification_result = None
        if face_id is not None and face_selfie is not None:
            verification_result = verify_faces(face_id, face_selfie, model_name='Facenet', distance_metric='cosine')
            if verification_result and 'verified' in verification_result:
                verified = verification_result['verified']

        # Update stats
        verification_stats.update(same_person_gt, verified)

        # Collect record
        record = {
            "sample_index": i,
            "id_image": sample["id_image"],
            "selfie_image": sample["selfie_image"],
            "ground_truth_ocr": expected_ocr,
            "recognized_ocr": {
                "first_name": recognized_first_name,
                "last_name": recognized_last_name,
                "birth_date": recognized_birth_date,
                "CNP": recognized_cnp
            },
            "ocr_cer": {
                "first_name": first_name_cer,
                "last_name": last_name_cer,
                "birth_date": birth_date_cer,
                "CNP": cnp_cer
            },
            "face_verification": {
                "same_person_ground_truth": same_person_gt,
                "verification_result": verification_result,
                "verified": verified
            }
        }
        results.append(record)

    # Summarize verification stats
    summary_stats = verification_stats.summary()
    print("\n===== Face Verification Aggregate Results =====")
    print(f"TP: {summary_stats['TP']}, TN: {summary_stats['TN']}, FP: {summary_stats['FP']}, FN: {summary_stats['FN']}")
    print(f"FAR: {summary_stats['FAR']:.3f}, FRR: {summary_stats['FRR']:.3f}")

    all_data = {
        "samples": results,
        "verification_stats": summary_stats
    }

    # Save to JSON
    with open(output_json_path, "w", encoding='utf-8') as f:
        json.dump(all_data, f, indent=2)

    return all_data

############################################
# 6. MAIN EXECUTION
############################################
if __name__ == "__main__":
    # Make sure test_images exist and match test_data entries
    # Then run the KPI tests and produce a JSON output
    kpi_results = run_kpi_tests(test_data, output_json_path="kpi_results.json")

    print("\n=== KPI RESULTS (In-Memory) ===")
    print(json.dumps(kpi_results, indent=2))
    print("\nFinished KPI Testing.")
