import streamlit as st
import os
from paddleocr import PaddleOCR, draw_ocr
import cv2
import re
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from facenet_pytorch import MTCNN
from deepface import DeepFace

# Set Streamlit page configuration
st.set_page_config(page_title="KYC Application", layout="wide")

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='ro')

# Define labels to exclude
labels_to_exclude = [
    "ROMANIA",
    "ROUMANIE",
    "CARTE",
    "CARTE DE IDENTITATE",
    "IDENTITY",
    "D'IDENTITE",
    "SERIA",
    "NR",
    "CARD",
    "CNP",
    "Nume/Nom/Lastname",
    "Prenume/Prenom/Firstname",
    "Cetatenie/Nationalite/Nationality",
    "Sex/Sexe/Sex",
    "Romanä\\ROU",
    "Loc nastere/Lieu de naissance/Placeof birth",
    "Jud",
    "Jud.",
    "SPCLEP",
    "Domiciliu/Adresse/Address",
    "717",
    "evo",
    "Emisäde/Delivree par/lssued by",
    "Valabilitate/Validite/Validity"
]

# Function to check if a text line is an MRZ line
def is_mrz_line(text):
    return bool(re.match(r'^[A-Z0-9<]{30,44}$', text))

# Function to parse MRZ lines for Romanian ID card
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

# Streamlit app
st.title("KYC Application")

# About Section
with st.sidebar:
    st.header("About")
    st.markdown(
        """
        **Know Your Customers (KYC) App**
        
        Project Overview
        The KYC application has essentially three main Computer Vision tasks:
        1. *OCR for Identification Documents (ID) data extraction*:
           - Goal: Extract textual information (name, date of birth, address, etc.) from a Romanian ID card image.
        
        2. *Face Detection and Extraction from ID*:
           - Goal: Locate and extract the face image printed on the ID card.
        
        3. *Face verification between ID Face and Selfie*:
           - Goal: Compare the face from the ID to a live/selfie image to confirm they represent the same person.

        - **Authors**: Toma Radu & Ivan Adrian
        - **Version**: 1.0
        - **Purpose**: Simplify ID processing and automate KYC workflows.
        """
    )
    
# Upload ID image
uploaded_file = st.file_uploader("Upload your ID image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    temp_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    result = ocr.ocr(temp_path, cls=True)

    lines = [(line[0], line[1][0], line[1][1]) for line in result[0]]
    filtered_lines = [(bbox, text, conf) for (bbox, text, conf) in lines if text not in labels_to_exclude]
    filtered_text = [text for (bbox, text, conf) in filtered_lines]

    mrz_lines = [t for t in filtered_text if is_mrz_line(t)]

    mrz_data = parse_mrz(mrz_lines)

    col1, col2 = st.columns(2)

    with col1:
        st.header("Extracted Information")
        fields_to_display = [
            "document_type"
            ,"issuing_country"
            ,"last_name"
            ,"first_name"
            ,"nationality"
            ,"birth_date"
            ,"sex"
            ,"CNP"
            ,"id_number"
            ,"expiration_date"
        ]

        if mrz_data:
            for key, value in mrz_data.items():
                if key in fields_to_display:
                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")
        else:
            st.write("No valid MRZ data found.")

    with col2:
        st.image(temp_path, caption="Uploaded ID Image", use_container_width=True)

    # New row to display the annotated image
    st.markdown("---")
    st.subheader("OCR Results")
    boxes = [line[0] for line in result[0]]
    txts = [line[1][0] for line in result[0]]
    confidences = [line[1][1] for line in result[0]]

    img = cv2.imread(temp_path)
    font_path = 'C:/Windows/Fonts/arial.ttf'  # Update if necessary
    img_with_boxes = draw_ocr(img, boxes, txts, confidences, font_path=font_path)
    img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)

    # Save and display the annotated image
    output_image_path = os.path.join("temp", "annotated_image.jpg")
    cv2.imwrite(output_image_path, cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))
    st.image(img_with_boxes, caption="Extracted Text with Confidence", use_container_width=True)
    
# Face Verification Section
st.markdown("---")
st.subheader("Upload a Selfie")
selfie_file = st.file_uploader("Upload your selfie image", type=["png", "jpg", "jpeg"], key="selfie")

if uploaded_file is not None and selfie_file is not None:
    # Save uploaded files to temporary paths
    id_image_path = os.path.join("temp", uploaded_file.name)
    selfie_image_path = os.path.join("temp", selfie_file.name)
    
    with open(id_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with open(selfie_image_path, "wb") as f:
        f.write(selfie_file.getbuffer())
    
    # Step 1: Detect and extract face from ID
    face_id = detect_and_extract_face_mtcnn(
        img_path=id_image_path,
        output_face_path=None,  # e.g., "cropped_id_face.jpg"
        show_result=True,
        device='cpu'            #'cuda' for GPU
    )

    # Step 2: Detect and extract face from Selfie
    face_selfie = detect_and_extract_face_mtcnn(
        img_path=selfie_image_path,
        output_face_path=None,  # e.g., "cropped_selfie_face.jpg"
        show_result=True,
        device='cpu'            #'cuda' for GPU
    )
    # Display the cropped faces
    if face_id is not None and face_selfie is not None:
        face_id_rgb = cv2.cvtColor(face_id, cv2.COLOR_BGR2RGB)
        face_selfie_rgb = cv2.cvtColor(face_selfie, cv2.COLOR_BGR2RGB)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(face_id_rgb, caption="ID Face", width=200)
        
        with col2:
            st.image(face_selfie_rgb, caption="Selfie Face", width=200)
        
    # Step 3: Verify the faces
    if face_id is not None and face_selfie is not None:
        with st.spinner('Processing...'):
            verification_result = verify_faces(face_id, face_selfie, model_name='Facenet', distance_metric='cosine')
            
        if verification_result is not None:
            if verification_result['verified']:
                st.success("The faces match!")
            else:
                st.error("The faces do NOT match.")
            st.write(f"Distance: {verification_result['distance']:.3f}, Threshold: {verification_result['threshold']:.3f}")
        else:
            st.error("Face verification failed.")
    else:
        st.error("No face detected in one or both images.")
    
# Cleanup temporary files
if os.path.exists("temp"):
    for file in os.listdir("temp"):
        os.remove(os.path.join("temp", file))

# Activate the virtual environment and run the Streamlit app

# deepface_env\Scripts\activate
# streamlit run app_v1/main.py