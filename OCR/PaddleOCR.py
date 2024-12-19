import logging
from paddleocr import PaddleOCR, draw_ocr
import cv2
import matplotlib.pyplot as plt
import re
import os

# Set logging level to ERROR to suppress debug logs
logging.getLogger('ppocr').setLevel(logging.ERROR)

# Initialize the OCR. Set use_angle_cls=True if text orientation matters.
ocr = PaddleOCR(use_angle_cls=True, lang='ro')

# Define labels to exclude
#labels_to_exclude = []
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
    "Romanä\ROU",
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
        birth_date = line2[13:19]
        birth_date_check_digit = line2[19]
        sex = line2[20]
        expiration_date = line2[21:27]
        expiration_date_check_digit = line2[27]
        cnp_series = line2[28]
        cnp_number = line2[29:35]
        final_check_digit = line2[35]

        return {
            "document_type": document_type,
            "issuing_country": issuing_country,
            "last_name": last_name,
            "first_name": first_name,
            "id_number": id_number,
            "id_number_check_digit": id_number_check_digit,
            "nationality": nationality,
            "birth_date": birth_date,
            "birth_date_check_digit": birth_date_check_digit,
            "sex": sex,
            "expiration_date": expiration_date,
            "expiration_date_check_digit": expiration_date_check_digit,
            "cnp_series": cnp_series,
            "cnp_number": cnp_number,
            "final_check_digit": final_check_digit
        }
    return {}

# Process each image in the images folder
image_folder = 'images'
for image_file in os.listdir(image_folder):
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_folder, image_file)

        # Perform OCR on the image
        # PaddleOCR's result format: [[ [box], (text, confidence) ], ...]
        result = ocr.ocr(image_path, cls=True)

        # Extract recognized text lines
        # result[0] contains a list of lines: [ [box, (text, confidence)], ...]
        txts = [line[1][0] for line in result[0]]
        confidences = [line[1][1] for line in result[0]]

        # Filter out unwanted labels
        filtered_text = [t for t in txts if t not in labels_to_exclude]

        # Extract MRZ lines
        mrz_lines = [text for text in filtered_text if is_mrz_line(text)]

        # Parse MRZ lines
        mrz_data = parse_mrz(mrz_lines)

        print(f"Filtered Extracted Text from {image_file}:")
        for text in filtered_text:
            print(text)

        print(f"\nParsed MRZ Data from {image_file}:")
        for key, value in mrz_data.items():
            print(f"{key}: {value}")
        print("\n" + "="*50 + "\n")

        # Visualize the OCR results:
        boxes = [line[0] for line in result[0]]
        img = cv2.imread(image_path)
        font_path = 'C:/Windows/Fonts/arial.ttf'  # Update font path if necessary
        img_with_boxes = draw_ocr(img, boxes, txts, confidences, font_path=font_path)
        img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
        
        # Save the image with OCR results to the output folder
        output_folder = 'output'
        os.makedirs(output_folder, exist_ok=True)
        output_image_path = os.path.join(output_folder, f"PaddleOCR_output_{os.path.basename(image_file)}")
        cv2.imwrite(output_image_path, cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))
        
        plt.figure(figsize=(10,10))
        plt.imshow(img_with_boxes)
        plt.axis('off')
        plt.show()