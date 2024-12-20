import easyocr
import re
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Initialize the easyocr reader - Romanian language
reader = easyocr.Reader(['ro'])

# Define labels to exclude
labels_to_exclude = [
    "ROUMANIE",
    "ROIIFsiJih",
    "ROMANIA",
    "CARTE",
    "CARTE",
    "DE",
    "IDENTITATE",
    "identity",
    "D'IDENTITE",
    "SERIA",
    "NR",
    "Card",
    "CNP", 
    "Nume/Nom/Last name", 
    "Prenume/Prenom/First name", 
    "CetătenielNationalitelNationality", 
    "SexlSexelSex", 
    "ROU",
    "Loc nastere/Lieu de naissancelPlace of birth", 
    "Jud",
    "Jud.",
    "SPCLEP",
    "Domiciliu/Adresse/Address", 
    "717",
    "evo",
    "Emisa delDelivree parllssued by", 
    "ValabilitatelValidite/ Validity"
]

# Function to check if a text line is an MRZ line
def is_mrz_line(text):
    return bool(re.match(r'^[A-Z0-9<]{30,44}$', text))

# Function to parse MRZ lines for Romanian ID card
def parse_mrz(mrz_lines):
    if len(mrz_lines) == 2 and all(len(line) == 36 for line in mrz_lines):  # Romanian ID card format
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
        # result is a list of (bbox, text, confidence)
        result = reader.readtext(image_path)

        # Extract and filter text, excluding specified labels
        filtered_results = [(bbox, text, conf) for (bbox, text, conf) in result if text not in labels_to_exclude]

        # Extract just the texts for MRZ checking
        filtered_text = [res[1] for res in filtered_results]

        # Extract MRZ lines
        mrz_lines = [text for text in filtered_text if is_mrz_line(text)]

        # Parse MRZ lines
        mrz_data = parse_mrz(mrz_lines)

        # Print filtered text and MRZ data in the console
        print(f"Filtered Extracted Text from {image_file}:")
        for t in filtered_text:
            print(t)
        
        print(f"\nParsed MRZ Data from {image_file}:")
        for key, value in mrz_data.items():
            print(f"{key}: {value}")
        print("\n" + "="*50 + "\n")
        
        # Visualization Setup
        fig, (ax_img, ax_text) = plt.subplots(1, 2, figsize=(15,10))

        # Draw bounding boxes and text on the image
        img = cv2.imread(image_path)
        for i, (bbox, text, confidence) in enumerate(filtered_results):
            pts = np.array([bbox], dtype=np.int32)
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)
            top_left_x = int(bbox[0][0])
            top_left_y = int(bbox[0][1]) - 10
            cv2.putText(img, text, (top_left_x, top_left_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)

        # Convert to RGB for matplotlib visualization
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax_img.imshow(img_rgb)
        ax_img.axis('off')
        ax_img.set_title("Detected Text on ID Image")

        ax_text.axis('off')
        ax_text.set_title("Recognized Text Lines & Confidence", fontsize=14)
        
        # Starting vertical position
        y_pos = 1.0
        line_height = 0.04  # adjust as needed
        
        for i, (bbox, text, conf) in enumerate(filtered_results, start=1):
            ax_text.text(0.01, y_pos, f"{i}: {text}  {conf:.3f}", fontsize=12, transform=ax_text.transAxes)
            y_pos -= line_height

        # Adjust layout
        plt.tight_layout()

        # Save the figure (image + text listing) to the output folder
        output_folder = 'output'
        os.makedirs(output_folder, exist_ok=True)
        output_image_path = os.path.join(output_folder, f"EasyOCR_output_{os.path.basename(image_file)}")
        plt.savefig(output_image_path, dpi=200)
        
        # Show the figure
        plt.show()

        # Prepare the text output for this specific image
        output_text = []
        output_text.append(f"Filtered Extracted Text from {os.path.basename(image_file)}:")
        
        # Include each filtered line with its confidence
        for i, (bbox, text, conf) in enumerate(filtered_results, start=1):
            output_text.append(f"{i}: {text}  {conf:.3f}")

        output_text.append(f"\nParsed MRZ Data from {os.path.basename(image_file)}:")
        for key, value in mrz_data.items():
            output_text.append(f"{key}: {value}")

        output_text.append("\n" + "="*50 + "\n")

        # Get the current date and time
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")

        # Create a unique text file name for each image
        text_output_file = f"EasyOCR_{os.path.splitext(os.path.basename(image_file))[0]}_{timestamp}.txt"
        output_text_path = os.path.join(output_folder, text_output_file)

        # Write the output text to a file
        with open(output_text_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(output_text))
