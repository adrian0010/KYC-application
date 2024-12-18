import easyocr
import re

# Initialize the easyocr reader - Romanian language
reader = easyocr.Reader(['ro'])

# Perform OCR on the image
result = reader.readtext('images/CI_Radu.jpeg')

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
"ValabilitatelValidite/ Validity"]

# Extract and filter text, excluding specified labels
filtered_text = []

for bbox, text, confidence in result:
    if text not in labels_to_exclude:
        filtered_text.append(text)
# Function to check if a text line is an MRZ line
def is_mrz_line(text):
    return bool(re.match(r'^[A-Z0-9<]{30,44}$', text))

# Extract MRZ lines
mrz_lines = [text for text in filtered_text if is_mrz_line(text)]

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

# Parse MRZ lines
mrz_data = parse_mrz(mrz_lines)

print("Filtered Extracted Text using EasyOCR:")
for text in filtered_text:
    print(text)

print("\nParsed MRZ Data:")
for key, value in mrz_data.items():
    print(f"{key}: {value}")