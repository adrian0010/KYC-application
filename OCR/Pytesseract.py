from PIL import Image
import pytesseract
import cv2
import numpy as np

# If on Windows and Tesseract isn't in your PATH:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the ID image
img_path = 'images/CI_Radu.jpeg'
img = cv2.imread(img_path)

# Resize the image to a standard size
img = cv2.resize(img, (800, 600))

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply adaptive threshold to handle uneven lighting
gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                             cv2.THRESH_BINARY, 11, 2)

# Apply median blur to reduce salt-and-pepper noise
gray = cv2.medianBlur(gray, 3)

# Use morphological operations to clean up small noise or fill small holes
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

# Enhance the image contrast
gray = cv2.equalizeHist(gray)

# Save the preprocessed image for debugging
cv2.imwrite('preprocessed_image.png', gray)

# Convert back to PIL Image for pytesseract
pil_img = Image.fromarray(gray)

# Perform OCR using pytesseract with Romanian language
extracted_text = pytesseract.image_to_string(pil_img, lang='ron')

print("Extracted Text using pytesseract:")
print(extracted_text)