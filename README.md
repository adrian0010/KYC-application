# KYC-application

This project is a Know Your Customer (KYC) application that uses various Optical Character Recognition (OCR) tools to extract and process text from images of identification documents.

## Project Structure

KYC-application
├── FaceRecognition
│   └── face_recognition.py     # Face recognition related code
├── images
│   └── CI_Specimen.jpeg        # Sample image for OCR
├── OCR
│   ├── EasyOCR.py              # EasyOCR script
│   ├── Keras-OCR.py            # Keras-OCR script
│   ├── PaddleOCR.py            # PaddleOCR script
│   └── Pytesseract.py          # Pytesseract script
├── output                      # Directory for OCR output results
├── .gitignore                  # Git ignore file
├── README.md                   # Project documentation
└── requirements.txt            # Project dependencies

## OCR Tools

The project uses the following OCR tools:

1. **EasyOCR**: A Python library for OCR that supports multiple languages.
2. **Keras-OCR**: An OCR library built on Keras and TensorFlow.
3. **PaddleOCR**: An OCR library developed by PaddlePaddle.
4. **Pytesseract**: A Python wrapper for Google's Tesseract-OCR Engine.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/KYC-application.git
    cd KYC-application
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### EasyOCR

To run the EasyOCR script:
```sh
python OCR/EasyOCR.py