# KYC-application

This project is a Know Your Customer (KYC) application that uses various Optical Character Recognition (OCR) tools to extract and process text from images of identification documents.

## Project Structure

```
KYC-application
├── app
│   └── kyc-app
│       └── templates
│           └── index.html      # HTML template for the KYC application
├── app_v1
│   ├── main.py                 # Main Streamlit application script
│   └── temp                    # Temporary directory for storing images
├── FaceRecognition
│   ├── DeepFace.py             # DeepFace related code
│   ├── face_detection_mtcnn.py # Face detection using MTCNN
│   ├── face_verification_deepface.py # Face verification using DeepFace
│   ├── ImagePreprocess.py      # Image preprocessing code
│   └── main copy.py            # Backup of the main script
├── images                      # Directory for storing images
├── kpi_results.json            # JSON file for storing KPI results
├── OCR
│   ├── EasyOCR.py              # EasyOCR script
│   ├── Keras-OCR.py            # Keras-OCR script
│   ├── PaddleOCR.py            # PaddleOCR script
│   └── Pytesseract.py          # Pytesseract script
├── output                      # Directory for OCR output results
├── test
│   ├── kpi_tester.py           # Script for testing KPIs
│   ├── test.ipynb              # Jupyter notebook for testing
│   └── test_images             # Directory for test images
├── uploads                     # Directory for uploaded files
├── .gitignore                  # Git ignore file
├── Automated_KYC_System_Presentation.pptx # Project presentation
├── ProjectRequirements.txt     # Project requirements
├── README.md                   # Project documentation
└── requirements.txt            # Project dependencies
```

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

### Running the Streamlit Application

To run the Streamlit application:
```sh
streamlit run app_v1/main.py
```

### Running the EasyOCR Script

To run the EasyOCR script:
```sh
python OCR/EasyOCR.py
```

### Running the KPI Tests

To run the KPI tests:
```sh
python test/kpi_tester.py
```

This will process the test images and produce a JSON output with the KPI results.

## Project Requirements

Refer to the `ProjectRequirements.txt` file for detailed project requirements and guidelines.