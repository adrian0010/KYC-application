import cv2
import os
import matplotlib.pyplot as plt

def process_image(img_path, show_steps=False):
    # Read the image with OpenCV
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"[ERROR] Cannot load image: {img_path}")
        return None

    original = img_bgr.copy()

    # Apply denoising
    img_bgr = cv2.fastNlMeansDenoisingColored(img_bgr, None, 10, 10, 7, 21)

    # Improve contrast using CLAHE (on the L-channel if we do LAB color)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l,a,b))
    img_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if show_steps:
        # Show the original vs. processed side by side using matplotlib
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        processed_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(original_rgb)
        plt.title("Original Image")
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(processed_rgb)
        plt.title("Processed Image")
        plt.axis('off')
        plt.show()

    return img_bgr

if __name__ == "__main__":
    img_path = "output/DeepFace_cropped_face4.jpg"

    # Check if the image exists
    if not os.path.exists(img_path):
        print(f"[ERROR] Image not found at {img_path}")
    else:
        processed_image = process_image(img_path, show_steps=True)
        if processed_image is not None:
            print("[INFO] Image processing completed successfully.")
        else:
            print("[ERROR] Image processing failed.")