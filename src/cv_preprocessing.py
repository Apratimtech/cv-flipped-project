import cv2
import os
import numpy as np

# ---- CONFIG ----
IMAGE_PATH = r"C:\Users\user\OneDrive\Desktop\cv-flipped-project\dataset\chest_xray\test\NORMAL\IM-0001-0001.jpeg"
OUTPUT_DIR = r"C:\Users\user\OneDrive\Desktop\cv-flipped-project\output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- READ IMAGE ----
img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Image not found—check the path again.")

cv2.imwrite(os.path.join(OUTPUT_DIR, "1_original.png"), img)

# ---- 1. ENHANCEMENT (CLAHE) ----
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
enhanced = clahe.apply(img)
cv2.imwrite(os.path.join(OUTPUT_DIR, "2_enhanced_CLAHE.png"), enhanced)

# ---- 2. SEGMENTATION (SIMPLE THRESHOLD) ----
_, segmented = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
cv2.imwrite(os.path.join(OUTPUT_DIR, "3_segmented_Thresh.png"), segmented)

# ---- 3. EDGE DETECTION (CANNY) ----
edges = cv2.Canny(img, 120, 200)
cv2.imwrite(os.path.join(OUTPUT_DIR, "4_edges_canny.png"), edges)

print("CV preprocessing completed! Check the 'output' folder.")
