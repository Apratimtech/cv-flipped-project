import cv2
import matplotlib.pyplot as plt

# Load all processed images
orig = cv2.imread("output/1_original.png")
enhanced = cv2.imread("output/2_enhanced_CLAHE.png")
segmented = cv2.imread("output/3_segmented_Thresh.png")   # FIXED FILENAME
edges = cv2.imread("output/4_edges_canny.png")

# Convert BGR to RGB for display
orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
segmented = cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB)
edges = cv2.cvtColor(edges, cv2.COLOR_BGR2RGB)

# Plot all images
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.title("Original X-ray")
plt.imshow(orig)
plt.axis("off")

plt.subplot(2, 2, 2)
plt.title("Enhanced (CLAHE)")
plt.imshow(enhanced)
plt.axis("off")

plt.subplot(2, 2, 3)
plt.title("Segmentation (Threshold)")
plt.imshow(segmented)
plt.axis("off")

plt.subplot(2, 2, 4)
plt.title("Edges (Canny)")
plt.imshow(edges)
plt.axis("off")

plt.tight_layout()
plt.show()
