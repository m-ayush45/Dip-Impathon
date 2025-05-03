import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image and background (ensure both are the same size)
image = cv2.imread('/Users/ayushgautam/Downloads/WPBuggati (2).jpg')  # Replace with your image path
background = cv2.imread('/Users/ayushgautam/Downloads/WPBuggati (2).jpg')  # Replace or simulate

# Resize background to match input (if needed)
if background is not None and background.shape != image.shape:
    background = cv2.resize(background, (image.shape[1], image.shape[0]))

# Convert images to RGB for matplotlib (OpenCV loads as BGR)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
background_rgb = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

# 1. Brightened Image
bright_image = cv2.add(image, np.ones(image.shape, dtype=np.uint8) * 50)
bright_image_rgb = cv2.cvtColor(bright_image, cv2.COLOR_BGR2RGB)

# 2. Image Composition (blending image + background)
composition = cv2.addWeighted(image, 0.7, background, 0.3, 0)
composition_rgb = cv2.cvtColor(composition, cv2.COLOR_BGR2RGB)

# 3. Background Elimination (subtract background from image)
background_elimination = cv2.absdiff(image, background)
background_elimination_rgb = cv2.cvtColor(background_elimination, cv2.COLOR_BGR2RGB)

# 4. Motion Detection (convert subtraction to grayscale mask)
motion_gray = cv2.cvtColor(background_elimination, cv2.COLOR_BGR2GRAY)

# Plotting all in one page
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(bright_image_rgb)
plt.title("Brightened Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(composition_rgb)
plt.title("Image Composition")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(background_elimination_rgb)
plt.title("Background Elimination")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(motion_gray, cmap='gray')
plt.title("Motion Detection")
plt.axis("off")

plt.tight_layout()
plt.show()
