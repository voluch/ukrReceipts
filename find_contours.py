# import required libraries
import cv2

# read the input image
img = cv2.imread('polygon.png')

# convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# apply thresholding to convert the grayscale image to a binary image
ret, thresh = cv2.threshold(gray, 50, 255, 0)

# find the contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print("Number of contours detected:", len(contours))
for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.001 * cv2.arcLength(cnt, True), True)
    (x, y) = cnt[0, 0]

    if len(approx) >= 5:
        img = cv2.drawContours(img, [approx], -1, (0, 255, 255), 3)
        cv2.putText(img, 'Polygon', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
cv2.imshow("Polygon", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


############ ChtGPT code
import numpy as np
from skimage.measure import find_contours
from shapely.geometry import Polygon


def find_polygons(mask):
    """Finds polygons on a binary mask.

    Args:
        mask (ndarray): A binary mask of shape (height, width).

    Returns:
        A list of shapely Polygon objects representing the polygons found in the mask.
    """
    # Find contours on the mask.
    contours = find_contours(mask, 0.5, fully_connected='high', positive_orientation='high')

    # Convert the contours to polygons.
    polygons = []
    for contour in contours:
        # Reverse the y-axis of the contour coordinates.
        contour[:, 0] = contour[:, 0][::-1]
        contour[:, 1] = mask.shape[0] - contour[:, 1] - 1

        # Convert the contour to a Polygon object.
        polygon = Polygon(contour)

        # Add the polygon to the list.
        polygons.append(polygon)

    return polygons


import matplotlib.pyplot as plt

# Generate a random binary mask.
mask = np.random.randint(0, 2, size=(100, 100)).astype(np.bool)

# Find the polygons on the mask.
polygons = find_polygons(mask)

# Plot the mask and polygons.
fig, ax = plt.subplots()
ax.imshow(mask, cmap='gray')
for polygon in polygons:
    ax.plot(*polygon.exterior.xy, color='red', linewidth=2)
plt.show()
