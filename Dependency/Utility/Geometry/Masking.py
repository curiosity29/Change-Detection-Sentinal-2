import cv2
import numpy as np
def poly_to_mask(mask_shape, contours):
  mask = np.zeros((mask_shape))
  # for contour in contours:
  mask = cv2.fillPoly(mask, pts = contours, color=(1))

  return mask

