import cv2
import numpy as np

def find_rectangles(frame, min_area=.005, apply_threshold=True):

    if apply_threshold:
        # Convert to gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Binarize
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    else:
        thresh = frame.astype(np.uint8)

    # Find contours
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    candidates = []
    minimum_area = min_area * frame.size
    for c in contours:

        # Eliminate small patches
        if cv2.contourArea(c) < minimum_area:
            continue

        # Find best fitting box
        b = cv2.minAreaRect(c)
        bp = cv2.boxPoints(b)[:, np.newaxis, :].astype('int32')

        # Compare with original shape
        out = np.zeros(frame.shape[:-1], dtype='uint8')
        cv2.drawContours(out, [bp], -1, 1, thickness=cv2.FILLED)
        v1 = out.sum()
        cv2.drawContours(out, [c], -1, 0, thickness=cv2.FILLED)
        v2 = out.sum()
        q = 1 - v2 / (v1 + 1e-5)

        # Compute aspect ratio
        ar = max(b[1]) / min(b[1])
        candidates.append((q, ar, b, bp, c))

    return sorted(candidates)
