import cv2
import numpy as np
from sklearn.cluster import MeanShift


def correct_boxes_pbc(image, boxes, fast_threshold=20, bandwidth=5):
    """Apply Prediction Box Correction (PBC) on YOLO boxes.

    Args:
        image (np.array): HxWx3 uint8 image
        boxes (np.array): shape (N, 4) in [x1, y1, x2, y2] format
        fast_threshold (int): threshold for FAST corner detection
        bandwidth (int): MeanShift bandwidth

    Returns:
        corrected_boxes (np.array): shape (N, 4) refined
    """
    corrected_boxes = []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for box in boxes:
        x1, y1, x2, y2 = box.astype(int)
        crop = gray[y1:y2, x1:x2]

        # 1. Histogram equalization
        crop_eq = cv2.equalizeHist(crop)

        # 2. FAST feature detection
        fast = cv2.FastFeatureDetector_create(threshold=fast_threshold, nonmaxSuppression=True)
        keypoints = fast.detect(crop_eq, None)

        if len(keypoints) == 0:
            corrected_boxes.append(box)
            continue

        pts = np.array([kp.pt for kp in keypoints])  # Nx2

        h, w = crop.shape

        # 3. Divide into four correction regions
        regions = {
            "left": pts[pts[:, 0] < w / 2],
            "right": pts[pts[:, 0] >= w / 2],
            "up": pts[pts[:, 1] < h / 2],
            "down": pts[pts[:, 1] >= h / 2],
        }

        new_x1 = int(x1 + np.mean(regions["left"][:, 0]) if len(regions["left"]) > 0 else x1)
        new_x2 = int(x1 + np.mean(regions["right"][:, 0]) if len(regions["right"]) > 0 else x2)
        new_y1 = int(y1 + np.mean(regions["up"][:, 1]) if len(regions["up"]) > 0 else y1)
        new_y2 = int(y1 + np.mean(regions["down"][:, 1]) if len(regions["down"]) > 0 else y2)

        # 4. Optionally: apply MeanShift for more robust clustering
        for key, pts_region in regions.items():
            if len(pts_region) > 1:
                ms = MeanShift(bandwidth=bandwidth)
                ms.fit(pts_region)
                densest = ms.cluster_centers_[0]
                if key == "left":
                    new_x1 = int(x1 + densest[0])
                elif key == "right":
                    new_x2 = int(x1 + densest[0])
                elif key == "up":
                    new_y1 = int(y1 + densest[1])
                elif key == "down":
                    new_y2 = int(y1 + densest[1])

        corrected_boxes.append([new_x1, new_y1, new_x2, new_y2])

    return np.array(corrected_boxes)
