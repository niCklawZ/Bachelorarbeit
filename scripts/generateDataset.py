"""
Author: Nick Kottek
Date: 14.05.2024
"""

import cv2
import os
from cvzoneHandTrackingModified import HandDetector

# For which letter should the images be collected?
letter = "ß"

# Set output directory
dataset_directory = os.getcwd() + "\..\dataset\\" + letter

dirExists = os.path.exists(dataset_directory)
if not dirExists:
    os.makedirs(dataset_directory)

os.chdir(dataset_directory)

# At which ID should be started?
img_id = 1

# At which ID should be ended?
max_img_id = 50

cam_id = 0
capture = cv2.VideoCapture(cam_id)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

detector = HandDetector(detectionCon=0.8, maxHands=1)

cv2.namedWindow("image")

while True:

    ret, img = capture.read()

    if ret:

        _, img_hand, bbox = detector.findHands(img, False)

        if bbox is not None:

            height, width, _ = img_hand.shape
            size = max(bbox[2], bbox[3]) + 140  # bounding box is structured like this: (x, y, width, height)
            center_x = int(bbox[0] + bbox[2] / 2)
            center_y = int(bbox[1] + bbox[3] / 2)
            x1 = int(center_x - size / 2)
            y1 = int(center_y - size / 2)
            x2 = int(center_x + size / 2)
            y2 = int(center_y + size / 2)
            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 > width:
                x2 = width
            if y2 > height:
                y2 = height

            # Reduce image boundaries to bounding box
            img_hand = img_hand[y1:y2, x1:x2]

            # Image Processing - Source: https://stackoverflow.com/a/60761202
            # Extract red color channel (because the hand color is more red than the background).
            gray = img_hand[:, :, 2]
            # Apply binary threshold using automatically selected threshold (using cv2.THRESH_OTSU parameter).
            ret, hand_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Use "opening" morphological operation for clearing some small dots (noise)
            hand_mask = cv2.morphologyEx(hand_mask, cv2.MORPH_OPEN,
                                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

            # Use "closing" morphological operation for closing small gaps
            hand_mask = cv2.morphologyEx(hand_mask, cv2.MORPH_CLOSE,
                                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

            # Apply mask to image
            img_hand = cv2.bitwise_and(img_hand, img_hand, mask=hand_mask)

            # convert to gray
            img_hand = cv2.cvtColor(img_hand, cv2.COLOR_RGB2GRAY)

            # resize to static dimensions
            img_hand = cv2.resize(img_hand, (500, 500))

            cv2.imshow("image", img_hand)
        else:
            cv2.imshow("image", img)

        # ESC to close window
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        # Space to save image to dataset
        if key == 32:
            cv2.imwrite(letter + format(img_id, '03d') + ".jpg", img_hand)
            if img_id == max_img_id:
                break
            img_id += 1
    else:
        break

cv2.destroyAllWindows()
