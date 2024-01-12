"""
Author: Nick Kottek
"""

import cv2
import numpy as np
import tensorflow as tf
import queue
import threading
from cvzoneHandTrackingModified import HandDetector

# Set URL of camera video stream to predict. Use the App 'IP Webcam' on Android
video_url = 'http://nick:1234@192.168.178.61:8080/video'

# Import trained model
model = tf.keras.models.load_model('../models/trainedModel-2024-01-12-15-00-03-eval_loss 0.019-eval_acc 0.992-train_time 66.104.keras')
model.summary()

# Define classes to classify
class_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 'sch', 't',
               'u', 'v', 'w', 'x', 'y']

# Create hand detector
detector = HandDetector(detectionCon=0.8, maxHands=1)


# Create a bufferless VideoCapture class. Its purpose is to automatically read the frames coming in from the source
# Doing so allows the prediction to handle the most recent frame. Otherwise, there would be a growing delay in
# prediction when the time to predict is longer than the time for a new frame
# Source: https://stackoverflow.com/a/54755738
class VideoCapture:

    def __init__(self):
        self.cap = cv2.VideoCapture(video_url)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()


# Define method that returns the necessary data for prediction and visualisation
def get_prediction_data(img):
    _, frame, bbox = detector.findHands(img, False)

    if bbox is not None:

        height, width, _ = frame.shape
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
        img_hand = frame[y1:y2, x1:x2]

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
        img_hand_big = cv2.resize(img_hand, (500, 500))  # for display
        img_hand_small = cv2.resize(img_hand, (64, 64))  # for prediction

        # Change type for prediction
        prediction_image = cv2.merge((img_hand_small, img_hand_small, img_hand_small))
        prediction_image = prediction_image.astype(np.float32)
        prediction_image_array = tf.expand_dims(prediction_image, 0)

        return prediction_image_array, frame, img_hand_big
    else:
        return None, frame, None


# Create windows for visualisation
cv2.namedWindow("Original Image")
cv2.namedWindow("Filtered Image")

# Start videocapture
cap = VideoCapture()

# Start live prediction
while True:
    # Get prediction and visualisation data from most recent frame
    pred_array, cam_frame, pred_image = get_prediction_data(cap.read())

    # Predict data if there is any
    if pred_array is not None:
        predictions = model.predict(pred_array)
        scores = tf.nn.softmax(predictions[0])
        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(scores)], 100 * np.max(scores))
        )

        # Print results on screen as visualisation
        font = cv2.FONT_HERSHEY_COMPLEX
        topLeftCornerOfText = (10, 40)
        fontScale = 1.5
        fontColor = (0, 0, 255)
        thickness = 3
        lineType = 2
        cv2.putText(cam_frame, 'Predicted Letter: {} - {:.2f}% confidence'.format(class_names[np.argmax(scores)],
                                                                                  100 * np.max(scores)),
                    topLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)

    # Show original camera frame (with prediction result if there is one)
    if cam_frame is not None:
        cv2.imshow("Original Image", cam_frame)

    # Show input image of prediction in second window
    if pred_image is not None:
        cv2.imshow("Filtered Image", pred_image)

    # ESC to close window and stop prediction
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cv2.destroyAllWindows()
