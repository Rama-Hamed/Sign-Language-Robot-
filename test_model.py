import cv2
import mediapipe as mp
import pickle
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

model = pickle.load(open("model.pkl", "rb"))
labels_list = pickle.load(open("labels.pkl", "rb"))

model_path = "hand_landmarker.task"
options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    
    results = detector.detect(mp_image)

    cv2.rectangle(frame, (0, 0), (350, 80), (45, 45, 45), -1)
    cv2.line(frame, (0, 80), (350, 80), (0, 255, 0), 2)

    if results.hand_landmarks:
        for hand in results.hand_landmarks:
            data_aux = []
            x_values = [lm.x for lm in hand]
            y_values = [lm.y for lm in hand]
            
            min_x = min(x_values)
            min_y = min(y_values)

            for lm in hand:
                data_aux.extend([lm.x - min_x, lm.y - min_y, lm.z])

            prediction = model.predict([data_aux])[0]

            for connection in HAND_CONNECTIONS:
                start_pt = (int(hand[connection[0]].x * w), int(hand[connection[0]].y * h))
                end_pt = (int(hand[connection[1]].x * w), int(hand[connection[1]].y * h))
                cv2.line(frame, start_pt, end_pt, (0, 255, 0), 2)

            for lm in hand:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), 1)

            cv2.putText(frame, f"Sign: {prediction}", (20, 55), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.4, (255, 255, 255), 2)

    cv2.imshow("Sign Language Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()