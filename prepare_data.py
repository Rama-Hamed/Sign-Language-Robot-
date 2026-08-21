import cv2
import mediapipe as mp
import csv
import os
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

model_path = "hand_landmarker.task"  
options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)

DATA_PATH = "dataset"
CSV_FILE = "landmarks_data.csv"

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        header = []
        for i in range(21):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        header.append("label")
        writer.writerow(header)

with open(CSV_FILE, "a", newline="") as f:
    writer = csv.writer(f)

    for label in os.listdir(DATA_PATH):
        class_path = os.path.join(DATA_PATH, label)
        if not os.path.isdir(class_path):
            continue

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            image = cv2.imread(img_path)
            if image is None: continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            results = detector.detect(mp_image)

            if results.hand_landmarks:
                for hand_landmarks in results.hand_landmarks:
                    data = []
                    
                    
                    x_values = [lm.x for lm in hand_landmarks]
                    y_values = [lm.y for lm in hand_landmarks]
                    
                    min_x = min(x_values)
                    min_y = min(y_values)

                    for lm in hand_landmarks:
                        
                        data.extend([lm.x - min_x, lm.y - min_y, lm.z]) 

                    data.append(label)
                    writer.writerow(data)
