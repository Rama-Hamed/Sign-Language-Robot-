import cv2
import mediapipe as mp
import pickle
import numpy as np
import socket
from collections import Counter
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

COMMAND_DESC = {
    "A": "Forward",
    "B": "Backward",
    "C": "Turn Right",
    "D": "Turn Left",
    "E": "Stop",
}

client_socket = None
try:
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(("localhost", 12345))
    client_socket.setblocking(False)
except:
    client_socket = None


def send_command(cmd):
    if client_socket is None:
        return
    try:
        client_socket.sendall(cmd.encode("utf-8"))
    except:
        pass


STABLE_FRAMES = 10
prediction_buffer = []
last_command = None
stable_command = None

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    results = detector.detect(mp_image)

    cv2.rectangle(frame, (0, 0), (500, 120), (45, 45, 45), -1)
    cv2.line(frame, (0, 120), (500, 120), (0, 255, 0), 2)

    if results.hand_landmarks:
        for hand in results.hand_landmarks:
            data_aux = []

            x_values = [lm.x for lm in hand]
            y_values = [lm.y for lm in hand]

            min_x = min(x_values)
            min_y = min(y_values)

            for lm in hand:
                data_aux.extend([lm.x - min_x, lm.y - min_y, lm.z])

            prediction = model.predict([data_aux])[0].strip().upper()

            # تعديل مؤقت لمشكلة C
            # إذا لاحظتي أن C يطلع بحرف ثاني، بدلي O بالحرف اللي فعلاً يطلع
            if prediction == "O":
                prediction = "C"

            prediction_buffer.append(prediction)

            if len(prediction_buffer) > STABLE_FRAMES:
                prediction_buffer.pop(0)

            most_common = Counter(prediction_buffer).most_common(1)[0]
            best_pred = most_common[0]
            best_count = most_common[1]

            if best_count >= STABLE_FRAMES * 0.7:
                stable_command = best_pred

                if stable_command != last_command:
                    send_command(stable_command)
                    last_command = stable_command

            for connection in HAND_CONNECTIONS:
                start_pt = (
                    int(hand[connection[0]].x * w),
                    int(hand[connection[0]].y * h)
                )
                end_pt = (
                    int(hand[connection[1]].x * w),
                    int(hand[connection[1]].y * h)
                )
                cv2.line(frame, start_pt, end_pt, (0, 255, 0), 2)

            for lm in hand:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), 1)

            raw_desc = COMMAND_DESC.get(prediction, "...")
            stable_desc = COMMAND_DESC.get(stable_command, "...")

            cv2.putText(
                frame,
                f"Raw: {prediction} ({raw_desc})",
                (20, 45),
                cv2.FONT_HERSHEY_DUPLEX,
                0.9,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Stable: {stable_command} ({stable_desc})",
                (20, 90),
                cv2.FONT_HERSHEY_DUPLEX,
                0.9,
                (255, 255, 255),
                2
            )

    else:
        prediction_buffer.clear()
        stable_command = None

        if last_command != "E":
            send_command("E")
            last_command = "E"

    cv2.imshow("Sign Language Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

send_command("E")

if client_socket:
    client_socket.close()

cap.release()
cv2.destroyAllWindows()
