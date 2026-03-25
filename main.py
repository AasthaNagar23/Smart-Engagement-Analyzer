import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

cap = cv2.VideoCapture(0)

def get_eye_ratio(landmarks, eye_points, w, h):
    points = [(int(landmarks[p].x * w), int(landmarks[p].y * h)) for p in eye_points]
    
    vertical = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    horizontal = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
    
    return vertical / horizontal

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    state = "No Face"

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            left_ear = get_eye_ratio(landmarks, LEFT_EYE, w, h)
            right_ear = get_eye_ratio(landmarks, RIGHT_EYE, w, h)

            ear = (left_ear + right_ear) / 2

            nose = landmarks[1]
            nose_x = int(nose.x * w)

            if ear < 0.20:
                state = "Drowsy"
            elif nose_x < w * 0.35 or nose_x > w * 0.65:
                state = "Distracted"
            else:
                state = "Attentive"

    cv2.putText(frame, state, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Engagement Analyzer", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()