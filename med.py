import cv2
import numpy as np

# Load the pre-trained face and hand cascade classifiers
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
hand_cascade = cv2.CascadeClassifier('haarcascade_hand.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize a blank frame for tracking the trajectory
trajectory_frame = None

# Define the line thickness (adjust to your preference)
line_thickness = 10

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)  # Green rectangle for faces

    hands = hand_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=20, minSize=(30, 30))

    for (x, y, w, h) in hands:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)  # Red rectangle for hands

        if trajectory_frame is None:
            trajectory_frame = np.zeros_like(frame)

        # Draw the trajectory line on the separate frame
        if trajectory_frame is not None:
            cv2.line(trajectory_frame, (x + w // 2, y + h // 2), (x + w // 2, y + h // 2), (0, 0, 255), line_thickness)

    # Display the frame with rectangles
    cv2.imshow('Face and Hand Detection', frame)

    # Display the frame with the trajectory line
    if trajectory_frame is not None:
        cv2.imshow('Trajectory', trajectory_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()