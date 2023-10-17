import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.3)

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Open a camera capture
cap = cv2.VideoCapture(0)  # 0 corresponds to the default camera (internal camera)

# Create a blank frame for hand trajectory
blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Adjust the dimensions as needed

# Initialize variables for hand trajectory
previous_positions = dict()

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    if not ret:
        continue

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image with MediaPipe Face Detection
    results_face = face_detection.process(rgb_frame)

    if results_face.detections:
        for detection in results_face.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            # Draw a   smaller rectangle for hand recognition with nothing inside
            rect_size = 50  # Adjust the size of the rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            

    # Process the image with MediaPipe Hand Tracking
    results_hands = hands.process(rgb_frame)

    if results_hands.multi_hand_landmarks:
        for hand, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
            for idx, landmark in enumerate(hand_landmarks.landmark):
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Red circle
                if idx == 8:
                    if previous_positions.get(hand, None) is not None:
                        cv2.line(blank_frame, previous_positions[hand], (x, y), (0, 0, 255), 5)
                    
                    previous_positions[hand] = (x, y)
                    
            # Add the current hand position to the trajectory
            hand_points = [(int(hand_landmarks.landmark[8].x * frame.shape[1]), int(hand_landmarks.landmark[8].y * frame.shape[0]))]
    else:
        previous_positions = dict()
        

    # Create a frame for hand trajectory by copying the blank frame
    trajectory_frame = blank_frame.copy()

    # Draw the hand trajectory on the trajectory frame
  
    # Display the face detection frame
    cv2.imshow('Face Detection', frame)

    # Display the hand trajectory frame
    cv2.imshow('Hand Trajectory', trajectory_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

