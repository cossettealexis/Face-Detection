import cv2
import dlib

capture = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = capture.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(frame)
    cv2.imshow("Frame", frame)

    for face in faces:
        landmarks = predictor(gray_frame, face)

        top_eyes = landmarks.part()

    key = cv2.waitKey(1)
    if key == 27:
        break
