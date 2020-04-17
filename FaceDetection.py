import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

capture = cv2.VideoCapture(0)


while True
    ret, image = capture.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for(x, y, w, h) in faces:
        cv2.rectangle(image, (x,y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]

    cv2.imshow('image', image)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

capture.release()
cv2.destroyAllWindows()