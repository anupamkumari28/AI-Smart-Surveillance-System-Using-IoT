import cv2
import os

# Folder create
if not os.path.exists("faces"):
    os.makedirs("faces")

# Face model load
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Camera start
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("Starting Camera...")

count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        print("Frame not received")
        continue

    # Show raw camera first (IMPORTANT)
    display_frame = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        face_img = frame[y:y+h, x:x+w]

        filename = "faces/face_" + str(count) + ".jpg"
        cv2.imwrite(filename, face_img)

        count += 1

        cv2.rectangle(display_frame,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow("Face Save System", display_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

