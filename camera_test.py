import cv2

camera_index = -1

# auto detect camera
for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.read()[0]:
        camera_index = i
        print("Camera found at index:", i)
        break
    cap.release()

if camera_index == -1:
    print("No camera detected")
    exit()

cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

cap.set(3,640)
cap.set(4,480)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera not working")
        break

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()