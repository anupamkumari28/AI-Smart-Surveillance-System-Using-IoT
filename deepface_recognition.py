import cv2
from deepface import DeepFace
import datetime
import os
import smtplib
from email.message import EmailMessage

# ================= EMAIL SETTINGS =================
SENDER_EMAIL = "anupam8091@gmail.com"
APP_PASSWORD = "isjccyhzilpefyfv"
RECEIVER_EMAIL = "anupam8091@gmail.com"
# ==================================================

# camera start
cap = cv2.VideoCapture(0)

cap.set(3,640)
cap.set(4,480)

# load face detector
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

known_folder = "known_faces"

print("Starting AI Surveillance System...")

email_sent = False

while True:

    ret, frame = cap.read()

    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x,y,w,h) in faces:

        face_img = frame[y:y+h, x:x+w]

        label = "Unknown"

        try:

            result = DeepFace.find(
                img_path = face_img,
                db_path = known_folder,
                enforce_detection=False
            )

            if len(result[0]) > 0:

                distance = result[0]['distance'][0]

                if distance < 0.40:
                    label = "Known Person"
                    email_sent = False

        except:
            label = "Unknown"


        # color setting
        if label == "Known Person":
            color = (0,255,0)
        else:
            color = (0,0,255)


        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)

        cv2.putText(
            frame,
            label,
            (x,y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2
        )


        # ================= UNKNOWN FACE =================
        if label == "Unknown":

            if not os.path.exists("unknown_faces"):
                os.makedirs("unknown_faces")

            filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"

            filepath = os.path.join("unknown_faces",filename)

            cv2.imwrite(filepath,face_img)


            if not email_sent:

                try:

                    msg = EmailMessage()

                    msg['Subject'] = "🚨 Unknown Person Detected!"

                    msg['From'] = SENDER_EMAIL

                    msg['To'] = RECEIVER_EMAIL

                    msg.set_content("Unknown person detected by AI Surveillance system.")


                    with open(filepath,'rb') as f:

                        file_data = f.read()

                        msg.add_attachment(
                            file_data,
                            maintype='image',
                            subtype='jpeg',
                            filename=filename
                        )


                    with smtplib.SMTP_SSL('smtp.gmail.com',465) as smtp:

                        smtp.login(SENDER_EMAIL,APP_PASSWORD)

                        smtp.send_message(msg)

                    print("Email Alert Sent!")

                    email_sent = True

                except Exception as e:

                    print("Email Error:",e)

    cv2.imshow("AI Surveillance",frame)

    if cv2.waitKey(1) == 27:
        break


cap.release()

cv2.destroyAllWindows()