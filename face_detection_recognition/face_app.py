import cv2
import urllib.request
import os

# Haar Cascade file
url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
file_name = "haarcascade_frontalface_default.xml"

# Download automatically if file does not exist
if not os.path.exists(file_name):
    print("Face detector download ho raha hai...")
    urllib.request.urlretrieve(url, file_name)
    print("Download complete!")

# Load face detector
face_cascade = cv2.CascadeClassifier(file_name)

# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera open nahi ho raha!")
    exit()

print("Camera started!")
print("Q press karke camera band karo.")

while True:

    ret, frame = camera.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Face Detected",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()