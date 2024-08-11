import cv2

# Import Required Libraries
# (cv2, numpy, etc.)
# Here we only need cv2 for this task

# Load Video Stream
# (Capture from Webcam)
video_capture = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not video_capture.isOpened():
    print("Error: Could not open video device.")
    exit()

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video
while True:
    # 1. Capture Frame from Video
    ret, frame = video_capture.read()

    # Check if the frame was captured correctly
    if not ret:
        print("Error: Could not read frame.")
        break

    # 2. Convert Frame to Grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect Faces in Frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw Rectangles around Faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the Frame with Detected Faces
    cv2.imshow('Live Video - Face Detection', frame)

    # Check for Exit Condition
    # (e.g., Press 'q' to Quit)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release Video Stream and Destroy All Windows
video_capture.release()
cv2.destroyAllWindows()

# End
