import cv2

# This CascadeClassifier is for object detection
# Here we are using front face detection algorithm
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     'haarcascade_frontalface_default.xml')
# Here we are using smile detection
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                      'haarcascade_smile.xml')

# VideoCapture funtion to record video from mobile phone
video = cv2.VideoCapture(0)
address = 'http://192.168.0.104:8080/video'
# Let's open the video
video.open(address)

while True:
    # This function returns two variables
    # check = boolean variable, indicates whether the frame successfully read or not
    # frame = actual frame that was read from video source
    check, frame = video.read()
    if not check:
        print("No frame read from video")
        break
    # First parameter is input image
    # Second parameter is how zoomed my image is, we must have to set a value where the output don't capture any noise.
    # If we set it to 1 or close to 1 then it can capture noise and it takes longer time to process
    # Third parameter is minimum neighbour pixels to calculate the face
    face = face_cascade.detectMultiScale(frame,
                                         scaleFactor=1.5,
                                         minNeighbors=5)
    # Output = [[x, y, h, w]]

    # Inside the face we have to calculate the smile
    for x_f, y_f, w_f, h_f in face:
        # Mark the face
        cv2.rectangle(frame, (x_f, y_f), (x_f + w_f, y_f + h_f), (0, 255, 0),
                      3)
        crop_face = frame[y_f:y_f + h_f, x_f:x_f + w_f]
        # Inside face we are trying to detect smile
        smile = smile_cascade.detectMultiScale(crop_face,
                                               scaleFactor=1.5,
                                               minNeighbors=20)
        for x_s, y_s, w_s, h_s in smile:
            # Mark the smile
            cv2.rectangle(frame, (x_f + x_s, y_f + y_s),
                          (x_f + x_s + w_s, y_f + y_s + h_s), (0, 0, 255), 3)

    # Show the video
    cv2.imshow('Output', frame)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows