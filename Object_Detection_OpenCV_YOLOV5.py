# Here is this Script we are going to detect object using OpenCV and YoloV5
# Importing Requirements
import torch
import numpy as np
import cv2
from time import time

# URL for video
URL = 'http://192.168.0.104:8080/video'


# Let's create Object detection class
class ObjectDetection:
    '''
    Class implements YoloV5 model to make prediction on video using OpenCV
    '''

    def __init__(self, url):
        '''
        Initialize the class with url and output file.
        : parameter url: Has to be as video URL, on which the prediction will happen.
        : parameter out_file: Our output file name, where we save our output.
        '''
        self.url = url
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('\nDevice Used here is: ', self.device)

    # Get video frames from URL
    def get_video_from_url(self):
        '''
        Load video from link
        '''
        video = cv2.VideoCapture()
        video.open(self.url)
        check, frame = video.read()

        return check, frame

    # Load model from pytorch
    def load_model(self):
        '''
        Load YoloV5 model from pytorch hub.
        '''
        model = torch.hub.load('ultralytics/yolov5',
                               'yolov5s',
                               pretrained=True)
        return model

    # Takes single frame as a input and gives the output from YoloV5
    def score_frame(self, frame, model):
        '''
        Take single frame as a input and predict is using YoloV5
        : parameter frame: single frame from out video
        : return: lebels and coordinates of objects detected by YoloV5
        '''
        model.to(self.device)
        frame = [frame]
        results = model(frame)

        # Labels and Coordinates
        output = results.xyxyn[0]
        return output

    # Let's plot the box
    def plot_boxes(self, results, frame, model):
        '''
        Take a frame and its results as input and plot the bounding boxs and labels on the frame
        '''
        output = results
        y_shape, x_shape = frame.shape[0], frame.shape[1]

        # Let's plot the output
        for data in output:
            if data[4] >= 0.4:
                x1, y1, x2, y2 = int(data[0] * x_shape), int(
                    data[1] * y_shape), int(data[2] * x_shape), int(data[3] *
                                                                    y_shape)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame, model.names[int(data[5])], (x2, y2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        return frame


# Let's create the object
object_detection = ObjectDetection(URL)

# Let's initialize the model
model = object_detection.load_model()

while True:
    check, frame = object_detection.get_video_from_url()

    # If there is no frame then break
    if not check:
        print("No frame read from video")
        break
    output = object_detection.score_frame(frame=frame, model=model)
    output_frame = object_detection.plot_boxes(results=output,
                                               frame=frame,
                                               model=model)
    # Let's show the
    cv2.imshow('Output', output_frame)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cv2.destroyAllWindows