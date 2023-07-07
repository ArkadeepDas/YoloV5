# Here is this Script we are going to detect object using OpenCV and YoloV5
# Importing Requirements
import torch
import numpy as np
import cv2
from time import time


# Let's create Object detection class
class ObjectDetection:
    '''
    Class implements YoloV5 model to make prediction on video using OpenCV
    '''

    def __init__(self, url, out_file):
        '''
        Initialize the class with url and output file.
        : parameter url: Has to be as video URL, on which the prediction will happen.
        : parameter out_file: Our output file name, where we save our output.
        '''
        self.url = url
        self.out_file = out_file
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('\nDevice Used here is: ', self.device)

    # Get video frames from URL
    def get_video_from_url(self):
        '''
        Load video from link
        '''
        video = cv2.VideoCapture(0)
        video.open(self.url)
        check, frame = video.read()
        if not check:
            return 0
        return frame

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
    def score_frame(self, frame):
        '''
        Take single frame as a input and predict is using YoloV5
        : parameter frame: single frame from out video
        : return: lebels and coordinates of objects detected by YoloV5
        '''
        self.model.to(self.device)
        frame = [frame]
        model = self.load_model()
        results = model(frame)

        # Labels and Coordinates
        output = results.xyxyn[0]
        return output

    # Let's plot the box
    def plot_boxes(self, results, frame):
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
                cv2.putText(frame, data[6], (x2, y2), cv2.FONT_HERSHEY_SIMPLEX,
                            (0, 0, 255))

        return frame