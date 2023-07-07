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

    def load_model(self):
        '''
        Load YoloV5 model from pytorch hub.
        '''
        model = torch.hub.load('ultralytics/yolov5',
                               'yolov5s',
                               pretrained=True,
                               device=self.device)
