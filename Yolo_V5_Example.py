import torch

# Importing pretrain Yolov5 model
# Yolov5 have many models. But here we are using small version s
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
