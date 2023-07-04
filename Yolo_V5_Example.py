import torch
import cv2
# Importing pretrain Yolov5 model
# Yolov5 have many models. But here we are using small version s
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device=0)

# capturing image from link
image = 'https://www.livemint.com/rf/Image-621x414/LiveMint/Period1/2012/10/01/Photos/Road621.jpg'

# Check prediction
results = model(image)
# Print output
print(results)
# Print co-ordinates
print(results.pandas().xyxy[0])
# Show the output image
results.show()
# We can save the image output with bounding box
results.save()