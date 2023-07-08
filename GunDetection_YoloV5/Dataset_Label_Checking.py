import cv2
import glob

# Path of the dataset
DATASET = 'D:/Deep_Learning/Algorithm/YOLOV5/export/'
# Image path
images_path = glob.glob(DATASET + 'images/*.jpg')
# Label path
labels_path = glob.glob(DATASET + 'labels/*.txt')

print(images_path[0])
print(labels_path[0])

if len(images_path) == len(labels_path):
    for image_path in images_path:
        file_name = image_path.split('/')[-1].split('.jpg')[0].split('\\')[-1]
        print('Image Name: ', file_name + '.jpg')
        print('Text File: ', file_name + '.txt')
        # Load the image
        image = cv2.imread(image_path)
        # Get image shape
        img_height, img_width = image.shape[0], image.shape[1]

        # Read the text file
        with open(DATASET + 'labels/' + file_name + '.txt') as f:
            data = f.readlines()
            for co_or in data:
                # Grab labels coordinates
                _, x, y, width, height = co_or.split(' ')
                # Convert from string to float
                x, y, width, height = int(float(x) * img_width), int(
                    float(y) * img_height), int(float(width) * img_width), int(
                        float(height) * img_height)
                # Getting x1, y1, x2, y2
                co_x1, co_y1, co_x2, co_y2 = int(x - (width / 2)), int(y - (
                    height / 2)), int(x + (width / 2)), int(y + (height / 2))
                # Creating bounding boxes
                cv2.rectangle(image, (co_x1, co_y1), (co_x2, co_y2),
                              (0, 255, 0), 3)
        # Show image with bounding box
        cv2.imwrite('Image.jpg', image)
        break