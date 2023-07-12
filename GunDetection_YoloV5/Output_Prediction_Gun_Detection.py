import torch
import cv2
import glob

# Dataset path
test_data_path = 'D:/Deep_Learning/Algorithm/YOLOV5/export/test/'
image_paths = glob.glob(test_data_path + '/*.jpg')

for image_path in image_paths:
    # Load the image
    file_name = image_path.split('/')[-1].split('\\')[-1]
    image = cv2.imread(image_path)
    output_image = image.copy()
    y_shape, x_shape = image.shape[0], image.shape[1]
    image = [image]

    # Load custom model
    model = torch.hub.load(
        'ultralytics/yolov5',
        'custom',
        path=
        'D:/Deep_Learning/Algorithm/YOLOV5/GunDetection_YoloV5/yolov5/runs/train/exp/weights/best.pt',
        device=0)
    output = model(image)
    output = output.xyxyn[0]
    for data in output:
        if data[4] > 0.4:
            x1, y1, x2, y2 = int(data[0] * x_shape), int(
                data[1] * y_shape), int(data[2] * x_shape), int(data[3] *
                                                                y_shape)
            # Let's draw the rectangle
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(output_image, model.names[int(data[5])], (x2, y2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imwrite(
        'D:/Deep_Learning/Algorithm/YOLOV5/GunDetection_YoloV5/test_predict/' +
        file_name, output_image)
