import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# Import COCO config
# sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
from samples.coco import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# class_names = ['car']

# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
# image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
image = skimage.io.imread('6.jpg')

# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])

image_c = image.copy()

upper_lx = 9999
upper_ly = 9999
lower_rx = 0
lower_ry = 0
car_size = 0

for i in range(len(r['class_ids'])):
    if r['class_ids'][i] == 3:
        visualize.draw_box(image_c, r['rois'][i], (255, 0, 0))
        r_size = (r['rois'][i][2] - r['rois'][i][0]) * (r['rois'][i][3] - r['rois'][i][1])
        if r_size > car_size:
            car_size = r_size
            upper_lx = r['rois'][i][0]
            upper_ly = r['rois'][i][1]
            lower_rx = r['rois'][i][2]
            lower_ry = r['rois'][i][3]

cv2.rectangle(image_c, (upper_ly, upper_lx), (lower_ry, lower_rx), (0, 0, 255), 3)
print(upper_lx, upper_ly, lower_rx, lower_ry)

skimage.io.imsave('car_detect.jpg', image_c)


def get_thresholded_image(img):
    # img, M = birds_eye(img)

    s_channel = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:, :, 2]

    l_channel = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)[:, :, 0]

    b_channel = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, 2]

    # Threshold color channel
    s_thresh_min = 180
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    b_thresh_min = 155
    b_thresh_max = 200
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= b_thresh_min) & (b_channel <= b_thresh_max)] = 1

    l_thresh_min = 225
    l_thresh_max = 255
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1

    cv2.imwrite('output_images/s.jpg', s_binary.astype('uint8') * 255)
    cv2.imwrite('output_images/l.jpg', l_binary.astype('uint8') * 255)
    cv2.imwrite('output_images/b.jpg', b_binary.astype('uint8') * 255)

    # color_binary = np.dstack((u_binary, s_binary, l_binary))

    combined_binary = np.zeros_like(s_binary)
    combined_binary[(l_binary == 1) | (s_binary == 1)] = 1

    return combined_binary


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0)

left_crop = gray[upper_lx:, 0:upper_ly]
bottom_crop = gray[lower_rx:, upper_ly:lower_ry]
right_crop = gray[upper_lx:, lower_ry:]

skimage.io.imsave('left_crop.jpg', left_crop)
skimage.io.imsave('bottom_crop.jpg', bottom_crop)
skimage.io.imsave('right_crop.jpg', right_crop)

for i in range(3):
    if i == 0:
        gray1 = left_crop.copy()
    elif i == 1:
        gray1 = bottom_crop.copy()
    elif i == 2:
        gray1 = right_crop.copy()
    v = np.median(gray1)
    lower = (1.0 - 0.33) * v
    upper = (1.0 + 0.33) * v
    while 1:
        # edges = auto_canny(gray)
        edges = cv2.Canny(gray1, lower, upper, 3)

        # Detect points that form a line
        lines1 = cv2.HoughLinesP(edges, 1, np.pi / 180, 44, minLineLength=15, maxLineGap=5)
        # lines1 = cv2.HoughLines(edges, 1, np.pi / 180, 30, None, 5, 5)
        if lines1 is None:
            upper -= 5
            lower -= 2
        elif len(lines1) < 8:
            upper -= 5
            lower -= 2
        elif len(lines1) > 15:
            lower += 5
            upper += 2
        else:
            print(lower, upper, len(lines1))
            break
        if lower >= upper:
            print('no lane lines detected')
            print(lower, upper, len(lines1),i)
            outname = 'output_images/edge' + str(i) + '.jpg'
            skimage.io.imsave(outname, edges)
            exit(1)

    outname = 'output_images/edge'+str(i)+'.jpg'
    skimage.io.imsave(outname, edges)
    out = image.copy()
    for line in lines1:
        x1, y1, x2, y2 = line[0]
        cv2.line(out, (x1, y1), (x2, y2), (255, 0, 0), 1)
    outname = 'output_images/out' + str(i) + '.jpg'
    skimage.io.imsave(outname, out)
