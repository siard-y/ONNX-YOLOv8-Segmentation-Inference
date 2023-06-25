import cv2
from yoloseg import YOLOSeg
from PIL import Image
import matplotlib.pyplot as plt

# Initialize YOLOv5 Instance Segmentator
model_path = "models/Yolov8_Farm9.onnx"
yoloseg = YOLOSeg(model_path, conf_thres=0.5, iou_thres=0.3)

image_name = "./doc/i1.jpg"

# Read image
cv2_img = cv2.imread(image_name)

# Detect Objects
img_path = "doc/i1.jpg"
cv2_img = cv2.imread(img_path)
pil_img = Image.open(img_path)
# Detect Objects
boxes, scores, class_ids, masks = yoloseg(cv2_img)
# print(boxes, scores, class_ids, len(masks))

# Draw detections
combined_img = yoloseg.draw_masks(cv2_img)
blackened_img = yoloseg.blacken_image(pil_img, masks)
blackened_img_cropped = yoloseg.blacken_image(pil_img, masks, do_crop=True)
blackened_img_resized = yoloseg.blacken_image(pil_img, masks, do_resize=True)
blackened_img_cr = yoloseg.blacken_image(pil_img, masks, do_crop=True, do_resize=True)

plt.imshow(combined_img)
plt.show()

plt.imshow(blackened_img)
plt.show()

plt.imshow(blackened_img_cropped)
plt.show()

plt.imshow(blackened_img_resized)
plt.show()

plt.imshow(blackened_img_cr)
plt.show()
