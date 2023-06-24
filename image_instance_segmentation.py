import cv2
from yoloseg import YOLOSeg

# Initialize YOLOv5 Instance Segmentator
model_path = "models/best.onnx"
yoloseg = YOLOSeg(model_path, conf_thres=0.5, iou_thres=0.3)

image_name = "i1.jpg"

# Read image
img = cv2.imread("i1.jpg")

# Detect Objects
boxes, scores, class_ids, masks = yoloseg(img)

# Draw detections
combined_img = yoloseg.draw_masks(img)
cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
cv2.imshow("Detected Objects", combined_img)
cv2.imwrite(f"doc/img/detected_objects{image_name}", combined_img)
cv2.waitKey(0)
