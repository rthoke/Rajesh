import numpy as np

Steps = """
The steps to perform Non-Maximum Suppression (NMS) in object detection using YOLO are:

1 - Sort the detections by their confidence scores in descending order.
2 - Select the detection with the highest confidence score and save it as a final detection.
3 - Calculate the Intersection over Union (IoU) between the selected detection and all the remaining detections.
4 - Remove all the detections with IoU greater than a certain threshold (usually 0.5) from the list of detections.
5 - Repeat steps 2-4 until no detections are left.

"""
def iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    area1 = w1 * h1
    area2 = w2 * h2
    
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    union_area = area1 + area2 - intersection_area
    
    iou = intersection_area / union_area
    return iou
def yolo_nms(boxes, scores, threshold=0.5):
    areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)

    sorted_idx = np.argsort(scores)[::-1]

    keep_idx = []
    while len(sorted_idx) > 0:
        i = sorted_idx[0]
        keep_idx.append(i)

        overlaps = iou(boxes[i], boxes[sorted_idx[1:]])
        idx_to_remove = np.where(overlaps > threshold)[0] + 1
        sorted_idx = np.delete(sorted_idx, idx_to_remove)

    return keep_idx
