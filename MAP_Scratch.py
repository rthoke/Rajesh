import numpy as np



Steps = """

The steps to calculate mean average precision (MAP) are as follows:

1 - Prepare the data: Collect the ground truth labels and predicted labels for each object in the dataset.

2 - Sort the predicted labels by confidence scores in descending order.

3 - Compute precision and recall: For each predicted label, compute precision and recall by comparing it to the ground truth labels. Precision is the fraction of predicted objects that are true positives, and recall is the fraction of true objects that are correctly predicted.

4 - Compute average precision (AP): For each class, compute the area under the precision-recall curve. This is the average precision (AP) for that class.

5 - Compute mAP: Compute the mean of the average precision (AP) for all classes.

"""
num_classes = ""
def calculate_iou(box1, box2):
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


def compute_ap(gt_boxes, pred_boxes, iou_threshold=0.5):
    pred_boxes = sorted(pred_boxes, key=lambda x: x[5], reverse=True)
    tp = [0] * len(pred_boxes)
    fp = [0] * len(pred_boxes)
    num_gt_boxes = len(gt_boxes)
    for i, pred_box in enumerate(pred_boxes):
        max_iou = -1
        max_j = -1
        for j, gt_box in enumerate(gt_boxes):
            iou = calculate_iou(pred_box[:4], gt_box[:4])
            if iou > max_iou:
                max_iou = iou
                max_j = j
        if max_iou >= iou_threshold and gt_boxes[max_j][4] == pred_box[4]:
            if not gt_boxes[max_j][-1]:
                tp[i] = 1
                gt_boxes[max_j][-1] = True
            else:
                fp[i] = 1
        else:
            fp[i] = 1
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / num_gt_boxes
    precision = np.concatenate(([0], precision, [0]))
    recall = np.concatenate(([0], recall, [1]))
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i+1])
    indices = np.where(recall[:-1] != recall[1:])[0] + 1
    ap = np.sum((recall[indices] - recall[indices-1]) * precision[indices])
    return ap

# calculate mAP over all classes
def compute_map(gt_dict, pred_dict, iou_threshold=0.5):
    aps = []
    for class_id in range(num_classes):
        gt_boxes = gt_dict[class_id]
        pred_boxes = pred_dict[class_id]
        ap = compute_ap(gt_boxes, pred_boxes, iou_threshold)
        aps.append(ap)
    mAP = np.mean(aps)
    return mAP
