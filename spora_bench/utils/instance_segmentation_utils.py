import numpy as np
from skimage.segmentation import relabel_sequential
from scipy.optimize import linear_sum_assignment



def _label_overlap(x : np.ndarray, y : np.ndarray):
    """
    Computes the overlap count between two label masks.

    Args:
        x: array of integers with same shape as x, each entry represents label
        y: array of integers with same shape as y, each entry represents label
    Returns:
        overlap: 2D array of shape (n_labels_x, n_labels_y) where
            overlap[i, j] is the number of pixels where x == i and y == j
            n_labels_x = x.max() + 1, n_labels_y = y.max() + 1
    """
    x = x.ravel()
    y = y.ravel()
    x_max = x.max() + 1
    y_max = y.max() + 1

    overlap = np.bincount(
        x.astype(np.int64) * y_max + y.astype(np.int64),
        minlength=x_max * y_max
    ).reshape(x_max, y_max)

    return overlap


def _intersection_over_union(overlap: np.ndarray):
    """
    Computes the intersection over union (IoU) for each pair of labels given the overlap counts.

    Args:
        overlap: 2D array of shape (n_labels_x, n_labels_y) where
            overlap[i, j] is the number of pixels where x == i and y == j
            n_labels_x = x.max() + 1, n_labels_y = y.max() + 1
    Returns:
        iou: 2D array of shape (n_labels_x, n_labels_y)
            iou[i, j] is the intersection over union of label i in x and label j in y
    """
    intersection = overlap
    area_x = overlap.sum(axis=1, keepdims=True)
    area_y = overlap.sum(axis=0, keepdims=True)
    union = area_x + area_y - intersection

    iou = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union != 0)

    return iou


def compute_matches(true_mask: np.ndarray, pred_mask: np.ndarray, iou_thresholds):
    """
    Computes true positives, false positives, and false negatives based on the IoU threshold. 
    Labeles are matched 1-to-1 using the Hungarian algorithm maximizing pairings with IoU above the threshold followed by the total sum of IoU scores.
    Matches are subsequently filtered based on their IoU threshold.

    Args:
        true_mask: 2D array of integers representing the ground truth labels
        pred_mask: 2D array of integers representing the predicted labels
        iou_threshold: list, the IoU thresholds to consider a prediction as a true positive
    Returns:
        List of tuples (thr, tp, fp, fn) for each IoU threshold
    """
    y_true, _, _ = relabel_sequential(true_mask)
    y_pred, _, _ = relabel_sequential(pred_mask)

    overlap = _label_overlap(y_true, y_pred)
    iou = _intersection_over_union(overlap)[1:, 1:]
    n_true, n_pred = iou.shape
    n_matched = min(n_true, n_pred)

    if n_matched == 0:
        return [(0, n_pred, n_true) for _ in iou_thresholds]

    results = []
    for thr in iou_thresholds:
        costs = -(iou >= thr).astype(float) - iou / (2 * n_matched)
        true_ind, pred_ind = linear_sum_assignment(costs)
        match_ok = iou[true_ind, pred_ind] >= thr
        tp = int(np.count_nonzero(match_ok))
        fp, fn = n_pred - tp, n_true - tp
        results.append((thr, tp, fp, fn))

    return results

