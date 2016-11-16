# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""Utility functions used in tf-yolo"""

from utils import *

def nms_serial(boxes, probs, threshold, form='lowerleft'):
    """ Non-Maximum supression.
        Args
            boxes:      array of [cx, cy, w, h] (center format) or [xmin, ymin, xmax, ymax]
            probs:      array of probabilities
            threshold:  two boxes are considered overlapping if their IOU is largher than this threshold
            form:       'center' or 'diagonal'
        Returns
            keep:       array of True or False.
    """
    
    assert form in ['center', 'diagonal', 'lowerleft'], 'bounding box format not accepted: {}.'.format(form)
    if form == 'diagonal':      # convert to center format
        boxes = [bbox_diagonal_to_lowerleft(b) for b in boxes]
    if form == 'center':        # convert to lowerleft format
        boxes = [bbox_center_to_lowerleft(b) for b in boxes]

        
    order = probs.argsort()[::-1]
    keep = [True]*len(order)

    for i in range(len(order)):
        if not keep[order[i]]:
            continue
        for j in range(i+1, len(order)):
            if lowerleft_iou(boxes[order[i]], boxes[order[j]]) > threshold:
                keep[order[j]] = False
    return keep
