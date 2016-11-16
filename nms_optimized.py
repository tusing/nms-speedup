import ctypes
import os
import numpy as np
from utils import *
from nms_serial import nms_serial

nms = ctypes.CDLL(os.path.abspath("nms.so"))
testboxes, testprobs = read_binary_file("dataset/boxes.dat")
testprobs = np.asarray(testprobs)

# Include optimized versions of NMS here
# Args:
#   boxes: array of [cx, cy, w, h] (center format) or [xmin, ymin, xmax, ymax]
#   probs: array of probabilities
#   threshold: two boxes are considered overlapping if their IOU is largher than
#       this threshold
#   form: 'center' or 'diagonal'
# Returns:
#   keep: array of True or False.


# C version of NMS, for benchmarking purposes only
def nms_c(boxes, probs, threshold, form='center'):
    assert form == 'center' or form == 'diagonal', 'bounding box format not accepted: {}.'.format(form)
    if form == 'diagonal':  # convert to center format
        boxes = [bbox_transform_inv(b) for b in boxes]

    c_boxes = flatten(boxes)
    c_boxes = (ctypes.c_float * len(c_boxes))(*c_boxes)

    order = probs.argsort()[::-1].tolist()
    c_order = (ctypes.c_int * len(order))(*order)

    keep = [1] * len(order)
    c_keep = (ctypes.c_int * len(keep))(*keep)
    print(c_keep)

    c_threshold = (ctypes.c_float)(float(threshold))
    nms.nms_c_src(c_boxes, c_order, c_keep,  len(keep)) # Work should be done in here
    
    return


# CPU optimized NMS
def nms_simd(boxes, probs, threshold, form='center'):
    assert form == 'center' or form == 'diagonal', 'bounding box format not accepted: {}.'.format(form)
    if form == 'diagonal':  # convert to center format
        boxes = [bbox_transform_inv(b) for b in boxes]
    nms.nms_simd_src()      # Work should be done in here
    return


# GPU optimized NMS
def nms_gpu(boxes, probs, threshold, form='center'):
    assert form == 'center' or form == 'diagonal', 'bounding box format not accepted: {}.'.format(form)
    if form == 'diagonal':  # convert to center format
        boxes = [bbox_transform_inv(b) for b in boxes]
    nms.nms_gpu_src()       # Work should be done in here
    return


def bbox_transform_inv(bbox):
    """ Convert a bbox of form [xmin, ymin, xmax, ymax] to [cx, cy, w, h]. 
        Works for numpy array or list of tensors.
    """
    xmin, ymin, xmax, ymax = bbox
    out_box = [[]]*4

    width       = xmax - xmin + 1.0
    height      = ymax - ymin + 1.0
    out_box[0]  = xmin + 0.5*width 
    out_box[1]  = ymin + 0.5*height
    out_box[2]  = width
    out_box[3]  = height

    return out_box
