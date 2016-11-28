import ctypes
import os
from utils import *
from nms_serial import nms_serial
import time

nms = ctypes.CDLL(os.path.abspath("nms.so"))


# Include optimized versions of NMS here
# Args:
#   boxes: array of [cx, cy, w, h] (center format) or
#                   [xmin, ymin, xmax, ymax] (diaganol format) or
#                   [xmin, ymin, w, h] (lowerleft format)
#   probs: array of probabilities
#   threshold: two boxes are considered overlapping if their IOU is largher than
#       this threshold
#   form: 'center' or 'diagonal'
# Returns:
#   keep: array of True or False.

def nms_harness(c_func, boxes, probs, threshold, form='lowerleft', benchmarked=False):
    assert form in ['center', 'diagonal', 'lowerleft'], 'bounding box format not accepted: {}.'.format(form)
    if form == 'diagonal':      # convert to center format
        boxes = [bbox_diagonal_to_lowerleft(b) for b in boxes]
    if form == 'center':        # convert to lowerleft format
        boxes = [bbox_center_to_lowerleft(b) for b in boxes]

    n = len(boxes)

    order = probs.argsort()[::-1].tolist()
    c_order = (ctypes.c_int * n)(*order)


    boxes = [boxes[order[j]] for j in range(n)]

    c_xmin = [box[0] for box in boxes]
    c_ymin = [box[1] for box in boxes]
    c_w = [box[2] for box in boxes]
    c_h = [box[3] for box in boxes]

    c_xmin = (ctypes.c_float*n)(*c_xmin)
    c_ymin = (ctypes.c_float*n)(*c_ymin)
    c_w = (ctypes.c_float*n)(*c_w)
    c_h = (ctypes.c_float*n)(*c_h)


    keep = [1] * n
    c_keep = (ctypes.c_int * n)(*keep)

    c_threshold = (ctypes.c_float)(float(threshold))

    c_len = (ctypes.c_int)(n)
    if benchmarked:
        starttime = time.time()
    c_func(c_xmin, c_ymin, c_w, c_h, c_order, c_keep, c_threshold, c_len) # Work should be done in here
    if benchmarked:
        elapsed = time.time() - starttime

    for i in range(n):
        keep[order[i]] = c_keep[i]
    if benchmarked:
        return (keep, elapsed)
    else:
        return keep


# C version of NMS, for benchmarking purposes only
def nms_c(boxes, probs, threshold, form='lowerleft', benchmarked=False):
    return nms_harness(nms.nms_c_src, boxes, probs, threshold, form, benchmarked)

# OpenMP version of NMS
def nms_omp(boxes, probs, threshold, form='lowerleft', benchmarked=False):
    return nms_harness(nms.nms_omp_src, boxes, probs, threshold, form, benchmarked)

# Alternate OpenMP version of NMS
def nms_omp1(boxes, probs, threshold, form='lowerleft', benchmarked=False):
    return nms_harness(nms.nms_omp1_src, boxes, probs, threshold, form, benchmarked)

# SIMD NMS
def nms_simd(boxes, probs, threshold, form='lowerleft', benchmarked=False):
    return nms_harness(nms.nms_simd_src, boxes, probs, threshold, form, benchmarked)


# GPU optimized NMS
def nms_gpu(boxes, probs, threshold, form='lowerleft', benchmarked=False):
    return nms_harness(nms.nms_gpu_src, boxes, probs, threshold, form, benchmarked)
