import ctypes
import os
from utils import *
from nms_serial import nms_serial
import time

nms = ctypes.CDLL(os.path.abspath("nms.so"))

def nms_preprocess(boxes, probs, threshold, form='lowerleft'):
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

    return (c_xmin, c_ymin, c_w, c_h, c_order, c_keep, c_threshold, c_len)


def nms_harness(c_func, boxes, probs, threshold, form='lowerleft', ordered=False, benchmarked=False):
    assert form in ['center', 'diagonal', 'lowerleft'], 'bounding box format not accepted: {}.'.format(form)
    if form == 'diagonal':      # convert to center format
        boxes = [bbox_diagonal_to_lowerleft(b) for b in boxes]
    if form == 'center':        # convert to lowerleft format
        boxes = [bbox_center_to_lowerleft(b) for b in boxes]

    n = len(boxes)
    empty = [0];
    if ordered:
        order = probs.argsort()[::-1].tolist()
        c_order = (ctypes.c_int * n)(*order)
        c_probs = (ctypes.c_float)(*empty)
        boxes = [boxes[order[j]] for j in range(n)]
    else:
        c_order = (ctypes.c_int)(*empty)
        c_probs = (ctypes.c_float * n)(*probs)

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
    c_func(c_xmin, c_ymin, c_w, c_h, c_order, c_keep, c_threshold, c_len, c_probs) # Work should be done in here
    if benchmarked:
        elapsed = time.time() - starttime

    for i in range(n):
        if ordered:
            keep[order[i]] = c_keep[i]
        else:
            keep[i] = c_keep[i]
    if benchmarked:
        return (keep, elapsed)
    else:
        return keep

# C version of NMS, for benchmarking purposes only
def nms_c(boxes, probs, threshold, form='lowerleft', benchmarked=False):
    ordered = True
    return nms_harness(nms.nms_c_src, boxes, probs, threshold, form, ordered, benchmarked)

# OpenMP version of NMS
def nms_omp(boxes, probs, threshold, form='lowerleft', benchmarked=False):
    ordered = True
    return nms_harness(nms.nms_omp_src, boxes, probs, threshold, form, ordered, benchmarked)

# Alternate OpenMP version of NMS
def nms_omp1(boxes, probs, threshold, form='lowerleft', benchmarked=False):
    ordered = True
    return nms_harness(nms.nms_omp1_src, boxes, probs, threshold, form, ordered, benchmarked)

# C version of NMS, for benchmarking purposes only
def nms_c_unsorted_src(boxes, probs, threshold, form='lowerleft', benchmarked=False):
    ordered = False
    return nms_harness(nms.nms_c_unsorted_src, boxes, probs, threshold, form, ordered, benchmarked)

# SIMD NMS
def nms_simd(boxes, probs, threshold, form='lowerleft', benchmarked=False):
    ordered = True
    return nms_harness(nms.nms_simd_src, boxes, probs, threshold, form, ordered, benchmarked)

# GPU optimized NMS
def nms_gpu(boxes, probs, threshold, form='lowerleft', benchmarked=False):
    ordered=True
    return nms_harness(nms.nms_gpu_src, boxes, probs, threshold, form, ordered, benchmarked)
