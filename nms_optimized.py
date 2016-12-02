import ctypes
import os
from utils import *
from nms_serial import nms_serial
import time

nms = ctypes.CDLL(os.path.abspath("nms.so"))

nms.nms_gpu_init()

def nms_preprocess(boxes, probs, threshold, form='lowerleft', ordered=False):
    assert form in ['center', 'diagonal', 'lowerleft'], 'bounding box format not accepted: {}.'.format(form)
    if form == 'diagonal':      # convert to center format
        boxes = [bbox_diagonal_to_lowerleft(b) for b in boxes]
    if form == 'center':        # convert to lowerleft format
        boxes = [bbox_center_to_lowerleft(b) for b in boxes]

    n = len(boxes)
    empty = [0];
    if ordered:
        order = probs.argsort()[::-1].tolist() + 8 * [0]
        c_order = (ctypes.c_int * (n + 8))(*order)
        c_probs = (ctypes.c_float)(*empty)
        boxes = [boxes[order[j]] for j in range(n)]
    else:
        order = -1
        c_order = (ctypes.c_int)(*empty)
        probs = probs.tolist() + 8 * [0.0]
        c_probs = (ctypes.c_float * (n + 8))(*probs)

    c_xmin = [box[0] for box in boxes] + 8 * [0.0]
    c_ymin = [box[1] for box in boxes] + 8 * [0.0]
    c_w = [box[2] for box in boxes] + 8 * [0.0]
    c_h = [box[3] for box in boxes] + 8 * [0.0]

    c_xmin = (ctypes.c_float*(n + 8))(*c_xmin)
    c_ymin = (ctypes.c_float*(n + 8))(*c_ymin)
    c_w = (ctypes.c_float*(n + 8))(*c_w)
    c_h = (ctypes.c_float*(n + 8))(*c_h)

    keep = [1] * (n + 8)
    c_keep = (ctypes.c_int * (n + 8))(*keep)
    c_threshold = (ctypes.c_float)(float(threshold))
    c_len = (ctypes.c_int)(n)

    return (c_xmin, c_ymin, c_w, c_h, c_order, c_keep, c_threshold, c_len, c_probs, n, order)
    
def nms_harness(c_func, c_xmin, c_ymin, c_w, c_h, c_order, c_keep, c_threshold, c_len, c_probs, n, order, ordered=False, benchmarked=False):
    if benchmarked:
        starttime = time.time()
    c_func(c_xmin, c_ymin, c_w, c_h, c_order, c_keep, c_threshold, c_len, c_probs) # Work should be done in here
    
    if benchmarked:
        elapsed = time.time() - starttime
    keep = [1] * n
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
    c_xmin, c_ymin, c_w, c_h, c_order, c_keep, c_threshold, c_len, c_probs, n, order = nms_preprocess(boxes, probs, threshold, form, ordered)
    return nms_harness(nms.nms_c_src, c_xmin, c_ymin, c_w, c_h, c_order, c_keep, c_threshold, c_len, c_probs, n, order, ordered, benchmarked)

# OpenMP version of NMS
def nms_omp(boxes, probs, threshold, form='lowerleft', benchmarked=False):
    ordered = True
    c_xmin, c_ymin, c_w, c_h, c_order, c_keep, c_threshold, c_len, c_probs, n, order = nms_preprocess(boxes, probs, threshold, form, ordered)
    return nms_harness(nms.nms_omp_src, c_xmin, c_ymin, c_w, c_h, c_order, c_keep, c_threshold, c_len, c_probs, n, order, ordered, benchmarked)

# Alternate OpenMP version of NMS
def nms_omp1(boxes, probs, threshold, form='lowerleft', benchmarked=False):
    ordered = True
    c_xmin, c_ymin, c_w, c_h, c_order, c_keep, c_threshold, c_len, c_probs, n, order = nms_preprocess(boxes, probs, threshold, form, ordered)
    return nms_harness(nms.nms_omp1_src, c_xmin, c_ymin, c_w, c_h, c_order, c_keep, c_threshold, c_len, c_probs, n, order, ordered, benchmarked)

# C version of NMS, for benchmarking purposes only
def nms_c_unsorted_src(boxes, probs, threshold, form='lowerleft', benchmarked=False):
    ordered = False
    c_xmin, c_ymin, c_w, c_h, c_order, c_keep, c_threshold, c_len, c_probs, n, order = nms_preprocess(boxes, probs, threshold, form, ordered)
    return nms_harness(nms.nms_c_unsorted_src, c_xmin, c_ymin, c_w, c_h, c_order, c_keep, c_threshold, c_len, c_probs, n, order, ordered, benchmarked)

# SIMD NMS
def nms_simd(boxes, probs, threshold, form='lowerleft', benchmarked=False):
    ordered = True
    c_xmin, c_ymin, c_w, c_h, c_order, c_keep, c_threshold, c_len, c_probs, n, order = nms_preprocess(boxes, probs, threshold, form, ordered)
    return nms_harness(nms.nms_simd_src, c_xmin, c_ymin, c_w, c_h, c_order, c_keep, c_threshold, c_len, c_probs, n, order, ordered, benchmarked)

# GPU optimized NMS
def nms_gpu(boxes, probs, threshold, form='lowerleft', benchmarked=False):
    ordered = True
    c_xmin, c_ymin, c_w, c_h, c_order, c_keep, c_threshold, c_len, c_probs, n, order = nms_preprocess(boxes, probs, threshold, form, ordered)
    nms.nms_gpu_mem_transfer(c_xmin, c_ymin, c_w, c_h, c_order, c_keep, c_threshold, c_len, c_probs)
    res = nms_harness(nms.nms_gpu_src, c_xmin, c_ymin, c_w, c_h, c_order, c_keep, c_threshold, c_len, c_probs, n, order, ordered, benchmarked)
    nms.nms_gpu_mem_cleanup()
    return res
