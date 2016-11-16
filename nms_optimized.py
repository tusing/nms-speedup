import ctypes
import os

nms = ctypes.CDLL(os.path.abspath("nms.so"))

# Include optimized versions of NMS here

# C version of NMS, for benchmarking purposes only
def nms_c(boxes, prob, threshold, form='center'):
    assert form == 'center' or form == 'diagonal', 'bounding box format not accepted: {}.'.format(form)
    if form == 'diagonal':  # convert to center format
        boxes = [bbox_transform_inv(b) for b in boxes]
    nms.nms_c_src()         # Work should be done in here
    return

# CPU optimized NMS
def nms_simd(boxes, prob, threshold, form='center'):
    assert form == 'center' or form == 'diagonal', 'bounding box format not accepted: {}.'.format(form)
    if form == 'diagonal':  # convert to center format
        boxes = [bbox_transform_inv(b) for b in boxes]
    nms.nms_simd_src()      # Work should be done in here
    return

# GPU optimized NMS
def nms_gpu(boxes, prob, threshold, form='center'):
    assert form == 'center' or form == 'diagonal', 'bounding box format not accepted: {}.'.format(form)
    if form == 'diagonal':  # convert to center format
        boxes = [bbox_transform_inv(b) for b in boxes]
    nms.nms_gpu_src()       # Work should be done in here
    return


def bbox_transform_inv(bbox):
    """convert a bbox of form [xmin, ymin, xmax, ymax] to [cx, cy, w, h]. Works
    for numpy array or list of tensors.
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
