# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""Utility functions used in tf-yolo"""
import tensorflow as tf

def iou(box1, box2):
  """Compute the Intersection-Over-Union of two given boxes.

  Args:
	box1: array of 4 elements [cx, cy, width, height].
	box2: same as above
  Returns:
	iou: a float number in range [0, 1]. iou of the two boxes.
  """

  lr = min(box1[0]+0.5*box1[2], box2[0]+0.5*box2[2]) - \
	  max(box1[0]-0.5*box1[2], box2[0]-0.5*box2[2]) + 1.0
  if lr > 0:
	tb = min(box1[1]+0.5*box1[3], box2[1]+0.5*box2[3]) - \
		max(box1[1]-0.5*box1[3], box2[1]-0.5*box2[3]) + 1.0
	if tb > 0:
	  intersection = tb*lr
	  union = box1[2]*box1[3]+box2[2]*box2[3]-intersection
	  return intersection/union

  return 0

def nms(boxes, probs, threshold, form='center'):
  """Non-Maximum supression.
  Args:
	boxes: array of [cx, cy, w, h] (center format) or [xmin, ymin, xmax, ymax]
	probs: array of probabilities
	threshold: two boxes are considered overlapping if their IOU is largher than
		this threshold
	form: 'center' or 'diagonal'
  Returns:
	keep: array of True or False.
  """

  assert form == 'center' or form == 'diagonal', \
	  'bounding box format not accepted: {}.'.format(form)

  if form == 'diagonal':
	# convert to center format
	boxes = [bbox_transform_inv(b) for b in boxes]

  order = probs.argsort()[::-1]
  keep = [True]*len(order)

  for i in range(len(order)):
	if not keep[order[i]]:
	  continue
	for j in range(i+1, len(order)):
	  if iou(boxes[order[i]], boxes[order[j]]) > threshold:
		keep[order[j]] = False
  return keep

def bbox_transform(bbox):
  """convert a bbox of form [cx, cy, w, h] to [xmin, ymin, xmax, ymax]. Works
  for numpy array or list of tensors.
  """
  with tf.variable_scope('bbox_transform') as scope:
	cx, cy, w, h = bbox
	out_box = [[]]*4
	out_box[0] = cx-w/2
	out_box[1] = cy-h/2
	out_box[2] = cx+w/2
	out_box[3] = cy+h/2

  return out_box

def bbox_transform_inv(bbox):
  """convert a bbox of form [xmin, ymin, xmax, ymax] to [cx, cy, w, h]. Works
  for numpy array or list of tensors.
  """
  with tf.variable_scope('bbox_transform_inv') as scope:
	xmin, ymin, xmax, ymax = bbox
	out_box = [[]]*4

	width       = xmax - xmin + 1.0
	height      = ymax - ymin + 1.0
	out_box[0]  = xmin + 0.5*width 
	out_box[1]  = ymin + 0.5*height
	out_box[2]  = width
	out_box[3]  = height

  return out_box
