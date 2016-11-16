#include <stdlib.h>

// def iou(box1, box2):
//   """Compute the Intersection-Over-Union of two given boxes.

//   Args:
//     box1: array of 4 elements [cx, cy, width, height].
//     box2: same as above
//   Returns:
//     iou: a float number in range [0, 1]. iou of the two boxes.
//   """

//   lr = min(box1[0]+0.5*box1[2], box2[0]+0.5*box2[2]) - \
//       max(box1[0]-0.5*box1[2], box2[0]-0.5*box2[2]) + 1.0
//   if lr > 0:
//     tb = min(box1[1]+0.5*box1[3], box2[1]+0.5*box2[3]) - \
//         max(box1[1]-0.5*box1[3], box2[1]-0.5*box2[3]) + 1.0
//     if tb > 0:
//       intersection = tb*lr
//       union = box1[2]*box1[3]+box2[2]*box2[3]-intersection
//       return intersection/union

//   return 0

/* Scalar naive implementation of NMS, for benchmarking */
// nms_simd(boxes, prob, threshold, form='center')
void* nms_c_src(float *boxes, int *order, int *keep, float threshold, int n) {
//  for i in range(len(order)):
// 	    if not keep[order[i]]:
//          continue
//      for j in range(i+1, len(order)):
//          if iou(boxes[order[i]], boxes[order[j]]) > threshold:
//              keep[order[j]] = False
//  return keep
  	for(int i=0; i<n; i++) {
  		if(!keep[order[i]]) {
  			continue;
  		}
  		for(int j=i+1; j<n; j++) {
  			if(iou(boxes[order[i]], boxes[order[j]]) > threshold) {
  				keep[order[j]] = 0;
  			}
  		}
  	}
  	return keep;  
}

/* Scalar naive implementation of NMS, for benchmarking */
void* nms_simd_src(float *boxes, int *order, int *keep, float threshold, int n) {

    return 0;
}

/* Scalar naive implementation of NMS, for benchmarking */
void* nms_gpu_src(float *boxes, int *order, int *keep, float threshold, int n) {

    return 0;
}

