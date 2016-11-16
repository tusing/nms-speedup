import struct
def read_binary_file(filename):
	'''
	boxes: array of [cx, cy, w, h] (center format)
        probs: array of probabilities
	'''
	f = open(filename, "rb")
	x = f.read(4)
        boxes = []
        probs = []
	while x:
		num_boxes = struct.unpack('i',x)
		for i in range(num_boxes[0]):
			box = []
			box.append(struct.unpack('f',f.read(4))[0])
			box.append(struct.unpack('f',f.read(4))[0])
			box.append(struct.unpack('f',f.read(4))[0])
			box.append(struct.unpack('f',f.read(4))[0])
			prob = struct.unpack('f',f.read(4))[0]
			boxes.append(box)
			probs.append(prob)
		x = f.read(4)

	return boxes, probs

def flatten(lst):
        if type(lst) is not list:
                return [lst]
        newlst = []
        for i in lst:
                newlst.extend(flatten(i))
        return newlst


def lowerleft_iou(box1, box2):
        """Compute the Intersection-Over-Union of two given boxes.
        
        Args:
        box1: array of 4 elements [cx, cy, width, height].
        box2: same as above
        Returns:
        iou: a float number in range [0, 1]. iou of the two boxes.
        """
        x0 = max(box1[0], box2[0])
        y0 = max(box1[1], box2[1])
        
        x1 = min(box1[0] + box1[2], box2[0] + box2[2])
        y1 = min(box1[1] + box1[3], box2[1] + box2[3])
        
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        
        if x1 > x0:
                union_area = (x1 - x0) * (y1 - y0)
        else:
                union_area = 0
                
        union_area = max(0, union_area)
        tot_area = area1 + area2 - union_area
        
        if tot_area > 0:

                return float(union_area) / tot_area
        else:
                return 0


def bbox_center_to_diagonal(bbox):
        """convert a bbox of form [xmin, ymin, xmax, ymax] to [cx, cy, w, h]. Works
        for numpy array
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

def bbox_center_to_lowerleft(bbox):
        """convert a bbox of form [cx, cy, w, h] to [minx, miny, w, h]. Works
        for numpy array
        """
        
        cx, cy, w, h = bbox
        out_box = [[]]*4
        

        out_box[0]  = cx - 0.5*w 
        out_box[1]  = cy - 0.5*h
        out_box[2]  = w
        out_box[3]  = h
        
        return out_box


def bbox_center_to_lowerleft(bbox):
        """convert a bbox of form [minx, miny, maxx, maxy] to [minx, miny, w, h]. Works
        for numpy array
        """
        
        minx, miny, maxx, maxy = bbox
        out_box = [[]]*4
        

        out_box[0]  = minx 
        out_box[1]  = miny
        out_box[2]  = maxx - minx
        out_box[3]  = maxy - miny
        
        return out_box
