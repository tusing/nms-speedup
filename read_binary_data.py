import struct
def read_binary_file(filename):
	'''
	boxes: array of [cx, cy, w, h] (center format)
    probs: array of probabilities
	'''
	f = open(filename, "rb")
	x = f.read(4)
	classes = []
	while x:
		num_boxes = struct.unpack('i',x)
		boxes = []
		probs = []
		for i in range(num_boxes[0]):
			box = []
			box.append(struct.unpack('f',f.read(4))[0])
			box.append(struct.unpack('f',f.read(4))[0])
			box.append(struct.unpack('f',f.read(4))[0])
			box.append(struct.unpack('f',f.read(4))[0])
			prob = struct.unpack('f',f.read(4))[0]
			boxes.append(box)
			probs.append(prob)
		classes.append((boxes, probs))
		x = f.read(4)

	return classes