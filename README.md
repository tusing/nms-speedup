# nms-speedup
A highly parallelized implementation of non-maximum suppression for object detection, used primarily for autonomous self-driving cars.

# Installation
~~~~
make
~~~~

Run benchmark with bichen's data
~~~~
python tests.py -f[v] [number of images]
full data benchmark expects an integer argument for the number of images to test, or -1 to test all images
~~~~
Bichen's data available at https://www.dropbox.com/s/3p8n1q0ldz5v10a/data.tgz?dl=0. Download and extract

Run correctness on single file from bichen's data
~~~~
python tests.py -c
~~~~

