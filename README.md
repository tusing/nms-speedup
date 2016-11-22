# nms-speedup
A highly parallelized implementation of non-maximum suppression for object detection, used primarily for autonomous self-driving cars.

# Installation
~~~~
make
~~~~

Run tests with 
~~~~
python tests.py
~~~~

Run GPU
~~~~
g++ -O2 -c nmsGPU.cpp -I/usr/local/cuda-7.5/include
g++ nmsGPU.o -o nmsGPU -L/usr/local/cuda-7.5/lib64 -lOpenCL
./nmsGPU
~~~~~
Run benchmark with bichen's data
~~~~
python tests.py -f
~~~~

Run correctness on single file from bichen's data WARNING THIS IS VERY SLOW (>5 minutes)
~~~~
python tests.py -c
~~~~

