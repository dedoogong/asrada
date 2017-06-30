# asrada
main reference : https://github.com/Theano/Theano/issues/5348

download gits 
1. window-caffe : https://github.com/BVLC/caffe/tree/windows
2. mtcnn-python : https://github.com/kuangliu/pycaffe-mtcnn
3. DeepAlignmentN : https://github.com/MarekKowalski/DeepAlignmentNetwork
Windows - Theano

  1-1.Install Anaconda3 under C:\
  1-2.conda create -n py27 anaconda python=2.7
  1-3. activate py27
  
  2. sudo apt-get install python-numpy python-scipy python-dev python-pip python-nose g++ libopenblas-dev git

    [global]
    floatx = float32
    force_device = True
    device = cpu
    mode = FAST_RUN
    cxx = C:\Anaconda3\envs\env_name35\Library\mingw-w64\bin\g++.exe

    [blas]
    ldflags = -LC:\Anaconda3\Library\bin -lmkl_rt

    [gcc]
    cxxflags = -LC:\Anaconda3\envs\env_name35\Library\mingw-w64\include -LC:\Anaconda3\envs\env_name35\Library\mingw-w64\lib -lm

    [nvcc]
    flags=--cl-version=2015 -D_FORCE_INLINES
    
    3. conda install theano
    
    4. download lasagne and modify requriements.txt to theano-0.9

Ubuntu - Theano

1. Install Anaconda2

2. Install Theano 

[global]
floatx = float32
device = gpu

[cuda]
root =/usr/local/cuda-8.0/cuda

[nvcc]
fastmath = True
flags=-I/usr/local/cuda-8.0/include

[blas]
ldflags =-lmkl_rt


Test Theano
------------------------------------------------------------

from theano import function, config, shared, tensor
import numpy
import time

vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 1000

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], tensor.exp(x))
print(f.maker.fgraph.toposort())
t0 = time.time()
for i in range(iters):
    r = f()
t1 = time.time()
print("Looping %d times took %f seconds" % (iters, t1 - t0))
print("Result is %s" % (r,))
if numpy.any([isinstance(x.op, tensor.Elemwise) and
              ('Gpu' not in type(x.op).__name__)
              for x in f.maker.fgraph.toposort()]):
    print('Used the cpu')
else:
    print('Used the gpu')
    
------------------------------------------------------------
