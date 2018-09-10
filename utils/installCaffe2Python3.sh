## install CUDA 9.2, nVIDIA graphic driver > 390.xx, cuDnn 7.2 first!

sudo python3 -m pip install --upgrade pip
sudo python3 -m pip install setuptools future numpy protobuf

sudo apt-get install -y --no-install-recommends libgflags-dev

mkdir caffe2-pytorch && cd caffe2-pytorch
git clone --recursive https://github.com/pytorch/pytorch.git ./
git submodule update --init

mkdir build && cd build

cmake -DPYTHON_INCLUDE_DIR='/usr/include/python3.5' -DPYTHON_EXECUTABLE='/usr/bin/python3' -DPYTHON_PACKAGES_PATH='/usr/local/lib/dist-packages' -DPYTHON_LIBRARY='/usr/local/lib/python3.5/dist-packages' ..

sudo make -j"$(nproc)" install
sudo ldconfig
sudo updatedb
locate libcaffe2.so
locate caffe2 | grep /usr/local/include/caffe2

# Verify that Caffe2 is installed successfully and it can run with GPU support

python3 -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"
python3 -c 'from caffe2.python import workspace; print(workspace.NumCudaDevices())'
 
