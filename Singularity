# Copyright (c) 2020 NVIDIA Corporation.  All rights reserved.

Bootstrap: docker
FROM: nvcr.io/nvidia/pytorch:22.04-py3

%runscript
 
    "$@"

%post

    apt-get update -y
    apt-get -y install git nvidia-modprobe
    pip3 install jupyterlab
    pip3 install ipywidgets
    
    #unzip openmpi, make, and install 
    cd /workspace/resources/
    gunzip -c openmpi-4.1.3.tar.gz | tar xf -    
    cd /workspace/resources/openmpi-4.1.3 
    ./configure --prefix=/usr/local
    make all install
    
    #install horovod
    pip3 install horovod
    
    chmod -R 777 /workspace/pytorch/source_code/data
    chmod -R 777 /workspace/pytorch/source_code/saved_models
    
    
    
%files

    English/* /workspace/

%environment
XDG_RUNTIME_DIR=

%labels

AUTHOR Tosin
