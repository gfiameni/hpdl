# Copyright (c) 2020 NVIDIA Corporation.  All rights reserved.

# To build the docker container, run: $ sudo docker build -t ai-multi-gpu:latest .
# To run: $ sudo docker run --rm -it --gpus=all -p 8888:8888 -p 8000:8000 ai-multi-gpu:latest
# Finally, open http://127.0.0.1:8888/

# Select Base Image 
FROM nvcr.io/nvidia/pytorch:22.04-py3

# Update the repo
RUN apt-get update -y
# Install required dependencies
RUN apt-get install -y git nvidia-modprobe
# Install required python packages
RUN pip3 install ipywidgets

##### TODO - From the Final Repo Changing this 

# TO COPY the data 
COPY resources/ /workspace/resources

WORKDIR /workspace/resources/
RUN gunzip -c openmpi-4.1.3.tar.gz | tar xf -
WORKDIR /workspace/resources/openmpi-4.1.3
RUN ./configure --prefix=/usr/local
RUN make all install
RUN pip3 install horovod

#WORKDIR /workspace/
## Uncomment this line to run Jupyter notebook by default
#CMD jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/workspace
