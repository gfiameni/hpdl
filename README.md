# Welcome to this repo about High Performance Deep Learning

This repository contains different example Jupyter notebooks dealing with large-scale models training with PyTorch. They are inspired by materials, examples, exercies mainly taken from official PyTorch tutotials and other authors. Each notebook contains the list of reference material. 

Topics covered in this PyTorch Multi-GPU approach to Deep learning Models include:

- Data and Model Parallelism
- Message Passing
- Distributed training using Horovord
- Mixed Precision and Memory Format
- Pipeline Parallelism
- and a challenge to test your knowledge



# Prerequisites
To run this tutorial you will need a machine with NVIDIA GPU and also install any of the two listed below.

- [PyTorch](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html), the primitives it provides for [writing distributed applications](https://pytorch.org/tutorials/intermediate/dist_tuto.html) as well as training [distributed models](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html).

- Install the latest [Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) or [Singularity](https://sylabs.io/docs/). Then start you will have to build a Docker or Singularity container.

### Docker Container
To build a docker container, run:
`sudo docker build --network=host -t <imagename>:<tagnumber> .`

For instance:
`sudo docker build -t pytorch:1.0 .`

The code labs have been written using Jupyter notebooks and a Dockerfile has been built to simplify deployment. The following command would expose port 8888 inside the container as port 8888 on the lab machine:

`sudo docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm --network=host -v ~/hpdl/Pytorch_Distributed_Deep_Learning/workspace:/workspace pytorch:1.0 jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/workspace`


The `--gpus` flag is used to enable `all` NVIDIA GPUs during container runtime. The `--rm` flag is used to clean an temporary images created during the running of the container. The `-it` flag enables killing the jupyter server with `ctrl-c`. The `--ipc=host --ulimit memlock=-1 --ulimit stack=67108864` enable sufficient memory allocation to run pytorch within the docker environment. 

The `jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/workspace` command launch the jupyter notebook inside the container. The flag `-v` allows the mapping of working directory on your local machine `~/hpdl/Pytorch_Distributed_Deep_Learning/workspace:/workspace` to `worspace` directory inside the container.


This command may be customized for your hosting environment. Now, open the jupyter notebook in browser: http://localhost:8888

Start by clicking on the `Start_Here.ipynb` notebook.


### Singularity Container
  
To build the singularity container, run:
`sudo singularity build --fakeroot <image_name>.simg Singularity`

For example:
`singularity build --fakeroot pytorch.simg Singularity`


Then, run the container:
`singularity run --nv --bind ~/hpdl/Pytorch_Distributed_Deep_Learning/workspace:/workspace pytorch.simg jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/workspace`

Then, open the jupyter notebook in browser: http://localhost:8888
Start working on the lab by clicking on the `Start_Here.ipynb` notebook.

#Tutorial Duration
The total bootcamp material would take approximately 4 hours.





