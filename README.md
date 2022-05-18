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
`sudo docker build -t myimage:1.0 .`

The code labs have been written using Jupyter notebooks and a Dockerfile has been built to simplify deployment. The following command would expose port 8888 inside the container as port 8888 on the lab machine:

`sudo docker run --rm -it --gpus=all -p 8888:8888 myimage:1.0`

The `--gpus` flag is used to enable `all` NVIDIA GPUs during container runtime. The `--rm` flag is used to clean an temporary images created during the running of the container. The `-it` flag enables killing the jupyter server with `ctrl-c`. This command may be customized for your hosting environment.


Inside the container launch the jupyter notebook by typing the following command
`jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/workspace/pytorch/jupyter_notebook/`

Then, open the jupyter notebook in browser: http://localhost:8888
Start by clicking on the `Start_Here.ipynb` notebook.

### Singularity Container

To build the singularity container, run:
`sudo singularity build --sandbox <image_name>.simg Singularity`

and copy the files to your local machine to make sure changes are stored locally:
`singularity run --writable <image_name>.simg cp -rT /workspace/ ~/workspace`


Then, run the container:
`singularity run --nv --writable <image_name>.simg jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/workspace/pytorch/jupyter_notebook/`

Then, open the jupyter notebook in browser: http://localhost:8888
Start working on the lab by clicking on the `Start_Here.ipynb` notebook.

#Tutorial Duration
The total bootcamp material would take approximately 4 hours.





