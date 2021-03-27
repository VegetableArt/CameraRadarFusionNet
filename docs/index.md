---
layout: page
title: Home
category: ja
excerpt: "Hello"
search_omit: true
---

# Reproduction project


---

## 1. Introduction

This project is about paper <a href="https://arxiv.org/abs/2005.07431/" target="_blank">[1]</a> which was published in May 2020. Based on this paper a reproduction has to be done for the course CS4240. The paper provides a link to the GitHub repository containing the code they used to produce the project <a href="https://github.com/TUMFTM/CameraRadarFusionNet/" target="_blank">[2]</a>. The aim of this project to describe how this project can be reproduced. First by trying the provided GitHub code and its instructions and then with adjustments to the code and process to get it working. Due to the hard nature of getting this repository to work (depending on the situation) the main goal of this project will be to describe what aspects are important to keep in mind when sharing code that is used to write a paper about deep learning and what errors can be encountered when trying to reproduce certain code given by an author of a deep learning paper. Also some guidelines will be given to prevent situations where it's almost impossible to reproduce code while given clear instructions.

## 2. Preparing system for usage

The GitHub page of the CRFNet has clear instructions on how to install and use the provided code. Those steps do not always work and there are some knock-off criteria that have to be met in order to follow the instructions. In this section the step by step process that will work is described.

### 2.1 Determining environment

The environment as described in the repository <a href="https://github.com/TUMFTM/CameraRadarFusionNet/" target="_blank">[2]</a> is as follows:

* Linux Ubuntu (tested on versions 16.04 and 18.04)
* CUDA 10.0
* Python 3.5
* Docker 19.03 (only if usage in Docker is desired)
* NVIDIA Container Toolkit (nvidia docker)

The environment itself is very briefly described, which could lead to wrong expectations by users who want to use the code, because there are some notes to be taken into account.

#### 2.1.1 Notes about Ubuntu

The code in the repository can be run using Ubuntu 18.04. This repository can also work in windows but may cause some problems regarding the use of the GPU driver. The method of using Docker will not work on windows due to the lack of nvidia-docker which would passthrough the GPU capabilities to docker.

#### 2.1.2 Notes about CUDA 10.0

The requirement CUDA 10.0 means that this repository will need the CUDA 10.0 which will make it possible to perform algorithmic applications using the GPU. For that reason there is a need for a compatible GPU that works well with the software. The list of compatible GPU's can be found <a href="https://developer.nvidia.com/cuda-gpus/" target="_blank">here</a>.

The lack of a compatible GPU will lead to errors if no compatible GPU with according driver is found.

#### 2.1.3 Notes about Python and its libraries

It needs to be noted that python 3.5 is used in this repository and the support for python 3.5 is dropped since 13 Sep 2020. This means that certain features, performance improvements and bug fixes will be missing which are added to newer python versions. The choice to use python 3.5 is not really clear because python 3.7 was already available since 2018 and it seems to make more sense to use that version than 3.5. It could be that certain utils that are used do not support newer python versions, but that is unlikely because features do get migrated to newer versions of python. 
Upgrading to a newer Python version can break certain components which rely on deprecated functions from Python 3.5. Especially the h5py library (version 2.7.1) will not compile using pip if a python version newer than 3.5 is used. To overcome that problem a newer h5py version can be installed, but that will mean that you are adjusting the requirements which could lead to other problems.
Another relevant note is that the numpy that is included in the `requirement.txt` file will try to install a newer version of numpy that is not backwards compatible with python 3.5. Thus the line with the numpy requirement needs to be removed and an older numpy version (1.16.* is used in this report) needs to be installed manually in order to overcome the dependency problem.

#### 2.1.4 Notes about Docker

Using docker to get the code from the repository working doesn't work initially. Using docker should prevent errors due to different versions of software used, but this had no success using this code. When running the build command the build fails at step: [9/14], which is related to upgrading setuptools. The error is related to invalid syntax used in the main.py of the pip cli. Using python version 3.6 instead of 3.5 in the Dockerfile seems to solve this, but there has no testing been done using that version.

#### 2.1.5 Notes about Google Compute Engine environment

Using a Virtual Machine from Google Compute Engine can be an attractive option in case collaboration is desired and the management of hardware needs to be done by another company. The Google Compute Engine does provide VM's with compatible GPU's which can be used for running this project. It can be expensive (in the range of \$2,500 USD for the minimal configuration: NVIDIA Tesla A100, 12vCPU, 85GB Memory)

### 2.2 Step-by-step installation guide

In this section the setup process will be outlined.

#### 2.2.1 System setup

The system used to perform the operations is: Lenovo Thinkpad P1 Gen, 512GB PCI-e NVMe SSD, 16GB DDR4-3200MHz, Nvidia Quadro T1000 and i7-10750H. The most important step is to check whether the GPU on the system is compatible with CUDA. This can be checked at the following <a href="https://developer.nvidia.com/cuda-gpus/" target="_blank">website</a>. 


In case the GPU is not supported this will not work for you, but there is a workaround to be able to still use this repository. If you don't have a supported GPU you can skip steps 3 till 10 in section <a href="#222-Preparing-OS">preparing OS </a> and check section <a href="#41-DLL-is-not-found--CUDA-lib-errors">gpu-errors </a> to see how to work around the lack of a GPU.


#### 2.2.2 Preparing OS

1. Install Ubuntu 18.04.5 LTS 64bit (Not as a virtual machine, because GPU passthrough is not possible or very hard)

2. Check whether you have a NVIDIA GPU with the following command:

   ```bash
   lspci | grep -i nvidia
   ```

   You should see your GPU

3. Add the key for the CUDA package:

   ```bash 
   sudo apt-key adv --fetch-keys         http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" | sudo tee /etc/apt/sources.list.d/cuda.list
   ```

4. Update package list and install CUDA:

   ```bash
   sudo apt-get update
   sudo apt-get -o Dpkg::Options::="--force-overwrite" install cuda-10-0 cuda-drivers
   ```

5. Reboot your system

6. Add new package to the path variable with the following commands:

   ```bash
   echo 'export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
   source ~/.bashrc
   sudo ldconfig
   ```

7. Check if the driver was correctly installed and configured:

   ```bash
   nvidia-smi
   ```

8. Download cudnn from this <a href="https://developer.nvidia.com/cudnn/" target="_blank">link</a>.  (select: cuDNN v7.5.0 (Feb 21, 2019), for CUDA 10.0)

9. Unpack and install cudnn:

   ```bash
   tar -xf cudnn-10.0-linux-x64-v7.5.0.56.tgz
   sudo cp -R cuda/include/* /usr/local/cuda-10.0/include
   sudo cp -R cuda/lib64/* /usr/local/cuda-10.0/lib64
   ```

10. Install libcupti:

    ```bash
    sudo apt-get install libcupti-dev
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:\$LD_LIBRARY_PATH' >> ~/.bashrc
    ```

11. Install GraphViz:

    ```bash
    sudo apt-get install graphviz
    ```

12. Download and install anaconda as instructed in this <a href="https://docs.anaconda.com/anaconda/install/linux/" target="_blank">link</a>. 

13. Clone the CRFNet repository from this <a href="https://github.com/TUMFTM/CameraRadarFusionNet/" target="_blank">link</a>.

14. Download the nuscnes dataset (we use the mini v1.0 dataset) from this <a href="https://www.nuscenes.org/" target="_blank">link</a>. 

#### 2.2.3 Preparing Python environment

1. Create a new anaconda environment and activate it:

   ```bash
   conda create -n yourenvname python=3.5
   conda activate yourenvname
   ```

2. Upgrade pip version and install numpy 1.16.*:

   ```bash
   pip install --upgrade pip
   pip install numpy==1.16.*
   ```

3. Remove the $numpy>=1.16.0$ from the *requirements.txt* in the crfnet folder.

4. Navigate to the your CameraRadarFusionNet/crfnet folder which contains the `requirements.txt` file. Then install the requirements:

   ```bash
   pip install -e .
   ```

5. Run the CRFNet setup

   ```bash
   python setup.py build_ext --inplace
   ```

6. Now CRFNet is ready for usage

1. 