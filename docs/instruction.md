---
layout: page
title: Instruction
category: ja
excerpt: "Hello"
search_omit: true
---

# Reproduction project
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

## 3. Configuring and using CRFNet

In this section the configuration and usage of CRFNet is described. We are using the mini dataset of CRFNet. If you are using the normal version of nuscenes you can leave the variable data\_set as it is.

### 3.1 Folder structure and files

All main scripts depend on the following subfolders:

1. configs folder contains the config files. All the settings for the main scripts are described in this file. A default.cfg (configs/default.cfg) shows an exemplary config file. It is recommended to give each configuration a unique name, as models and tensorboard logs are also named after it.
2. data_processing folder contains all functions for preprocessing before the neural network. Data fusion functions are placed in this folder. VIsualization of the generated radar augmented image (RAI) is provided by the generator script for the corresponding data set.
3. model folder contains all the neural network models that can be used. Based on RetinaNet, architectures that fuse camera and radar data at the feature extractor or the FPN are stored here.
4. utils folder contains helper functions that are needed in many places in this repository.

| File               | Description                                                  |
| ------------------ | ------------------------------------------------------------ |
| train_crfnet.py    | Used to train the CRF-Net.                                   |
| evaluate_crfnet.py | Used to evaluate a trained CRF-Net model on the validation set. |
| test_crfnet.py     | Used to test a trained CRF-Net model on the test set. This script can be used to record videos. |
| requirements.txt   | Contains the requirements for the scripts in this repository. |
| setup.py           | Installs the requirements for this repository and registers this repository in the python modules. |

### 3.2 Configuring CRFNet

Create a new config file in the config folder in the crfnet folder and change: 

```
data_set = nuscenes
data_path = /data/nuscenes 
```

to 

```
data_set = nuscenes, mini
data_path = /yourpathtonuscenes
```

To find out what else you can configure you can check the `README.md` file in the `configs` folder. That describes everything that can be configured. Check whether you are happy with the other parameters and change accordingly.

### 3.3 Training

Training on the dataset can be done by running:

```bash
python train_crfnet.py --config configs/crf_net.cfg
```

With options:

* --config <path to your config> to use your config. Per default the config file found at ./configs/local.cfg is used.

Notes:  This will train a model on a given dataset specified in the configs. The result will be stored in saved_models and the logs in tb_logs.

### 3.4 Evaluating

Evaluation of the model to calculate the precision and recall values for a model on the data specified in the config file can be done by running:

```bash
python evaluate_crfnet.py --model saved_models/crf_net.h5 --config configs/crf_net.cfg --st 0.5    
```

With options:

* --config <path to your config> show the path of training config.

* --model <path to model> model file saved from prior training

* --st <score trehshold> select a custom threshold at which predictions are considered as positive.  

* --render to show images with predicted bounding boxes during execution

* --eval\_from\_detection\_pickle to load saved detection files from the hard drive instead of running the model to evaluate it.

Notes: The values and curves are saved onto the hard drive.

### 3.5 Testing model

Testing of the model to run a inference model on the data specified in the config file can be done by running:   

```bash
python test_crfnet.py --model saved_models/crf_net.h5 --config configs/crf_net.cfg --st 0.5 
```

With options:

* --config <path to your config> show the path of training config.
* --model <path to model> model file saved from prior training.
* --st <score trehshold> select a custom threshold at which predictions are considered as positive.
* --render to show images with predicted bounding boxes during execution.
* --no\_radar\_visualization suppresses the radar data in the visualization.
* --inference to run the network on all samples (not only the labeled ones, only TUM camera dataset).

Notes: With the --render option, running this command will show the predicted obstacles with bounding boxes in an image. The results of the renders will be saved in a folder.

### 3.6 Obtained results from tests

To assess the performance of CRFNet three tests have been done and the results are visually compared.

1. The supplied weights are used to test
   1. Results in mAP: 0.6015
2. The mini dataset is used to train using 12 workers and 20 epochs.
   1. Results in mAP: 0.5816 
3. The mini dataset is used to train using 12 workers and 40 epochs
   1. Results in mAP 0.6964

Comparison of visuals shows that the result are mostly similar but show some differences.

IMAGESIMAGES

## 4. Troubleshooting errors

When running the train\_crfnet.py, evaluate\_crfnet.py or test\_crfnet.py file, several errors can occur. The most common errors and their fixes are listed below. Do note that while using CRFNet multiple warnings can occur which are not important to fix (CRFNet will run perfectly fine without fixing them). In case your process quits this means that the error is important to note and fix.
    

#### 4.1 DLL is not found / CUDA lib errors

In case there is no compatible GPU with the according drivers found, the script will result in errors. If the script still needs to be used there is a workaround possible. This workaround will however mean that the performance will drop significantly because only the cpu will be used for processing. To make the repository work without a GPU you can install tensorflow 1.13.* (make sure you are in the correct conda environment): 

```bash
pip install tensorflow==1.13.*
```

#### 4.2 EOFError: Ran out of input    

This error means that the number of python workers doing multiprocessing on the background exceed your computers capabilities. To fix this error go to the local.cfg file and set the "workers" variable to 0 if you have a cpu. If you have a gpu, try reducing the number and increase the batch size. Note: reducing the number of parallel processes will increase training-time significantly. 

#### 4.3 Tensorflow Allocation Memory: Allocation of x exceeds 10% of system memory

This means that the program uses more memory than a threshold, try reducing the batch size or the amount of workers. It could also be that your system doesn't have enough resources available to run the CRFNet properly.

#### 4.4 Killed

This means that the program uses more resources than it has available, try reducing the batch size or the amount of workers. It could also be that your system doesn't have enough resources available, which means that you have to scale up your resources. You can check your System monitor to assess which resource topped as can be seen in the figure below where the swap memory and cpu have topped and the process is killed.

![](https://codimd.s3.shivering-isles.com/demo/uploads/upload_011a15de7d0cdf87db3ce813e0a70fb3.png)


#### 4.5 CUDA\_ERROR\_OUT\_OF\_MEMORY

This error means that you ran out of memory on your GPU. A way to solve this is to change the batch size (bigger or smaller). In our case this problems was solved after changing the batch size from 1 to 5. A reboot of the system could also help. Another time the solution which helped us is adding the following lines below the import statement of tensorflow in `train_crfnet.py` and `test_crfnet.py` :

```python
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
config = tf.ConfigProto(gpu_options=gpu_options) 
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
```

#### 4.6 cudnn PoolForward launch failed

If you receive this error there is probably something wrong with memory allocation by tensorflow. To fix this you can add the following lines directly bellow the line that imports tensorflow into the script (this will possibly happen in the test\_crfnet.py script).

```python
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
config = tf.ConfigProto(gpu_options=gpu_options) 
config.gpu_options.allow_growth = True
session = tf.Session(config=config)    
```

## 5. Conclusion

While initially getting the repository to work may seem impossible due to numerous errors coming from legacy scripts it's possible to use these steps and criteria to make it work (or to accept that it won't work on a certain setup). By following these steps on different setups we have been able to check what works and what doesn't. Some guidelines to make sure that interested parties can easily use a certain repository can be set:

* Define a list of hardware and software requirements.
* Test a repository on different system and inspect the behavior.
* Indicate whether the repository will be monitored for new issues and whether they will be fixed in order to help possible users.
* List common encountered errors and their possible source and fix.

## 6. References

1. Nobix, F., Chair of Automotive Technology, Technical University of Munich, Geisslinger, M., Weber, M., Betz, J., \& Lienkamp, M. (2020). A Deep Learning-based Radar and Camera Sensor Fusion Architecture for Object Detection. ArXiv, 1. https://arxiv.org/abs/2005.07431
2. Nobis, F. \& TUM - Institute of Automotive Technology. (n.d.). TUMFTM/CameraRadarFusionNet. GitHub. Retrieved March 16, 2021, from https://github.com/TUMFTM/CameraRadarFusionNet