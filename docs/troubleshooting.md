---
title: Troubleshooting
---

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
