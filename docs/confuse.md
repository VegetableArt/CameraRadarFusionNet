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