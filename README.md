# CRF-Net for Object Detection
This repository provides a neural network for object detection based on camera and radar data. It builds up on the work of [Keras RetinaNet](https://github.com/fizyr/keras-retinanet). 
The network performs a multi-level fusion of the radar and camera data within the neural network.
The network can be tested on the [nuScenes](https://www.nuscenes.org/) dataset, which provides camera and radar data along with 3D ground truth information.

## Requirements
- Linux Ubuntu (18.04)
- Supported GPU: [CUDA GPUs | NVIDIA Developer](https://developer.nvidia.com/cuda-gpus)
- CUDA 10.0
- Python 3.5

## Installation
For the installation instructions check out [this website](https://vegetableart.github.io/CameraRadarFusionNet/).

# Contributions
[1] M. Geisslinger, "Autonomous Driving: "Object Detection using Neural Networks for Radar and Camera Sensor Fusion," Master's Thesis, Technical University of Munich, 2019

[2] M. Weber, "Autonomous Driving: Radar Sensor Noise Filtering and Multimodal Sensor Fusion for Object Detection with Artificial Neural Networks," Master’s Thesis, Technical University of Munich, 2019.

If you find our work useful in your research, please consider citing:

    @INPROCEEDINGS{nobis19crfnet,
        author={Nobis, Felix and Geisslinger, Maximilian and Weber, Markus and Betz, Johannes and Lienkamp, Markus},
        title={A Deep Learning-based Radar and Camera Sensor Fusion Architecture for Object Detection},
        booktitle={2019 Sensor Data Fusion: Trends, Solutions, Applications (SDF)},
        year={2019},
        doi={10.1109/SDF.2019.8916629},
        ISSN={2333-7427}
    }
