# Attention-guided-Feature-Fusion-for-SOD
Code of paper [``Attention-guided Feature Fusion for Small Object Detection``](https://ieeexplore.ieee.org/document/10355735)

<img src="https://github.com/JamesYang568/Attention-guided-Feature-Fusion-for-Small-Object-Detection/assets/49474766/4966a75c-de18-49a1-9c29-157db347a0dd" alt="Overall" width="70%">

### AFFM
Attention guided feature fusion module for contextual and spatial alignment of adjacent feature maps.

<img src="https://github.com/JamesYang568/Attention-guided-Feature-Fusion-for-Small-Object-Detection/assets/49474766/6a33ae3e-e8b5-48a8-836d-fa962f59d16a" alt="AFFM" width="50%">

### SFSM
Shallow Feature supplement module for small object feature reinforcement using cross-attention.

<img src="https://github.com/JamesYang568/Attention-guided-Feature-Fusion-for-Small-Object-Detection/assets/49474766/18aa73c4-b208-4fc9-9d1f-108baaeedf8d" alt="SFSM" width="50%">

## Installation

For MMYOLO, please see https://mmyolo.readthedocs.io/

Placing files in our repository to the appropriate place like MMYOLO, just overwrite or copy is enough.

## Usage

Same as MMYOLO, please see [15 minutes to get started with MMYOLO object detection — MMYOLO 0.5.0 documentation](https://mmyolo.readthedocs.io/en/latest/get_started/15_minutes_object_detection.html#) and our config file is in `configs/att-guided_yolov6s/yolov6_s_myneck.py`.

We used a very simple and intuitive way to present the code, with all modules plug and play.

## Result

TABLE I.       Experimental results on COCO 2017 *test-dev* 

| **Method**          | **Backbone**        |  **AP**  | **AP50** | **AP75** |  **APS** | **APM**  |  **APL** |
| ------------------- | ------------------- | :-----:  | :------: | :------: | :------: | :------: | :------: |
| ABFPN               | ResNet-50           |   38.6   |   61.3   |    -     |   24.4   |   42.0   |   49.9   |
| YOLOX-s             | Modified CSPNet     |   40.5   |   59.7   |   44.2   |   24.1   |   45.2   |   54.0   |
| CL-FPN              | ResNet-101          |   41.0   |   62.9   |   44.5   |   23.4   |   44.0   |   52.0   |
| AC-FPN              | ResNet-101          |   42.4   | **65.1** |   46.2   |   25.0   |   45.2   |   53.2   |
| PPYOLOE-s           | CSPRepResNet        |   43.1   |   60.5   |   46.6   |   23.2   |   46.4   |   56.9   |
| YOLOv6-s (baseline) | EfficientRep        |   43.5   |   60.4   |   46.8   |   23.7   |   48.9   |   59.9   |
| YOLOv8-s            | Modified CSPNet C2f |   44.2   |   61.1   | **47.9** | **25.9** |   49.1   | **60.1** |
| Ours                | EfficientRep        | **44.3** |   61.8   |   47.4   |   24.6   | **49.6** |   59.9   |

TABLE II.       Experimental results on VisDrone2017

| **Method**          |  **AP**  | **AP50** | **AP75** | **APS**  | **APM**  | **APL**  | **Epoch** |
| ------------------- | :------: | :------: | :------: | :------: | :------: | :------: | :-------: |
| YOLOX-s             |   17.6   |   33.9   |   16.2   |   9.0    |   27.7   |   45.0   |    50     |
| YOLOv6-s (baseline) |   19.5   |   33.2   |   19.5   |   9.7    |   30.8   |   47.4   |    50     |
| PPYOLOE-s           |   20.0   |   34.6   |   20.0   |   10.5   |   31.5   |   51.0   |    50     |
| Zhan et al.         |   20.6   |   37.6   |    -     |    -     |    -     |    -     |    300    |
| YOLOv8-s            |   20.9   |   36.8   |   20.7   |   10.7   |   33.2   |   49.7   |    50     |
| FE-YOLOv5           |   21.0   |   37.0   |   20.7   |   13.2   |   29.5   |   39.1   |    300    |
| AMMFN               | **24.7** | **48.1** |   22.9   | **17.0** | **43.6** | **60.1** |    300    |
| Ours                |   24.1   |   37.5   | **24.7** |   14.2   |   33.8   |   49.2   |    50     |

<details> 
    <summary>①</summary>
    The bold ones mean the top performance. 
</details>
<details> 
    <summary>②</summary>
    Test on RTX3080Ti, after 50 epochs training.
</details>

## Contact

jiaxiongyang at tongji dot edu dot cn

## Citation
If you find this project useful for your research, please use the following BibTeX entry.
    
    @INPROCEEDINGS{10355735,
      author={Yang, Jiaxiong and Liu, Xianhui and Liu, Zhuang},
      booktitle={2023 IEEE International Conference on Imaging Systems and Techniques (IST)}, 
      title={Attention-guided Feature Fusion for Small Object Detection}, 
      year={2023},
      pages={1-6},
      doi={10.1109/IST59124.2023.10355735}}

