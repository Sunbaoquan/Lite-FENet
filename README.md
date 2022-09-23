# Lite-FENet

This is the official implementation of our paper [**Lite-FENet: Lightweight Multi-scale Feature Enrichment Network for Few-shot Segmentation**](). 



## Get Started

### Environment

- NVIDIA RTX 3090
- cuda==11.1

- Pytorch==1.10  

- torchvision== 0.11.0

- python==3.8


### Datasets

- **PASCAL-5<sup>i</sup> :** consists of [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) and part of [SBD](http://home.bharathh.info/pubs/codes/SBD/download.html). Download this benchmark labels from [here](https://drive.google.com/file/d/1nOpsl-Z1fntyFOqIxN_v26FR3ivHvJ7R/view?usp=sharing).

- **COCO-20<sup>i</sup> :** constructed by [COCO2014](https://cocodataset.org/#download), and our project `lists/coco/extract_coco_mask.py` could make this benchmark labels.

  For more details about these two benchmarks, please check [OSLSM](https://arxiv.org/abs/1709.03410) and [FWB](https://openaccess.thecvf.com/content_ICCV_2019/html/Nguyen_Feature_Weighting_and_Boosting_for_Few-Shot_Segmentation_ICCV_2019_paper.html).

### Overview

- `config/` includes configuration files of two datasets,  `.yaml` file can be modified as necessary.
- `lists/ `includes train/val list files. Please replace the relative paths of the original images/labels in all `.txt` files in this directory with your own or keep with ours.
- `model/ ` includes our proposed model and backbone network.
- `util/` includes data processing, and other method of tools.

### Models

- Before you are ready to run the code, please download [backbone weights](https://drive.google.com/drive/folders/1h7xtjI0BKEg6bFV1R4lDcgVWScG6qtBC?usp=sharing) and put them into the `LightFENet /initmodel/` directory.

- We provide [pre-trained Lite-FENet](https://drive.google.com/drive/folders/1Sj1wskORwgAA7-PbmgrSzCmGmun9JlAK?usp=sharing) under 4 splits of PASCAL-5<sup>i</sup> and COCO-20<sup>i</sup> based on ResNet50 in order to quickly reproduce our paper results.

- Train the model

  ```
  sh train.sh {dataset} {split_backbone}
  ```

​		**For example:** `sh train.sh pascal split0_resnet50 `

- Test the model

  ```
  sh test.sh {dataset} {split_backbone}
  ```

  **For example:** `sh test.sh pascal split0_resnet50 `



## References

This  project is built on [PFENet](https://github.com/Jia-Research-Lab/PFENet/)、[HRNet]([HRNet (github.com)](https://github.com/HRNet))、[DCP](https://github.com/chunbolang/DCP). Thanks their great contribution！