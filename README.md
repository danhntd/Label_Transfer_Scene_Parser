# Nighttime Scene Understanding with Label Transfer Scene Parser

This reposity is the official implementation of the paper entitled: **Nighttime Scene Understanding with Label Transfer Scene Parser**. <br>
**Authors**: Thanh-Danh Nguyen, Nguyen Phan, Tam V. Nguyen*, Vinh-Tiep Nguyen, and Minh-Triet Tran.


## 1. Environment Setup
Download and install Anaconda with the recommended version from [Anaconda Homepage](https://www.anaconda.com/download): [Anaconda3-2019.03-Linux-x86_64.sh](https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh) 
 
```
cd <your_root>/Label_Transfer_Scene_Parser/
curl -O https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
bash Anaconda3-2019.03-Linux-x86_64.sh
```

After completing the installation, please create and initiate the workspace with the specific versions below.

```
conda create --name LTSP python=3
conda activate LTSP
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.2 -c pytorch
conda env update -f enviroment.yml --prune
```


## 2. Data Preparation
#### 2.1. Image Domain Translation



```
<code>
```


#### 2.2. Semantic Scence Parser

In this work, we used Cityscapes dataset as our main semantic segmentation training and validation dataset. We also utilized Nighttime Driving Test as our testing set. Other segmentation datasets are considered appropriate when they follow the data structure and labels of Cityscapes. 
Readers can reach the original published work [Cityscapes](https://github.com/mcordts/cityscapesScripts.git) for details.
Please follow the folder structure to prepare the data:


```
Cityscapes
|---leftImg8bit
    |---train
        |---cityA
            |---*.png
    |---val
    |---test
|---gtFine_trainvaltest
    |---train
        |---gtFine
            |---cityA
                |---*_gtFine_color.png
                |---*_gtFine_labelIds.png
                |---*_gtFine_labelTrainIds.png
                |---*_gtFine_polygons.json
    |---val
    |---test
```

In `mypath.py`, adjust the path to Cityscapes dataset: `<root_path>/Cityscapes/`



## 3. Training Pipeline
Our proposed Label Transfer Scene Parser includes a 5-step pipeline:
<img align="center" src="https://hackmd.io/_uploads/ByCVa1G3h.png">

#### 3.1. Image Domain Translation Training
```
cd <your_root>/Label_Transfer_Scene_Parser/Domain_Translator/
```
```
<code>
```

#### 3.2. Synthetic Nighttime Inference

```
<code>
```

#### 3.3. Semantic Scene Parser Training
```
cd <your_root>/Label_Transfer_Scene_Parser/Semantic_Segmentor/
```
```
CUDA_VISIBLE_DEVICES=0 python train_val_CL_CE6FL4_stage1_cosine_UNIT.py \
        --dataset Cityscapes \
        --save_dir <path/to/save/result>/run_stage1_combine_CE6FL4/
```

#### 3.4. Inference on Unlabeled Nighttime Data
```
CUDA_VISIBLE_DEVICES=0 python infer.py \
    --experiment_dir ./Semantic_Segmentor/run_stage1_combine_CE6FL4 \
    --path_to_unlabel_set /path/to/unlabel/set/ \
    --path_to_save /path/to/dataset/Cityscapes/
```

#### 3.5. Semantic Scene Parser Re-training

```
CUDA_VISIBLE_DEVICES=0 python train_val_CL_CE6FL4_stage2_cosine_UNIT.py \
        --dataset Cityscapes \
        --save_dir <path/to/save/result>/run_stage2_combine_CE6FL4/
        --checkpoint ./Semantic_Segmentor/saved_checkpoints/run_stage1_combine_CE6FL4/Cityscapes/fpn-resnet101/model_best.pth.tar
```

**For testing on the trained models:**

```
CUDA_VISIBLE_DEVICES=0 python test.py \
        --dataset Cityscapes \ 
        --experiment_dir ./Semantic_Segmentor/run_stage2_combine_CE6FL4
```

**For inferencing on testing images:**
```
CUDA_VISIBLE_DEVICES=0 python predict.py \
    --experiment_dir ./Semantic_Segmentor/run_stage2_combine_CE6FL4
    --path_to_save /path/to/destination/folder/
    --path_to_test_set /path/to/dataset/Cityscapes/
```

The whole script commands can be found in `scripts.sh`.

**Released checkpoints and results:**

We provide the checkpoints of our final model including 2 stages: [S1_CE6FL4](https://1drv.ms/u/s!AjGw2N4vyrj-nUwhrMB3PBV7QIbL?e=SmCRsZ) 
and [S2_CE6FL4](https://1drv.ms/u/s!AjGw2N4vyrj-nUuA_QRb6_mJtAs4?e=YMze9a).
Our prediction results on Nighttime Driving Dataset is available at [this link](https://1drv.ms/u/s!AjGw2N4vyrj-nUqBphg65PE5YdK2?e=w4mc4s).

## 4. Visualization
<p align="center">
  <img width="600" src="https://hackmd.io/_uploads/HJNrAJz3n.png">
</p>


## Citation
Please use this bibtex to cite this repository:
```
@article{nguyen2023ltsp,
  title={Nighttime Scene Understanding with Label Transfer Scene Parser},
  author={Nguyen, Thanh-Danh and Phan, Nguyen and Nguyen, Tam V. and Nguyen, Vinh-Tiep and Tran, Minh-Triet},
  journal={-},
  volume={-},
  pages={-},
  year={2023},
  publisher={-}
}
```


## Acknowledgements

[FPN-Semantic-Segmentation](https://github.com/Andy-zhujunwen/FPN-Semantic-segmentation)
[FCN-Pytorch](https://github.com/pochih/FCN-pytorch)
[Pytorch-Deeplab-Xception](https://github.com/jfzhang95/pytorch-deeplab-xception)
[Pytorch-FPN](https://github.com/kuangliu/pytorch-fpn)
[FPN.Pytorch](https://github.com/jwyang/fpn.pytorch)