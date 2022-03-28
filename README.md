# Aided-Diagnosis-System-for-Cervical-Cancer-Screening

[Journal Link](https://www.nature.com/articles/s41467-021-25296-x) | [Quick Start](#quick-start) | [Cite](#reference)

> Recommend the top k lesion cells and predict the positive probability of WSIs.


---
## Requirements
### Hardware 
GPU: Nvidia 1080Ti or better (at least 10G memory)\
CPU: Intel i7 or better\
System Memory: 16G or better
### Software
System: Win10 \
Nvidia GPU corresponding driver\
CUDA: cuda 9.0\
cudnn: cudnn 7.0
### Python
Python: 3.6\
Tensorflow-gpu: 1.7.0\
Tensorboard: 1.7.0\
Keras: 2.1.2\
Keras-Applications: 1.0.6\
Numpy: 1.19.5\
Openslide-python: 1.1.1\
Opencv-python: 3.4.1.15\
Pandas: 0.20.3\
Scikit-image: 0.17.2\
Scikit-learn: 0.23.2

---

## Supported WSI formats
WSI formats supported by the opensource OpenSlide library, including `x.svs`, `x.mrxs`, `x.tif`, etc;
WSI resolution: 20× or 40× （0.1 – 0.6 um/pixel, 0.1 – 0.4 um/pixel is better)

---

<h2 id="quick-start">Quick Start</h2>

<details>
<summary>Installation</summary>

**Step1.** Install [CUDA v9.0](https://developer.nvidia.com/cuda-90-download-archive) and [cuDNN v7.0.5](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.0.5/prod/9.0_20171129/cudnn-9.0-windows10-x64-v7)

**Step2.** Download Aided-Diagnosis-System-for-Cervical-Cancer-Screening
```shell
git clone git@github.com:ShenghuaCheng/Aided-Diagnosis-System-for-Cervical-Cancer-Screening.git
cd Aided-Diagnosis-System-for-Cervical-Cancer-Screening
```

**Step3.** Install requirements
```shell
pip3 install -U pip && pip3 install -r requirements.txt
```
</details>

<details>
<summary>Train</summary>

### Train Model1
```shell
# train model1 classifier
python tools/train.py -n model1-cls -b 16
# train model1 locator based on model1 classifier's backbone
python tools/train.py -n model1-loc -b 16 -w [path to model1-cls weight]
```
### Train Model2
```shell
# train model2 classifier
python tools/train.py -n model2-cls -b 32
```
### Train WSI Classifier
```shell
# train WSI classifier
python tools/train.py -n wsi-cls -b 64
```
</details>

<details>
<summary>Evaluation</summary>

Evaluate classifiers.
```shell
python tools/eval.py -n model1-cls -b 16 -w [path to evaluated weight]
                        model2-cls -b 32
                        wsi-cls    -b 64
```
</details>

<details>
<summary>Inference</summary>

### Python
Do inference to WSIs according to config file.
```shell
python tools/inference.py -c configs/wsi_inference.py -f [path to WSI or path to WSI list files] [--intermediate]
```
### C++ Software
**Prepare:** convert `h5` weights to `pb` files.
```shell
python tools/convert_to_pb.py -m model1 -w [path to weights] -o [path to save]
                                 model2
                                 wsi_clf_top10
                                 wsi_cls_top20
                                 wsi_clf_top30
```
**Do inference:** see [C++ software](#c-software)
</details>

<details>
<summary>Tutorials</summary>

- For dataset config file, see [config for dataset](./datasets/README.md)
- For train and eval config file, see [configs](./configs/README.md#train-config)
- For inference config file, see [configs](./configs/README.md#inference-config)
</details>

---
<h2 id="c-software">C++ software</h2>

[The C++ software](./SoftwareManual/SoftwareManual.md) with test WSIs is available at [Baidu Cloud](https://pan.baidu.com/s/1UmQzASwvlpKLO7hbwaDc_A).
Correspongding [user manual pdf](./SoftwareManual/Software%20User%20Manual.pdf) is uploaded.\
Codes can be provided by email xlliu@mail.hust.edu.cn or chengshen@hust.edu.cn.

---
<h2 id="reference">Reference</h2>

If our work is useful for your research, please consider citing our [paper](https://www.nature.com/articles/s41467-021-25296-x):

Cheng, S., Liu, S., Yu, J. et al. Robust whole slide image analysis for cervical cancer screening using deep learning. Nat Commun 12, 5639 (2021). https://doi.org/10.1038/s41467-021-25296-x

```
@article{cheng2021robust,
  title={Robust whole slide image analysis for cervical cancer screening using deep learning},
  author={Cheng, Shenghua and Liu, Sibo and Yu, Jingya and Rao, Gong and Xiao, Yuwei and Han, Wei and Zhu, Wenjie and Lv, Xiaohua and Li, Ning and Cai, Jing and others},
  journal={Nature communications},
  volume={12},
  number={5639},
  year={2021},
  publisher={Nature Publishing Group}
}
```
