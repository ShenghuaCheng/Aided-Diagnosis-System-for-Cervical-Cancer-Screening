# Aided-Diagnosis-System-for-Cervical-Cancer-Screening
> Recommend the top k lesion cells and predict the positive probability of WSIs.
 ---
## Computer requirements
System: Win10 \
GPU: Nvidia 1080Ti or better (at least 10G memory)\
CPU: Intel i7 or better\
System Memory: 16G or better

## Environment requirements
Nvidia GPU corresponding driver\
CUDA: cuda 9.0\
cudnn: cudnn 7.0\
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

## Supported WSI formats
WSI formats supported by the opensource OpenSlide library, including `x.svs`, `x.mrxs`, `x.tif`, etc;
WSI resolution: 20× or 40× （0.1 – 0.6 um/pixel, 0.1 – 0.4 um/pixel is better)

## Functions
### Model training and inferring
- `train_*.py` for `model1`, `model2`, `rnn` training.
- `eval_*.py` for `model1`, `model2`, `rnn` evaluation.
- `predict.py` for `model1` & `model2` joint inference, rnn scores can be calculated by `eval_rnn.py` with the predicted images, final out put can be calculated by `scripts/rawRNNtopTOFinalScore.py` with rnn scores of single slide.
### Utils classes and functions

---
# C++ software 
[The C++ software](./SoftwareManual/SoftwareManual.md) with test WSIs is available at [Baidu Cloud](https://pan.baidu.com/s/1UmQzASwvlpKLO7hbwaDc_A).
Correspongding [user manual pdf](./SoftwareManual/User%20Manual%20of%20C++%20Software.pdf) is uploaded.\
Codes can be provided by email xlliu@mail.hust.edu.cn or chengshen@hust.edu.cn.

