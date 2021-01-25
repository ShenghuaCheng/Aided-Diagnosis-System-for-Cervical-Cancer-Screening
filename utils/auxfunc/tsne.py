# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: AidedDiagnosisSystem
@File: tsne.py
@Date: 2020/1/14 
@Time: 11:03
@Desc: 包含tsne降维以及其可视化
'''
import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import manifold


def plot_embedding(embedding, score, label, title=None):
    # 归一化
    x_min, x_max = np.min(embedding, 0), np.max(embedding, 0)
    data = (embedding - x_min) / (x_max - x_min)
    # 作图
    plt.figure()
    ax = plt.subplot(111)
    for i, d in enumerate(data):
        c = 'g'
        a = 0.5
        if label[i]:
            c = 'r'
        # if np.abs(label[i]-score[i])<0.5:
        #     s = 10
        # else:
        #     s = 500*(np.abs(label[i]-score[i]))+10

        # s = 50 * (1-np.abs(label[i] - score[i])) + 50

        s = 50
        ax.scatter(d[0], d[1], s, c, alpha=a)
    # # 将标签打印在图上
    # for i in range(data.shape[0]):
    #     plt.text(data[i, 0], data[i, 1], str(label[i]),
    #              color=plt.cm.Set1(label[i]),
    #              fontdict={'weight': 'bold', 'size': 9})

    # # 将图片放在坐标上
    # if hasattr(offsetbox, 'AnnotationBbox'):
    #     # only print thumbnails with matplotlib > 1.0
    #     shown_images = np.array([[32, 32]])  # just something big
    #     for i in range(imgs.shape[0]):
    #         dist = np.sum((data[i] - shown_images) ** 2, 1)
    #         if np.min(dist) < 4e-3:
    #             # don't show points that are too close
    #             continue
    #         shown_images = np.r_[shown_images, [data[i]]]
    #         imagebox = offsetbox.AnnotationBbox(
    #             offsetbox.OffsetImage(cv2.resize(imgs[i][..., ::-1], (64, 64))),
    #             data[i])
    #         ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


def embedding_imgs(imgs, model):
    [scores, feature] = model.predict(imgs)
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    embedding = tsne.fit_transform(feature)
    return scores, embedding

