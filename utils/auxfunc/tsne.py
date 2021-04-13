# -*- coding:utf-8 -*-
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
        s = 50
        ax.scatter(d[0], d[1], s, c, alpha=a)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


def embedding_imgs(imgs, model):
    [scores, feature] = model.predict(imgs)
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    embedding = tsne.fit_transform(feature)
    return scores, embedding

