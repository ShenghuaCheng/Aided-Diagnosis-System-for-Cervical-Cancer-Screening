# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: AidedDiagnosisSystem
@File: visualizingfunc.py
@Date: 2019/12/23 
@Time: 11:27
@Desc: 存放用于可视化分数分布的方程,包括绘制 直方图 和 分位数占比
'''
import os
import numpy as np
import matplotlib.pyplot as plt


def survey(results, category_names):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('RdYlGn')(
        np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5,
                label=colname, color=color)
        xcenters = starts + widths / 2

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            ax.text(x, y, str(int(c)), ha='center', va='center',
                    color=text_color)
    ax.legend(ncol=1, bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')

    return fig, ax


def draw_distribution(title, scores, labels, percentile=[20, 40, 60, 80], fix_split=None, save_root=None):
    scores_c = np.concatenate((scores[labels.astype(bool)], scores[(1 - labels).astype(bool)]))
    boundary = int(np.sum(labels))

    if fix_split is None:
        s_p = np.percentile(scores_c, percentile)
    else:
        s_p = fix_split
    count_p = {i: 0 for i in range(5)}
    count_n = {i: 0 for i in range(5)}
    for i, s in enumerate(scores_c):
        label = 0
        if i < boundary:
            label = 1
        if s <= s_p[0]:
            if label:
                count_p[0] += 1
            else:
                count_n[0] += 1
        elif s_p[0] < s <= s_p[1]:
            if label:
                count_p[1] += 1
            else:
                count_n[1] += 1
        elif s_p[1] < s <= s_p[2]:
            if label:
                count_p[2] += 1
            else:
                count_n[2] += 1
        elif s_p[2] < s <= s_p[3]:
            if label:
                count_p[3] += 1
            else:
                count_n[3] += 1
        elif s_p[3] < s:
            if label:
                count_p[4] += 1
            else:
                count_n[4] += 1

    category_names = ['Pos', 'Neg']
    results = {
        'Level 1 (0.00, %.2f]' % s_p[0]: [count_p[0], count_n[0]],
        'Level 2 (%.2f, %.2f]' % (s_p[0], s_p[1]): [count_p[1], count_n[1]],
        'Level 3 (%.2f, %.2f]' % (s_p[1], s_p[2]): [count_p[2], count_n[2]],
        'Level 4 (%.2f, %.2f]' % (s_p[2], s_p[3]): [count_p[3], count_n[3]],
        'Level 5 (%.2f, 1.00]' % s_p[-1]: [count_p[4], count_n[4]]
    }
    survey(results, category_names)
    plt.title(title, fontsize=25, fontweight='bold', verticalalignment='bottom')
    if save_root:
        if fix_split is None:
            plt.savefig(os.path.join(save_root, 'percentile.png'))
        else:
            plt.savefig(os.path.join(save_root, 'threshold.png'))
    else:
        plt.show()


def draw_hist(title, scores, labels, save_root=None):
    scores_c = np.concatenate((scores[labels.astype(bool)], scores[(1 - labels).astype(bool)]))
    boundary = int(np.sum(labels))

    neg = np.histogram(scores_c[boundary:], bins=10, range=(0, 1))
    pos = np.histogram(scores_c[:boundary], bins=10, range=(0, 1))
    nb_neg, position_neg = neg[0], neg[1]
    nb_pos, position_pos = pos[0], pos[1]

    plt.figure(figsize=(10, 6.18))
    plt.title(title, fontsize=25, fontweight='bold', verticalalignment='bottom')
    # neg = plt.hist(scores[boundary:], bins=20, range=(0, 1), alpha=0.3)
    # pos = plt.hist(scores[:boundary], bins=20, range=(0, 1), alpha=0.3)
    plt.bar(position_neg[:-1] + 0.05, nb_neg, width=-0.04, align='edge', color='g', linewidth=1, alpha=0.6)
    plt.bar(position_pos[:-1] + 0.05, nb_pos, width=0.04, align='edge', color='r', linewidth=1, alpha=0.6)
    x_ticks = [x+0.05 for x in np.linspace(0, 1, 11)]
    plt.xlim((0, 1))
    plt.tick_params(labelsize=15)
    plt.xticks(x_ticks[:-1], ['%.1f~%.1f' % (x-0.05, x+0.05) for x in x_ticks[:-1]], rotation=30)
    plt.legend(['neg %d' % len(scores_c[boundary:]), 'pos %d' % len(scores_c[:boundary])],
               ncol=1, loc='upper center', fontsize='xx-large')

    font = {'family': 'serif',
            'style': 'normal',
            'weight': 'bold',
            'size': 15
            }
    for y, x in zip(list(nb_pos), list(position_pos[:-1])):
        plt.text(x+0.05+0.02, y, '%d' % y, color='r', alpha=0.6, horizontalalignment='center', fontdict=font)
    font = {'family': 'serif',
            'style': 'normal',
            'weight': 'bold',
            'size': 15
            }
    for y, x in zip(list(nb_neg), list(position_neg[:-1])):
        plt.text(x+0.05-0.02, y, '%d' % y, color='g', alpha=0.6, horizontalalignment='center', fontdict=font)

    if save_root:
        plt.savefig(os.path.join(save_root, 'distribution.png'))
    else:
        plt.show()
