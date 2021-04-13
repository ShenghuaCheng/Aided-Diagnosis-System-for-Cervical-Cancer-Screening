# -*- coding:utf-8 -*-
import os
import argparse
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils.manu.aux_func import create_encoders, rd_imgs

IMG_LIST = {
    'A': {
        'P': r'P.txt',
        'N': r'N.txt'
    },
    'B': {
        'P': r'P.txt',
        'N': r'N.txt'
    },
    'C': {
        'P': r'P.txt',
        'N': r'N.txt'
    },
    'D': {
        'P': r'P.txt',
        'N': r'N.txt'
    },
    'E': {
        'P': r'P.txt',
        'N': r'N.txt'
    },
    'F': {
        'P': r'P.txt',
        'N': r'N.txt'
    },
    'G': {
        'P': r'P.txt',
        'N': r'N.txt'
    },
    'H': {
        'P': r'P.txt',
        'N': r'N.txt'
    },
    'I': {
        'P': r'P.txt',
        'N': r'N.txt'
    },
    'J': {
        'P': r'P.txt',
        'N': r'N.txt'
    },
    'K': {
        'P': r'P.txt',
        'N': r'N.txt'
    },
    'L': {
        'P': r'P.txt',
        'N': r'N.txt'
    }
}


def do_plot(features_embedded, labels, save_root, fig_name=None):
    mkr = {1:'1', 2:'o', 3:'+', 4:'D', 5:'x', 6:'s', 7:'v', 8:'^', 9:'<', 10:'>', 11:'*', 12:'p', }
    clr = {
        1: {
            0: 'g',
            1: 'r',
        },
        2: {
            0: 'g',
            1: 'r',
        },
        3: {
            0: 'c',
            1: 'orange',
        },
        4: {
            0: 'c',
            1: 'orange',
        },
        5: {
            0: 'b',
            1: 'm',
        },
        6: {
            0: 'b',
            1: 'm',
        },
        7: {
            0: 'lime',
            1: 'tomato',
        },
        8: {
            0: 'lime',
            1: 'tomato',
        },
        9: {
            0: 'skyblue',
            1: 'gold',
        },
        10: {
            0: 'skyblue',
            1: 'gold',
        },
        11: {
            0: 'slateblue',
            1: 'deeppink',
        },
        12: {
            0: 'slateblue',
            1: 'deeppink',
        }
    }
    s = 6
    fig = plt.figure(figsize=(13, 13))
    tmp_lb = 0
    tmp_pts = []
    for pt, lb in zip(features_embedded.tolist(), labels):
        if tmp_lb != lb:
            if len(tmp_pts):
                tmp_pts = np.array(tmp_pts)
                plt.scatter(tmp_pts[:, 0], tmp_pts[:, 1],
                            s=s,
                            c=clr[tmp_lb//10][tmp_lb%10],
                            marker=mkr[tmp_lb//10],
                            label='{}{}'.format(chr(ord('A')-1+tmp_lb//10), 'P' if tmp_lb%10 else 'N')
                            )
            tmp_lb = lb
            tmp_pts = [pt]
        else:
            tmp_pts.append(pt)
    tmp_pts = np.array(tmp_pts)
    plt.scatter(tmp_pts[:, 0], tmp_pts[:, 1],
                s=s,
                c=clr[tmp_lb // 10][tmp_lb % 10],
                marker=mkr[tmp_lb // 10],
                label='{}{}'.format(chr(ord('A') - 1 + tmp_lb // 10), 'P' if tmp_lb % 10 else 'N')
                )
    plt.legend()
    title = 't-SNE of {} at Group'.format(os.path.split(save_root)[-1])
    lbs = set([lb//10 for lb in set(labels)])
    if len(lbs) == 6:
        title += ' ALL'
    else:
        for lb in lbs:
            title += ' {}'.format(chr(ord('A') - 1 + lb))
    plt.title(title)
    plt.xlim((AXES_MIN, AXES_MAX))
    plt.ylim((AXES_MIN, AXES_MAX))
    # plt.show()
    fig.savefig(os.path.join(save_root, '{}.png'.format(fig_name if fig_name else 't-SNE')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_root', default=r'\t-SNE_12')
    parser.add_argument('--m_name', choices=['HR_model', 'LR_model', 'Baseline', 'Mined', 'Enhanced', 'Origin'])
    parser.add_argument('--gpus', type=str, default='1')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    save_root = args.save_root
    m_name = args.m_name
    # create encoder
    encoder, in_shape = create_encoders(m_name)
    # encode images
    save_root = os.path.join(save_root, m_name)
    os.makedirs(save_root, exist_ok=True)
    all_dirs = []
    all_labels = []
    all_features = []
    all_scores = []
    for group, dataset in IMG_LIST.items():
        for lab, dirs_ls in dataset.items():
            img_dirs = [l.strip() for l in open(dirs_ls, 'r').readlines()]
            labels = [int('{:d}{:d}'.format(ord(group)-ord('A')+1, 1 if lab=='P' else 0))]*len(img_dirs)

            if group in ['A', 'B', 'C', 'D', 'E', 'F']:
                m_results = encoder.predict(rd_imgs(img_dirs, in_shape))
            elif group in ['G', 'H', 'I', 'J', 'K', 'L']:
                m_results = encoder.predict(rd_imgs(img_dirs, in_shape, re_size=(384, 384)))
            else:
                raise ValueError('wrong group %s' % group)

            scores = m_results[0]
            features = m_results[1]
            np.savez(os.path.join(save_root, 'feature_data_{}_{}.npz'.format(group, lab)),
                     dirs=np.array(img_dirs),
                     labels=np.array(labels),
                     features=features,
                     scores=scores,
                     )

            all_dirs+=img_dirs
            all_labels+=labels
            all_features+=features.tolist()
            all_scores+=scores.tolist()
    # save all
    np.savez(os.path.join(save_root, 'feature_data.npz'),
             dirs=np.array(all_dirs),
             labels=np.array(all_labels),
             features=np.array(all_features),
             scores=np.array(all_scores)
             )
    # do t-SNE
    features_embedded = TSNE(n_components=2,
                             perplexity=50,
                             early_exaggeration=18.,
                             learning_rate=200.,
                             n_iter=3000,
                             n_iter_without_progress=300,
                             min_grad_norm=1e-8,
                             metric="euclidean",
                             init='pca',
                             verbose=1,
                             random_state=42,
                             method='barnes_hut',
                             angle=0.2,
                             n_jobs=16
                             ).fit_transform(np.array(all_features))
    np.save(os.path.join(save_root, 'features_embedded.npy'), features_embedded)
    # do plot
    global AXES_MIN
    global AXES_MAX
    AXES_MIN = np.min(features_embedded)
    AXES_MAX = np.max(features_embedded)
    f_items = np.vsplit(features_embedded, 12)
    l_items = np.hsplit(np.array(all_labels), 12)
    do_plot(features_embedded, all_labels, save_root)
    do_plot(f_items[0], l_items[0], save_root, fig_name='t-SNE_A')
    do_plot(f_items[1], l_items[1], save_root, fig_name='t-SNE_B')
    do_plot(f_items[2], l_items[2], save_root, fig_name='t-SNE_C')
    do_plot(f_items[3], l_items[3], save_root, fig_name='t-SNE_D')
    do_plot(f_items[4], l_items[4], save_root, fig_name='t-SNE_E')
    do_plot(f_items[5], l_items[5], save_root, fig_name='t-SNE_F')
    do_plot(f_items[6], l_items[6], save_root, fig_name='t-SNE_G')
    do_plot(f_items[7], l_items[7], save_root, fig_name='t-SNE_H')
    do_plot(f_items[8], l_items[8], save_root, fig_name='t-SNE_I')
    do_plot(f_items[9], l_items[9], save_root, fig_name='t-SNE_J')
    do_plot(f_items[10], l_items[10], save_root, fig_name='t-SNE_K')
    do_plot(f_items[11], l_items[11], save_root, fig_name='t-SNE_L')
    #
    # do_plot(np.vstack(f_items[:2]), np.hstack(l_items[:2]), save_root, fig_name='t-SNE_AB')
    #
    # do_plot(np.vstack(f_items[:1]+f_items[4:6]), np.hstack(l_items[:1]+l_items[4:6]), save_root, fig_name='t-SNE_AEF')
    #
    # do_plot(np.vstack(f_items[2:4]), np.hstack(l_items[2:4]), save_root, fig_name='t-SNE_CD')
    # do_plot(np.vstack(f_items[4:6]), np.hstack(l_items[4:6]), save_root, fig_name='t-SNE_EF')
    # do_plot(np.vstack(f_items[:4]), np.hstack(l_items[:4]), save_root, fig_name='t-SNE_ABCD')
    # do_plot(np.vstack(f_items[2:6]), np.hstack(l_items[2:6]), save_root, fig_name='t-SNE_CDEF')
