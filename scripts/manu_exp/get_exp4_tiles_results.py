# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: AidedDiagnosisSystem
@File: get_exp4_tiles_results.py
@Date: 2020/9/29 
@Time: 10:10
@Desc: 该脚本通过人工检查的tile结果获取病理学家在ROC曲线上的点
'''
import os
import json
from collections import namedtuple
from operator import attrgetter
from xml.etree import ElementTree as ET
from sklearn.metrics import confusion_matrix

Tile = namedtuple('Tile', 'batch id label')


def get_EF_labels():
    e2_dir = r'I:\20200929_EXP4\RAWDATA\exp_4_yjy4000\shuffle_list_E_2.json'
    f2_dir = r'I:\20200929_EXP4\RAWDATA\exp_4_yjy4000\shuffle_list_F_2.json'
    e2 = json.load(open(e2_dir, 'r'))
    f2 = json.load(open(f2_dir, 'r'))
    E = [Tile('E', i, l) for i, l in enumerate(e2['label'])]
    F = [Tile('F', i, l) for i, l in enumerate(f2['label'])]
    return E+F


def parse_cvat_results(cvat_dir):
    cvat_tree = ET.parse(cvat_dir)
    annotations = cvat_tree.getroot()
    results = []
    for image in annotations.findall('image'):
        name = image.get('name')
        _, bat, idx = name.split('/')
        idx = int(idx.split('.')[0])
        tag = 1 if len(image.findall('tag')) else 0
        results.append(Tile(bat, idx, tag))
    return sorted(results, key=attrgetter('batch', 'id'))


def get_confusion_m(labels, results, con=['E','F']):
    # filter by batch
    filter_func = lambda x: x.batch in con
    labels = list(filter(filter_func, labels))
    results = list(filter(filter_func, results))
    # get sample checked manually
    seen = set([(r.batch, r.id) for r in results])
    labels = [l for l in labels if (l.batch, l.id) in seen]
    # calculate confusion metrix
    return confusion_matrix([l.label for l in labels], [r.label for r in results])


if __name__ == '__main__':
    labels = get_EF_labels()
    cvat_dir = r'I:\20200929_EXP4\Tiles\CVAT_EXP4_2.xml'
    results = parse_cvat_results(cvat_dir)
    print('E')
    confusion_m = get_confusion_m(labels, results, con=['E'])
    print(confusion_m)
    print('F')
    confusion_m = get_confusion_m(labels, results, con=['F'])
    print(confusion_m)
    print('EF')
    confusion_m = get_confusion_m(labels, results, con=['E', 'F'])
    print(confusion_m)
