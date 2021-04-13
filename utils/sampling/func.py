# -*- coding:utf-8 -*-
import os
from lxml import etree
import numpy as np

import cv2
from skimage import exposure


def traverse_dirs(dirs):
    """
    :param dirs:
    :return:
    """


def img_trans(src_dir, resize=(1536, 1536), gamma=0.6, dst_dir=None):
    """
    :param src_dir:  源图片地址
    :param resize: None for original size, or feed a tuple
    :param gamma: None for original color, or feed a float
    :param dst_dir: None for returning a img, or feed a dir
    :return: a BGR img
    """
    dst_img = cv2.imread(src_dir)

    if gamma is not None:
        dst_img = exposure.adjust_gamma(dst_img, 0.6)

    if resize is not None:
        dst_img = cv2.resize(dst_img, (1536, 1536))

    if dst_dir is not None:
        cv2.imwrite(dst_dir, dst_img)
    else:
        return dst_img


def parse_xml(xml_path):
    tree = etree.parse(xml_path)
    root = tree.getroot()
    Labels = []
    Shapes = []
    IndexOrigin = []
    Contours = []
    Annotations = root[0]
    for Annotation in Annotations.findall('Annotation'):
        Labels.append(Annotation.get('PartOfGroup'))
        Shapes.append(Annotation.get('Type'))
        IndexOrigin.append(Annotation.get('Name').split(' ')[1])
        points=[]
        Coordinates = Annotation[0]
        for Coordinate in Coordinates:
            points.append([int(float(Coordinate.get('X'))), int(float(Coordinate.get('Y')))])
        points = np.stack(points)
        Contours.append(points)
    return Labels, Shapes, IndexOrigin, Contours
