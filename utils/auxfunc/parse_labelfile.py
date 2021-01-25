# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: AidedDiagnosisSystem
@File: parse_labelfile.py
@Date: 2020/1/16 
@Time: 16:41
@Desc: 存放解析标签文件的额函数
'''
import os
import numpy as np
from lxml import etree


def parse_csv(csv_path):
    """解析以csv格式存放的标签数据
    :param csv_path: 含有file1.csv file2.csv的标签文件夹
    :return: 以此返还标签、形状、外接矩形和轮廓
    """
    # 读csv
    fileCsv1 = os.path.join(csv_path, 'file1.csv')
    fileCsv2 = os.path.join(csv_path, 'file2.csv')
    csv1 = open(fileCsv1, "r").readlines()
    csv2 = open(fileCsv2, "r").readlines()
    Labels = []
    Shapes = []
    Contours = []
    Rects = []
    for i in range(len(csv1)):
        line = csv2[i]
        elems = line.strip().split(',')

        label = elems[0]  # 标签
        shape = elems[1].strip().split(' ')[0]  # 标注形状

        # 获取外接矩形左上点坐标和偏移量
        index_Y = int(float(elems[2]))
        index_X = int(float(elems[3]))
        dy = int(float(elems[4]))
        dx = int(float(elems[5]))
        # 获取标注坐标信息
        line = csv1[i]
        line = line[1:(len(line) - 2)]
        coordinates = line.strip().split('Point:')[1:]
        n = len(coordinates)
        points = []
        for j in range(n):
            coordinate = coordinates[j].strip().split(',')

            x = np.int(float(coordinate[0]))
            y = np.int(float(coordinate[1]))

            points.append(np.array([x, y]))
        points = np.stack(points)

        # 获取外接矩形中心点
        rect = [index_Y, index_X, dy, dx]
        # Save
        Labels.append(label)
        Shapes.append(shape)
        Rects.append(rect)
        Contours.append(points)

    return Labels, Shapes, Rects, Contours


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
        points = []
        Coordinates = Annotation[0]
        for Coordinate in Coordinates:
            points.append([int(float(Coordinate.get('X'))), int(float(Coordinate.get('Y')))])
        points = np.stack(points)
        Contours.append(points)
    return Labels, Shapes, IndexOrigin, Contours
