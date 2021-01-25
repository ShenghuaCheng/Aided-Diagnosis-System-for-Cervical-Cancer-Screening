# -*- coding: utf-8 -*-

import numpy as np
import cv2
import scipy.ndimage as ndi
from openslide import OpenSlide
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
def SortSp(csv_lines1, csv_lines2):
    csv_lines_new = [[y,x] for y,x in zip(csv_lines1, csv_lines2) if 'ASCH' in x or 'ASCUS' in x or 'HSIL' in x or 'LSIL' in x or 'pos' in x]
    csv_lines1_new = [y for y,x in csv_lines_new]
    csv_lines2_new = [x for y,x in csv_lines_new]
    return csv_lines1_new, csv_lines2_new

def Generate_labelmap_csv(pathfolder_csv, pathfolder_mrxs,pathfolder_Labelmap, filename, leveltemp, Rate_Exchange):
    path1 = pathfolder_csv + filename + '/file1.csv'
    path2 = pathfolder_csv + filename + '/file2.csv'
    pathmrxs = pathfolder_mrxs + filename + '.svs'

    ors = OpenSlide(pathmrxs)
    sizeMask= ors.level_dimensions[leveltemp]
    sizeMask = sizeMask[::-1]
    csv_lines1 = open(path1,"r").readlines()
    csv_lines2 = open(path2,"r").readlines()
#    return csv_lines2
    csv_lines1, csv_lines2 = SortSp(csv_lines1, csv_lines2)
    imgMaskk= np.zeros((sizeMask[0], sizeMask[1]), dtype=np.uint8)
    #######填充轮廓
    for i in range(0, len(csv_lines1)):
        line = csv_lines2[i]
        elems = line.strip().split(',')
        label = elems[0]
        label1 = elems[1]
        label1 = label1.strip().split(' ')
        label1 = label1[0]
        line = csv_lines1[i]
        line = line[1:(len(line)-2)]
        elems = line.strip().split('Point:')
        if label1 == "Ellipse":
            n = len(elems)
            points = [0]*(n-1)
            for j in range(1, n):
                s = elems[j]
                s1 = s.strip().split(',')
                x = int(np.round(float(s1[0])/(Rate_Exchange**leveltemp)))
                y = int(np.round(float(s1[1])/(Rate_Exchange**leveltemp)))
                points[j-1] = [x, y]
            points = np.stack(points)
            center = (int(np.mean(points[:,0])), int(np.mean(points[:,1])))
            axes = (int(np.max(points[:,0])/2-np.min(points[:,0])/2), int(np.max(points[:,1])/2-np.min(points[:,1])/2))
        elif label1 == "Polygon" or label1 == "Rectangle":
            n = len(elems)
            points = [0]*(n-1)
            for j in range(1, n):
                s = elems[j]
                s1 = s.strip().split(',')
                x = int(np.round(float(s1[0])/(Rate_Exchange**leveltemp)))
                y = int(np.round(float(s1[1])/(Rate_Exchange**leveltemp)))
                points[j-1] = [x, y]
        if label1 == "Polygon" or label1 == "Rectangle":
            cv2.fillPoly(imgMaskk, [np.stack(points)], color=255)
        elif label1 == "Ellipse":
            pts = cv2.ellipse2Poly(center, axes, angle=0, arcStart=0, arcEnd=360, delta=5)
            cv2.fillPoly(imgMaskk, [pts], color=255)
    cv2.imwrite(pathfolder_Labelmap + filename +".tif", imgMaskk)
    print('Complete labelmap:' + filename)


if __name__ == '__main__':
    leveltemp = 2
    Rate_Exchange = 4#
    pathfolder_csv = 'H:/TCTDATA/our/LabelFiles/csv_Shengfuyou_4th/'
    pathfolder_mrxs = 'H:/TCTDATA/our/Positive/Shengfuyou_4th/'
    pathfolder_Labelmap = 'H:/TCTDATA/our/make_sample/Labelmap/Shengfuyou_4th/'
    filelist = os.listdir(pathfolder_mrxs)
    filelist = [cz for cz in filelist if '.svs' in cz]
    label = []
    pp = []
    for item in  tqdm(filelist):
        try:
            filename = item[:-4]#mrxs:-5 svs:-4
            Generate_labelmap_csv(pathfolder_csv, pathfolder_mrxs,pathfolder_Labelmap, filename, leveltemp, Rate_Exchange)
# =============================================================================
#             label = [x.split(',')[0] for x in label]
#             label = np.array(label)
#             if np.sum(label =='null')!=0:
#                 pp.append(item)
# =============================================================================
        except:
            print('wrong'+filename)
            
# =============================================================================
#     label = [x.split(',')[0] for x in label]
#     label = np.array(label)
#     print(np.unique(label))
# =============================================================================
