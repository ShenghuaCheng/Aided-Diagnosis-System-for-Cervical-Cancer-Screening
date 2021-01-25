# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: AidedDiagnosisSystem
@File: get_exp4_top20_results.py
@Date: 2020/10/5 
@Time: 16:12
@Desc: 获取医生各个账号的对推荐区域的判读情况，同一医生的posCheck和AllCheck算作两份账号
'''
import os
import json
import pandas as pd
import xml.etree.ElementTree as ET
from collections import namedtuple
from functools import partial

Slide = namedtuple('Slide', 'name tops_tag')


def parse_single_xml(xml_dir, debug=False):
    sld_ck = ET.parse(xml_dir)
    asap_annos = sld_ck.getroot()
    tags = []
    for anno in asap_annos.findall('Annotation'):
        if debug:
            tag = anno.get('State')
            # tag = anno.get('PartOfGroup')
        else:
            s = anno.get('State')
            if s == 'checked':
                tag = 1
            elif s == 'delete':
                tag = 0
            else:
                tag = 2
        tags.append(tag)
    return tags


if __name__ == '__main__':
    debug = False
    name = 'allCheck5'
    xml_root = r'I:\20200929_EXP4\Slides\local_xml_split_all145_pos145\%s' % name
    save_excel = r'I:\20200929_EXP4\Slides\local_xml_split_all145_pos145\%s.xlsx' % name
    # 获取xml中的内容
    xml_names = [xml.split('.')[0] for xml in os.listdir(xml_root)]
    func_get_dir = partial(os.path.join, xml_root)
    func_parse = lambda x: parse_single_xml(func_get_dir(x), debug=debug)
    slds_tags = list(map(func_parse, os.listdir(xml_root)))
    # 获取切片的类别信息
    labels_dir = r'I:\20200929_EXP4\Slides\EF_sld_labels_updateposcheck5.json'
    labels = json.load(open(labels_dir, 'r'))
    labels = dict(zip(labels['name'], labels['label']))

    if debug:
        # 检查tag里的内容
        a = []
        for t in slds_tags:
            a+=t
        a = set(a)
        print(a)
    else:
        # cvt results to dataframe
        summary_df = {}
        detail_df = {}
        for sld, tags in zip(xml_names, slds_tags):
            lab = labels[sld + '.sdpc']
            if lab == 0:  # 去掉更新标注后标签为0的切片
                continue
            summary_df[sld] = {}
            detail_df[sld] = {}
            summary_df[sld]['Label'] = lab
            detail_df[sld]['Label'] = lab
            # exclude negative slides
            # if
            if 2 in tags:
                summary_df[sld]['Finish'] = False
                detail_df[sld]['Finish'] = False
            else:
                summary_df[sld]['Finish'] = True
                detail_df[sld]['Finish'] = True
            detail_df[sld]['nb_uncheck'] = len([i for i, t in enumerate(tags) if t == 2])
            detail_df[sld]['nb_checked'] = len([i for i, t in enumerate(tags) if t == 1])
            detail_df[sld]['nb_deleted'] = len([i for i, t in enumerate(tags) if t == 0])
            detail_df[sld]['id_uncheck'] = [i for i, t in enumerate(tags) if t == 2]
            detail_df[sld]['id_checked'] = [i for i, t in enumerate(tags) if t == 1]
            detail_df[sld]['id_deleted'] = [i for i, t in enumerate(tags) if t == 0]
            summary_df[sld]['Top10'] = len([i for i, t in enumerate(tags[:10]) if t == 1])
            summary_df[sld]['Top20'] = len([i for i, t in enumerate(tags) if t == 1])
        summary_df = pd.DataFrame(summary_df)
        detail_df = pd.DataFrame(detail_df)
        summary_df = summary_df.T
        detail_df = detail_df.T
        # wrt to excel
        excel_wrt = pd.ExcelWriter(save_excel)
        summary_df.to_excel(excel_wrt, sheet_name=name+'_sumary')
        detail_df.to_excel(excel_wrt, sheet_name=name+'_detail')
        excel_wrt.close()
