# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: AidedDiagnosisSystem
@File: analyse_top10_results.py
@Date: 2020/9/17 
@Time: 10:12
@Desc:
'''
import os
import doctest
from lxml import etree


def parse_single_xml(xml_dir):
    xml = etree.parse(xml_dir)
    # get tile tag
    tile_tag = []
    for anno in xml.xpath('//Annotation'):
        tile_tag.append(anno.get('PartOfGroup').split('_')[0])
    # get sld tag
    sld_tag = None
    anno = xml.xpath('//Group')
    if len(anno)-1:
        sld_tag = anno[1].get('PartOfGroup')
    return tile_tag, sld_tag


def tag2label():
    pass


if __name__ == '__main__':
    xml_root = r'F:\LiuSibo\Codes\Projects\AidedDiagnosisSystem\doc\manu_yjy\local_xml'
    xml_names = [n for n in os.listdir(xml_root) if '.xml' in n]
    xml_dirs = list(map(lambda x: os.path.join(xml_root, x), xml_names))

    tags = dict(zip(['tiles', 'slide'], [[], []]))
    for xml_dir in xml_dirs:
        tile_tag, sld_tag = parse_single_xml(xml_dir)
        tags['tiles'].append(tile_tag)
        tags['slide'].append(sld_tag)

