# -*- coding: utf-8 -*-
"""
This file is a part of project "Aided-Diagnosis-System-for-Cervical-Cancer-Screening".
See https://github.com/ShenghuaCheng/Aided-Diagnosis-System-for-Cervical-Cancer-Screening for more information.

File name: load_config
Description: load config file for training.
"""
import importlib
import os
import sys

__all__ = [
    "load_config"
]

DEFAULT_CONFIG = {
    "model1-cls": "model1_cls.py",
    "model1-loc": "model1_loc.py",
    "model2-cls": "model2_cls.py",
    "wsi-cls": "wsi_cls.py",
}


def load_config_file(cfg_file):
    try:
        sys.path.append(os.path.dirname(cfg_file))
        config = importlib.import_module(os.path.basename(cfg_file).split(".")[0])
        config = config.Exp()
    except Exception:
        raise ImportError("{} doesn't contains class named 'Exp'".format(cfg_file))
    return config


def load_config_name(cfg_name):
    import core
    proj_dir = os.path.dirname(os.path.dirname(core.__file__))
    cfg_file = os.path.join(proj_dir, "configs", DEFAULT_CONFIG[cfg_name])
    return load_config_file(cfg_file)


def load_config(cfg_file, cfg_name):
    """Get Config object by file or name.
    """
    assert (cfg_file is not None or cfg_name is not None), "Config should be provided."
    if cfg_file is None:
        return load_config_name(cfg_name)
    else:
        return load_config_file(cfg_file)
