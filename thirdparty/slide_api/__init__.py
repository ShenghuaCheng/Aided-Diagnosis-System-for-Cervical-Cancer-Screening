# -*- coding: utf-8 -*-
"""
This file is a part of project "Aided-Diagnosis-System-for-Cervical-Cancer-Screening".
See https://github.com/ShenghuaCheng/Aided-Diagnosis-System-for-Cervical-Cancer-Screening for more information.

File name: __init__.py.py
Description: This module is the original API for slides in "*.sdpc" or "*.srp" format.
"""

import platform
from .sdpcreader.sdpc_reader import Sdpc
SYSTEM_TYPE = platform.system()
if SYSTEM_TYPE == "Windows":
    from .srpreader.srp_python_win.pysrp import Srp
elif SYSTEM_TYPE == "Linux":
    from .srpreader.srp_python_linux.pysrp import Srp