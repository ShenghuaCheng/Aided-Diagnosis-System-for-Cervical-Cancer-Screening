# -*- coding:utf-8 -*-
import platform
from .sdpcreader.sdpc_reader import Sdpc
SYSTEM_TYPE = platform.system()
if SYSTEM_TYPE == "Windows":
    from .srpreader.srp_python_win.pysrp import Srp
elif SYSTEM_TYPE == "Linux":
    from .srpreader.srp_python_linux.pysrp import Srp
