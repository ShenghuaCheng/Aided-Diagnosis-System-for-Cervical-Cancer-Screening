# -*- coding:utf-8 -*-

class Args:
    sld_root = r""
    result_root = r""

    res = {"A": 0.18, "B": 0.293, "C": 0.243}
    post_fix = {"A": ".sdpc", "B": ".svs", "C": ".mrxs"}

    fld_dict = {
        "A": [
        ],
        "B": [
        ]
        # ......
    }

    labeled = False
    circle_r = 250

    aim_res = 0.243
    aim_size = 384
    top_n = 100

    save_root = r"\top100"
