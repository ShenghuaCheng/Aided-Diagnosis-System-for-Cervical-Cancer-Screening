# -*- coding: utf-8 -*-
"""
This file is a part of project "Aided-Diagnosis-System-for-Cervical-Cancer-Screening".
See https://github.com/ShenghuaCheng/Aided-Diagnosis-System-for-Cervical-Cancer-Screening for more information.

File name: convert_to_pb
Description: convert model weight to pb.
"""

import os
import argparse
from loguru import logger

from tensorflow.python.framework import graph_io
from keras import backend as K

from core.utils import load_weights, freeze_session
from core.models import wsi_clf, model2, model1
from core.common import check_dir

MODEL_DICT = {
    "model1": model1,
    "model2": model2,
    "wsi_clf_top10": wsi_clf(10),
    "wsi_clf_top20": wsi_clf(20),
    "wsi_clf_top30": wsi_clf(30),
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="This file convert model to pb.")
    parser.add_argument("-m", "--model_name", type=str, help="model name.")
    parser.add_argument("-w", "--weights", type=str, nargs=argparse.ONE_OR_MORE, help="model weight.")
    parser.add_argument("-o", "--output", type=str, help="output path.")
    args = parser.parse_args()

    model_name = args.model_name
    weights = args.weights
    output = args.output
    pb_filename = os.path.basename(output)
    if not pb_filename.endswith(".pb"):
        raise ValueError("output should be a pb file")
    pb_filedir = os.path.dirname(output)
    check_dir(pb_filedir, True, logger)

    logger.info(f"creating {model_name} ...")
    K.set_learning_phase(0)
    model = MODEL_DICT[args.model_name]()
    model = load_weights(model, weights)

    input_layers = [l.name.split(':')[0] for l in model.inputs]
    output_layers = [l.name.split(':')[0] for l in model.outputs]
    logger.info(f"model inputs: {', '.join(input_layers)}")
    logger.info(f"model outputs: {', '.join(output_layers)}")

    frozen_graph = freeze_session(K.get_session(), output_names=output_layers)

    graph_io.write_graph(frozen_graph, pb_filedir, pb_filename, as_text=False)
    logger.info(f"saved the constant graph (ready for inference) at:\n{output}")



