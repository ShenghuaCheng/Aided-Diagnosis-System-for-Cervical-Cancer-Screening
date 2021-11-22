# -*- coding: utf-8 -*-
"""
This file is a part of project "Aided-Diagnosis-System-for-Cervical-Cancer-Screening".
See https://github.com/ShenghuaCheng/Aided-Diagnosis-System-for-Cervical-Cancer-Screening for more information.

File name: model_weight
Description: function for model operation.
"""

from collections import Iterable

from loguru import logger

import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants

__all__ = [
    "load_weights",
    "freeze_session",
    "pb_forward"
]


def load_weights(model, weights=None):
    """ load weight for model, inplace opt
    """
    if weights is None:
        return model
    elif isinstance(weights, str):
        model.load_weights(weights, by_name=True)
        logger.info(f"load {weights}")
    elif isinstance(weights, Iterable):
        for w in weights:
            model = load_weights(model, w)
    return model


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a prunned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    prunned so subgraphs that are not neccesary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # print(len(output_names))
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


def pb_forward(pb_dir, input_name, output_name, data):
    output_graph_def = tf.GraphDef()
    with open(pb_dir, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    input_x = sess.graph.get_tensor_by_name(input_name)
    output_prob = sess.graph.get_tensor_by_name(output_name)
    preds = []
    for val in data:
        input_val = val
        pred = sess.run(output_prob, feed_dict={input_x: np.expand_dims(input_val, 0)})
        preds.append(pred)
    return preds
