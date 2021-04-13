# -*- coding:utf-8 -*-
import os

from tensorflow.python.framework import graph_io
from keras.models import load_model
from keras import backend as K

from utils.h5_to_pb import freeze_session

if __name__ == "__main__":
    # ========================= setting ================================
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    input_fld = r'models'
    output_fld = r'pbs'

    multi_output_flag = True
    weight_file = '.h5'
    output_graph_name = '.pb'
    # ==================================================================
    weight_file_path = os.path.join(input_fld, weight_file)

    if not os.path.isdir(output_fld):
        os.mkdir(output_fld)

    K.set_learning_phase(0)
    net_model = load_model(weight_file_path)

    print('input is :', net_model.input.name)
    if multi_output_flag:
        for op in net_model.output:
            print('output is:', op.name)
        frozen_graph = freeze_session(K.get_session(), output_names=[op.op.name for op in net_model.output])
    else:
        print('output is:', net_model.output.name)
        frozen_graph = freeze_session(K.get_session(), output_names=[net_model.output.op.name])

    graph_io.write_graph(frozen_graph, output_fld, output_graph_name, as_text=False)

    print('saved the constant graph (ready for inference) at: ', os.path.join(output_fld, output_graph_name))