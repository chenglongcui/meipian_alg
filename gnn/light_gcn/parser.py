# -*- coding: utf-8 -*-
'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
# import argparse
import tensorflow as tf


def parse_args():
    # parser = argparse.ArgumentParser(description="Run light_gcn.")
    # parser.add_argument('--weights_path', nargs='?', default='',
    #                     help='Store model path.')
    # # parser.add_argument('--data_path', nargs='?',
    # #                     default='oss://ivwen-recsys.oss-cn-shanghai-internal.aliyuncs.com/experiment/light_gcn/data/',
    # #                     help='Input data path.')
    #
    # parser.add_argument('--data_path', nargs='?',
    #                     default='Data/',
    #                     help='Input data path.')
    # parser.add_argument('--proj_path', nargs='?', default='',
    #                     help='Project path.')
    #
    # parser.add_argument('--dataset', nargs='?', default='gowalla',
    #                     help='Choose a dataset from {gowalla, yelp2018, amazon-book}')
    #
    # parser.add_argument('--pretrain', type=int, default=0,
    #                     help='0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models.')
    # parser.add_argument('--verbose', type=int, default=1,
    #                     help='Interval of evaluation.')
    # parser.add_argument('--is_norm', type=int, default=1,
    #                     help='Interval of evaluation.')
    # parser.add_argument('--epoch', type=int, default=1000,
    #                     help='Number of epoch.')
    #
    # parser.add_argument('--embed_size', type=int, default=64,
    #                     help='Embedding size.')
    # parser.add_argument('--layer_size', nargs='?', default='[64, 64, 64, 64]',
    #                     help='Output sizes of every layer')
    # parser.add_argument('--batch_size', type=int, default=1024,
    #                     help='Batch size.')
    #
    # parser.add_argument('--regs', nargs='?', default='[1e-5,1e-5,1e-2]',
    #                     help='Regularizations.')
    # parser.add_argument('--lr', type=float, default=0.01,
    #                     help='Learning rate.')
    #
    # parser.add_argument('--model_type', nargs='?', default='lightgcn',
    #                     help='Specify the name of model (lightgcn).')
    # parser.add_argument('--adj_type', nargs='?', default='pre',
    #                     help='Specify the type of the adjacency (laplacian) matrix from {plain, norm, mean}.')
    # parser.add_argument('--alg_type', nargs='?', default='lightgcn',
    #                     help='Specify the type of the graph convolutional layer from {ngcf, gcn, gcmc}.')
    #
    # parser.add_argument('--gpu_id', type=int, default=0,
    #                     help='0 for NAIS_prod, 1 for NAIS_concat')
    #
    # parser.add_argument('--node_dropout_flag', type=int, default=0,
    #                     help='0: Disable node dropout, 1: Activate node dropout')
    # parser.add_argument('--node_dropout', nargs='?', default='[0.1]',
    #                     help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    # parser.add_argument('--mess_dropout', nargs='?', default='[0.1]',
    #                     help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    #
    # parser.add_argument('--Ks', nargs='?', default='[20]',
    #                     help='Top k(s) recommend')
    #
    # parser.add_argument('--save_flag', type=int, default=0,
    #                     help='0: Disable model saver, 1: Activate model saver')
    #
    # parser.add_argument('--test_flag', nargs='?', default='part',
    #                     help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
    #
    # parser.add_argument('--report', type=int, default=0,
    #                     help='0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels')

    # 方便在PAI上调试
    tf.flags.DEFINE_string("weights_path", "", "Store model path.")
    # tf.flags.DEFINE_string("data_path", "Data/", "local Input data path.")
    tf.flags.DEFINE_string("data_path",
                           "oss://ivwen-recsys.oss-cn-shanghai-internal.aliyuncs.com/experiment/light_gcn/data/",
                           "OSS Input data path.")
    tf.flags.DEFINE_string("proj_path", "", "Project path.")
    tf.flags.DEFINE_string("dataset", "gowalla", "Choose a dataset from {gowalla, yelp2018, amazon-book}")

    tf.flags.DEFINE_integer("pretrain", 0,
                            "0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models.")
    tf.flags.DEFINE_integer("verbose", 1, "Interval of evaluation.")
    tf.flags.DEFINE_integer("is_norm", 1, "Interval of evaluation.")
    tf.flags.DEFINE_integer("epoch", 100, "Number of epoch.")
    tf.flags.DEFINE_integer("embed_size", 64, "Embedding size.")
    tf.flags.DEFINE_string("layer_size", "[64, 64, 64, 64]", "Output sizes of every layer")
    tf.flags.DEFINE_integer("batch_size", 1024, "Batch size.")
    tf.flags.DEFINE_string("regs", "[1e-5,1e-5,1e-2]", "Regularizations.")
    tf.flags.DEFINE_float("lr", 0.001, "Learning rate.")

    tf.flags.DEFINE_string("model_type", "lightgcn", "Specify the name of model (lightgcn).")
    tf.flags.DEFINE_string("adj_type", "pre",
                           "Specify the type of the adjacency (laplacian) matrix from {plain, norm, mean}.")
    tf.flags.DEFINE_string("alg_type", "lightgcn",
                           "Specify the type of the graph convolutional layer from {ngcf, gcn, gcmc}.")
    tf.flags.DEFINE_integer("gpu_id", 0, "0 for NAIS_prod, 1 for NAIS_concat")
    tf.flags.DEFINE_integer("node_dropout_flag", 0, "0: Disable node dropout, 1: Activate node dropout")
    tf.flags.DEFINE_string("node_dropout", "[0.1]",
                           "Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.")
    tf.flags.DEFINE_string("mess_dropout", "[0.1]",
                           "Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.")
    tf.flags.DEFINE_string("Ks", "[20]", "Top k(s) recommend")
    tf.flags.DEFINE_integer("save_flag", 0, "0: Disable model saver, 1: Activate model saver")
    tf.flags.DEFINE_string("test_flag", "part",
                           "Specify the test type from {part, full}, indicating whether the reference is done in mini-batch")
    tf.flags.DEFINE_integer("report", 0,
                            "0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels")

    FLAGS = tf.flags.FLAGS

    return FLAGS
