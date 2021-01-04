# -*- coding: utf-8 -*-
import tensorflow as tf
from two_tower_model_fr_v1 import TwoTowerModelFRV1
# 可以单独用它生成 timeline，也可以使用下面两个对象生成 timeline
from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import option_builder
from tensorflow.python.client import timeline
import utils
import time

print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

tf.flags.DEFINE_string("train_file_dir", "", "train_file")
tf.flags.DEFINE_string("pred_file_dir", "", "pred_file_dir")
tf.flags.DEFINE_float("learning_rate", 0.01, "learning_rate")
tf.flags.DEFINE_integer("batch_size", 32, "batch size")
tf.flags.DEFINE_integer("item_embedding_size", 32, "embedding size")
tf.flags.DEFINE_integer("cate_embedding_size", 16, "embedding size")
tf.flags.DEFINE_integer("tag_embedding_size", 8, "embedding size")
tf.flags.DEFINE_integer("is_train", 1, "1:training stage 0:predicting stage")
tf.flags.DEFINE_integer("local", 1, "1:local 0:online")
tf.flags.DEFINE_string("output_table", "", "output table name in MaxComputer")
tf.flags.DEFINE_string("checkpointDir", "", "checkpointDir")
tf.flags.DEFINE_string("saved_model_dir", "", "saved model dir")
tf.flags.DEFINE_string("buckets", "", "oss host")
tf.flags.DEFINE_string("recall_cnt_file", "", "recall_cnt_file")
tf.flags.DEFINE_string("item_tower_file", "", "item_tower_file")
tf.flags.DEFINE_string("top_k_num", "", "number of top k")
tf.flags.DEFINE_string("neg_sample_num", "", "number of top k")

FLAGS = tf.flags.FLAGS


def main(_):
    # Data File
    train_file_dir = FLAGS.train_file_dir
    pred_file_dir = FLAGS.pred_file_dir
    item_tower_file = FLAGS.item_tower_file

    # Hyper Parameters
    learning_rate = FLAGS.learning_rate
    batch_size = FLAGS.batch_size
    item_embedding_size = FLAGS.item_embedding_size
    cate_embedding_size = FLAGS.cate_embedding_size
    tag_embedding_size = FLAGS.tag_embedding_size
    is_train = FLAGS.is_train
    output_table = FLAGS.output_table
    saved_model_dir = FLAGS.saved_model_dir
    checkpoint_dir = FLAGS.checkpointDir
    oss_bucket_dir = FLAGS.buckets
    local = FLAGS.local
    recall_cnt_file = FLAGS.recall_cnt_file
    top_k_num = int(FLAGS.top_k_num)
    neg_sample_num = int(FLAGS.neg_sample_num)
    print("train_file_dir: %s" % train_file_dir)
    print("pred_file_dir: %s" % pred_file_dir)
    print("is_train: %d" % is_train)
    print("learning_rate: %f" % learning_rate)
    print("item_embedding_size: %d" % item_embedding_size)
    print("cate_embedding_size: %d" % cate_embedding_size)
    print("tag_embedding_size: %d" % tag_embedding_size)
    print("batch_size: %d" % batch_size)
    print("output table name: %s " % output_table)
    print("checkpoint_dir: %s " % checkpoint_dir)
    print("oss bucket dir: %s" % oss_bucket_dir)
    print("recall_cnt_file: %s" % recall_cnt_file)

    if local:
        # summary_dir = "../summary/"
        # recall_cnt_file = "../data/youtube_recall_item_cnt*"
        pass
    else:
        # oss_bucket_dir = "oss://ivwen-recsys.oss-cn-shanghai-internal.aliyuncs.com/"
        # summary_dir = oss_bucket_dir + "experiment/summary/"
        train_file_dir = oss_bucket_dir + train_file_dir
        pred_file_dir = oss_bucket_dir + pred_file_dir
        recall_cnt_file = oss_bucket_dir + recall_cnt_file
        item_tower_file = oss_bucket_dir + item_tower_file
        saved_model_dir = oss_bucket_dir + saved_model_dir

    # get item cnt
    item_count, cate_count, tag_count = utils.get_item_cnt(recall_cnt_file)
    item_tower_file = [utils.get_file_name(item_tower_file)]
    print("item tower file: %s" % item_tower_file)
    print("item_count: ", item_count)
    print("cate_count: ", cate_count)
    print("tag_count: ", tag_count)
    print("saved_model_dir: %s " % saved_model_dir)
    # GPU config
    # gpu_config = tf.ConfigProto()
    # gpu_config.gpu_options.allow_growth = True
    #

    with tf.Session() as sess:
        train_file_name = utils.get_file_name(train_file_dir)
        pred_file_name = utils.get_file_name(pred_file_dir)

        two_tower_model = TwoTowerModelFRV1(train_file_dir=train_file_name,
                                            pred_file_dir=pred_file_name,
                                            item_tower_file=item_tower_file,
                                            is_train=is_train,
                                            item_embedding_size=item_embedding_size,
                                            cate_embedding_size=cate_embedding_size,
                                            tag_embedding_size=tag_embedding_size,
                                            batch_size=batch_size,
                                            learning_rate=learning_rate,
                                            local=local,
                                            item_count=item_count,
                                            cate_count=cate_count,
                                            tag_count=tag_count,
                                            output_table=output_table,
                                            top_k_num=top_k_num,
                                            neg_sample_num=neg_sample_num,
                                            sess=sess
                                            )

        two_tower_model.restore_model(sess, checkpoint_dir)
        print("restore model finished!!")
        two_tower_model.save_model_as_savedmodel(sess=sess,
                                                 dir=saved_model_dir,
                                                 inputs=two_tower_model.saved_model_inputs,
                                                 outputs=two_tower_model.saved_model_outputs)

        # if is_train == 1:
        #     print("time: %s\tckpt model save start..." % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        #     two_tower_model.save_model(sess=sess, path=checkpoint_dir)
        #     print("time: %s\tsave_model save start..." % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        #     two_tower_model.save_model_as_savedmodel(sess=sess,
        #                                              dir=saved_model_dir,
        #                                              inputs=two_tower_model.saved_model_inputs,
        #                                              outputs=two_tower_model.saved_model_outputs)


if __name__ == "__main__":
    tf.app.run()
