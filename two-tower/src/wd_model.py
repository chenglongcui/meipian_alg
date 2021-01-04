# encoding:utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import shutil
import sys
import importlib
import os
import pandas as pd  ### make from env problem
import tensorflow as tf
from tensorflow.python.feature_column.feature_column import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

##define input line field
##--from sample_v5
_CSV_COLUMNS = [
    "label", "user_id", "user_type", "item_id", "item_type", "class_id", "tag_ids", "author_id", "act_hour",
    "act_isweekend", "act_noon", "act_day", "dev_type", "dev_brand", "dev_brand_type", "dev_brand_level", "dev_os",
    "dev_net", "client_type", "dev_carrier", "dev_lang", "dev_loc", "dev_ip_short", "location_id", "user_city",
    "user_city_level", "class_sab", "item_age", "click_seq_50size", "gender", "age", "consume_level", "career",
    "education", "user_app_list", "user_set_cates", "unclick_seq_50size"]
_CSV_COLUMN_DEFAULTS = [['x'] for x in _CSV_COLUMNS]

_CSV_COLUMNS_EXP = ["ic1_ctr", "ic3_ctr", "ic7_ctr", "ic14_ctr", "ic30_ctr", "ic60_ctr", "uc1_ctr", "uc3_ctr",
                    "uc7_ctr", "uc14_ctr", "uc30_ctr"]
_CSV_COLUMNS.extend(_CSV_COLUMNS_EXP)
_CSV_COLUMN_DEFAULTS.extend([[0.0] for x in _CSV_COLUMNS_EXP])

##define model feature for export model
_FEATURES = ['act_hour', 'act_day', 'act_isweekend', 'act_noon',
             'dev_loc', 'dev_ip_short',
             'dev_type', 'dev_brand', 'dev_brand_type', 'dev_os', 'dev_net', 'dev_carrier',
             'user_id', 'click_seq_50size', 'unclick_seq_50size',
             'item_id', 'class_id', 'tag_ids',
             'client_type', 'user_app_list', 'age', 'gender', 'consume_level', 'user_type',
             'location_id']
_TYPES = [tf.string for x in _FEATURES]

_FEATURES_EXP = ["ic1_ctr", "ic3_ctr", "ic7_ctr", "ic14_ctr", "ic30_ctr", "ic60_ctr", "uc1_ctr", "uc3_ctr", "uc7_ctr",
                 "uc14_ctr", "uc30_ctr"]
_FEATURES.extend(_FEATURES_EXP)
_TYPES.extend([tf.float32 for x in _FEATURES_EXP])


def c_ctr_norm(x):
    return x + 1.0001


def input_fn(filename_queue, num_epochs=1, batch_size=256, perform_shuffle=False):
    def parse_csv(value):
        columns = tf.decode_csv(value, field_delim='\t', record_defaults=_CSV_COLUMN_DEFAULTS)  ##set default value
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('label')
        classes = tf.equal(labels, '1')
        classes = tf.to_float(classes)
        return features, classes

    dataset = tf.data.TextLineDataset(filename_queue)
    dataset = dataset.prefetch(buffer_size=batch_size * 100)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size * 10)
    dataset = dataset.map(parse_csv, num_parallel_calls=10)

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    features, label = iterator.get_next()
    return features, label


def build_model_columns():
    ##"build feature column"
    pos_voclist = ['x']
    pos_voclist.extend([str(x) for x in range(50)])
    dict_location_id = categorical_column_with_vocabulary_list('location_id', pos_voclist, num_oov_buckets=0,
                                                               dtype=tf.string)

    hour_voclist = ['x']
    hour_voclist.extend([str(x) for x in range(24)])
    dict_act_hour = categorical_column_with_vocabulary_list('act_hour', hour_voclist, num_oov_buckets=0,
                                                            dtype=tf.string)
    day_voclist = ['x', '0', '1', '2', '3', '4', '5', '6', '7']
    dict_act_day = categorical_column_with_vocabulary_list('act_day', hour_voclist, num_oov_buckets=0, dtype=tf.string)
    noon_voclist = ['x', '0', '1', '2', '3', '4']
    dict_act_noon = categorical_column_with_vocabulary_list('act_noon', noon_voclist, num_oov_buckets=0,
                                                            dtype=tf.string)
    weekend_voclist = ['x', '0', '1']
    dict_act_weekend = categorical_column_with_vocabulary_list('act_isweekend', weekend_voclist, num_oov_buckets=0,
                                                               dtype=tf.string)

    devtype_voclist = ['x', '0', '1', '2', '3']
    dict_devtype = categorical_column_with_vocabulary_list('dev_type', devtype_voclist, num_oov_buckets=0,
                                                           dtype=tf.string)
    devnet_voclist = ['x', '0', '1', '2', '3', '4', '5']
    dict_devnet = categorical_column_with_vocabulary_list('dev_net', devnet_voclist, num_oov_buckets=0, dtype=tf.string)
    devcarrier_voclist = ['x', '0', '1', '2', '3', '4', '5']
    dict_devcarrier = categorical_column_with_vocabulary_list('dev_carrier', devcarrier_voclist, num_oov_buckets=0,
                                                              dtype=tf.string)
    clienttype_voclist = ['x', '0', '1']
    dict_clienttype = categorical_column_with_vocabulary_list('client_type', clienttype_voclist, num_oov_buckets=0,
                                                              dtype=tf.string)
    devbrandtype_voclist = ['x']
    devbrandtype_voclist.extend([str(x) for x in range(24)])
    dict_devbrandtype = categorical_column_with_vocabulary_list('dev_brand_type', devbrandtype_voclist,
                                                                num_oov_buckets=0, dtype=tf.string)
    # hash_devos = categorical_column_with_hash_bucket('dev_os', hash_bucket_size=1000, dtype=tf.string)
    hash_devloc = categorical_column_with_hash_bucket('dev_loc', hash_bucket_size=10000, dtype=tf.string)
    hash_devip = categorical_column_with_hash_bucket('dev_ip_short', hash_bucket_size=100000, dtype=tf.string)

    item_bucksize = 1000000
    hash_click_seq_fixed = categorical_column_with_hash_bucket("click_seq_50size", hash_bucket_size=item_bucksize,
                                                               dtype=tf.string)
    hash_unclick_seq_fixed = categorical_column_with_hash_bucket("unclick_seq_50size", hash_bucket_size=item_bucksize,
                                                                 dtype=tf.string)

    user_bucksize = 5000000
    hash_user_id = categorical_column_with_hash_bucket("user_id", hash_bucket_size=user_bucksize, dtype=tf.string)
    hash_applist = categorical_column_with_hash_bucket("user_app_list", hash_bucket_size=1000, dtype=tf.string)
    # hash_set_cates = categorical_column_with_hash_bucket("user_set_cates", hash_bucket_size=3000, dtype=tf.string)

    usertype_voclist = ['x', '0', '1']
    dict_usertype = categorical_column_with_vocabulary_list('user_type', usertype_voclist, num_oov_buckets=0,
                                                            dtype=tf.string)
    clevel_voclist = ['x', '0', '1', '2', '-1']
    dict_clevel = categorical_column_with_vocabulary_list('consume_level', clevel_voclist, num_oov_buckets=0,
                                                          dtype=tf.string)
    age_voclist = ['x', '0', '1', '2', '3', '4', '-1']
    dict_age = categorical_column_with_vocabulary_list('age', age_voclist, num_oov_buckets=0, dtype=tf.string)
    gender_voclist = ['x', '0', '1', '-1']
    dict_gender = categorical_column_with_vocabulary_list('gender', gender_voclist, num_oov_buckets=0, dtype=tf.string)

    uc30_ctr = tf.feature_column.numeric_column('uc30_ctr', normalizer_fn=c_ctr_norm)
    uc14_ctr = tf.feature_column.numeric_column('uc14_ctr', normalizer_fn=c_ctr_norm)
    uc7_ctr = tf.feature_column.numeric_column('uc7_ctr', normalizer_fn=c_ctr_norm)
    uc3_ctr = tf.feature_column.numeric_column('uc3_ctr', normalizer_fn=c_ctr_norm)
    uc1_ctr = tf.feature_column.numeric_column('uc1_ctr', normalizer_fn=c_ctr_norm)

    hash_item_id = categorical_column_with_hash_bucket("item_id", hash_bucket_size=item_bucksize, dtype=tf.string)
    hash_tag_ids = categorical_column_with_hash_bucket("tag_ids", hash_bucket_size=50000, dtype=tf.string)
    hash_class_id = categorical_column_with_hash_bucket("class_id", hash_bucket_size=3000, dtype=tf.string)

    ic60_ctr = tf.feature_column.numeric_column('ic60_ctr', normalizer_fn=c_ctr_norm)
    ic30_ctr = tf.feature_column.numeric_column('ic30_ctr', normalizer_fn=c_ctr_norm)
    ic14_ctr = tf.feature_column.numeric_column('ic14_ctr', normalizer_fn=c_ctr_norm)
    ic7_ctr = tf.feature_column.numeric_column('ic7_ctr', normalizer_fn=c_ctr_norm)
    ic3_ctr = tf.feature_column.numeric_column('ic3_ctr', normalizer_fn=c_ctr_norm)
    ic1_ctr = tf.feature_column.numeric_column('ic1_ctr', normalizer_fn=c_ctr_norm)

    ##"@sparse column: build wide input field"
    wide_columns = []

    wide_columns.append(dict_act_hour)
    wide_columns.append(dict_act_day)
    wide_columns.append(dict_act_noon)
    wide_columns.append(dict_act_weekend)
    wide_columns.append(dict_devtype)
    wide_columns.append(dict_devnet)
    wide_columns.append(dict_devcarrier)
    wide_columns.append(dict_devbrandtype)
    # wide_columns.append(hash_devos)
    wide_columns.append(hash_devloc)
    wide_columns.append(hash_devip)

    wide_columns.append(hash_tag_ids)
    wide_columns.append(hash_class_id)
    wide_columns.append(hash_item_id)
    wide_columns.append(dict_location_id)

    wide_columns.append(dict_clienttype)
    wide_columns.append(dict_age)
    wide_columns.append(dict_gender)
    wide_columns.append(dict_clevel)
    wide_columns.append(dict_usertype)

    ##build cross filed
    wide_columns.append(crossed_column([dict_age, "class_id"], hash_bucket_size=1000))
    wide_columns.append(crossed_column([dict_gender, "class_id"], hash_bucket_size=500))
    wide_columns.append(crossed_column([dict_age, dict_gender, "class_id"], hash_bucket_size=5000))
    wide_columns.append(crossed_column([dict_act_noon, "class_id"], hash_bucket_size=1000))
    wide_columns.append(crossed_column([dict_act_weekend, "class_id"], hash_bucket_size=500))
    wide_columns.append(crossed_column([dict_act_noon, dict_act_weekend, "class_id"], hash_bucket_size=5000))
    wide_columns.append(crossed_column([dict_clienttype, "class_id"], hash_bucket_size=500))
    wide_columns.append(crossed_column([dict_clienttype, dict_devcarrier, "class_id"], hash_bucket_size=5000))
    wide_columns.append(crossed_column([dict_clevel, "class_id"], hash_bucket_size=1000))
    wide_columns.append(crossed_column([dict_devbrandtype, "class_id"], hash_bucket_size=10000))
    wide_columns.append(crossed_column(["dev_loc", "class_id"], hash_bucket_size=50000))

    deep_columns = []
    deep_columns.append(uc1_ctr)
    deep_columns.append(uc3_ctr)
    deep_columns.append(uc7_ctr)
    deep_columns.append(uc14_ctr)
    deep_columns.append(uc30_ctr)
    deep_columns.append(ic1_ctr)
    deep_columns.append(ic3_ctr)
    deep_columns.append(ic7_ctr)
    deep_columns.append(ic14_ctr)
    deep_columns.append(ic30_ctr)
    deep_columns.append(ic60_ctr)

    deep_columns.append(indicator_column(dict_age))
    deep_columns.append(indicator_column(dict_gender))
    deep_columns.append(embedding_column(hash_applist, dimension=8))
    deep_columns.append(embedding_column(hash_tag_ids, dimension=12))
    deep_columns.append(embedding_column(hash_class_id, dimension=6))
    # deep_columns.extend(shared_embedding_columns([hash_set_cates, hash_class_id], 6))
    deep_columns.extend(shared_embedding_columns([hash_item_id, hash_click_seq_fixed, hash_unclick_seq_fixed], 32))
    return wide_columns, deep_columns


def build_estimator(model_dir, model_type='liner_n_deep'):
    wide_columns, deep_columns = build_model_columns()
    print("use_model:%s" % (model_type))
    if 'liner_n_deep' == model_type:
        dnn_lr_combined = importlib.import_module("net.dnn_linear_combined")
        m = dnn_lr_combined.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=[1024, 256, 128])
    else:  # 'lr' == model_type:
        m = tf.estimator.LinearClassifier(
            model_dir=model_dir,
            feature_columns=wide_columns)
    return m


def export_model(model, export_dir):
    def json_serving_input_fn():
        """Build the serving inputs."""
        inputs = {}
        for feat, dtype in zip(_FEATURES, _TYPES):
            inputs[feat] = tf.placeholder(shape=[None], dtype=dtype)
        return tf.estimator.export.ServingInputReceiver(inputs, inputs)

    ##export model via saved_model
    model.export_savedmodel(export_dir, json_serving_input_fn)


def build_config():
    flags = tf.app.flags
    flags.DEFINE_string('task_type', 'train', 'type: train,eval,export')
    flags.DEFINE_string('model_type', 'liner_n_deep', 'type: lr,deep,liner_n_deep,fm_n_deep')
    flags.DEFINE_string('model_dir', '/home/work/wsx_job/tf_work/ckpt/', '')
    flags.DEFINE_string('export_dir', '/home/work/wsx_job/tf_work/export/', '')
    flags.DEFINE_string('in_file_train', '/data/ossfs/rank_ctr_sample/20190718/app_feed_rank_sample_ctr__*', '')
    flags.DEFINE_string('in_file_test', '/data/ossfs/rank_ctr_sample/20190719/app_feed_rank_sample_ctr__*', '')
    flags.DEFINE_integer('eval_steps', 1000, 'Number of steps to eval.')
    flags.DEFINE_integer('train_epoch', 2, 'Number of epoch to train.')
    return flags.FLAGS


def main(_):
    m = build_estimator(FLAGS.model_dir, FLAGS.model_type)
    print("begin task_type:%s" % (FLAGS.task_type))

    if "train" in FLAGS.task_type:
        print("start train model from[%s] via[%s]" % (FLAGS.model_dir, FLAGS.in_file_train))
        m.train(input_fn=lambda: input_fn(tf.gfile.Glob(FLAGS.in_file_train), num_epochs=FLAGS.train_epoch))
        results = m.evaluate(input_fn=lambda: input_fn(tf.gfile.Glob(FLAGS.in_file_train), perform_shuffle=True),
                             steps=FLAGS.eval_steps)
        print("Train:")
        for key in sorted(results):
            print("T-KK: %s: %s" % (key, results[key]))

    if "eval" in FLAGS.task_type:
        print("start eval model from[%s] via data[%s]" % (FLAGS.model_dir, FLAGS.in_file_test))
        results = m.evaluate(input_fn=lambda: input_fn(tf.gfile.Glob(FLAGS.in_file_test), perform_shuffle=True),
                             steps=FLAGS.eval_steps)
        print("Valid:")
        for key in sorted(results):
            print("V-KK: %s: %s" % (key, results[key]))

    if "export" in FLAGS.task_type:
        print("start export model from [%s] to[%s]" % (FLAGS.model_dir, FLAGS.export_dir))
        ret = export_model(m, FLAGS.export_dir)
        print("export model done, ret", ret)

    print("end task_type:%s" % (FLAGS.task_type))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS = build_config()
    tf.app.run()
    sys.exit(0)
