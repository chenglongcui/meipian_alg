# -*- coding: utf-8 -*-
from __future__ import print_function, division
import tensorflow as tf
from tt_ctr_model_v4 import TTFRCTRModelV4
# 可以单独用它生成 timeline，也可以使用下面两个对象生成 timeline
from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import option_builder
from tensorflow.python.client import timeline
from sklearn.metrics import roc_auc_score
import numpy as np
import utils
import time

print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

tf.flags.DEFINE_string("tables", "", "tables info")
tf.flags.DEFINE_string("test_file_dir", "", "test_file_dir")
tf.flags.DEFINE_float("learning_rate", 0.01, "learning_rate")
tf.flags.DEFINE_integer("batch_size", 128, "batch size")
tf.flags.DEFINE_integer("item_embedding_size", 64, "embedding size")
tf.flags.DEFINE_integer("cate_embedding_size", 16, "embedding size")
tf.flags.DEFINE_integer("tag_embedding_size", 8, "embedding size")
tf.flags.DEFINE_string("mode", "train", "1:training stage 0:predicting stage")
tf.flags.DEFINE_integer("local", 1, "1:local 0:online")
tf.flags.DEFINE_string("output_table", "", "output table name in MaxComputer")
tf.flags.DEFINE_string("checkpointDir", "", "checkpointDir")
tf.flags.DEFINE_string("saved_model_dir", "", "saved model dir")
tf.flags.DEFINE_string("buckets", "", "oss host")
tf.flags.DEFINE_string("recall_cnt_file", "", "recall_cnt_file")
tf.flags.DEFINE_string("item_tower_file", "", "item_tower_file")
tf.flags.DEFINE_string("top_k_num", "1", "number of top k")
tf.flags.DEFINE_string("neg_sample_num", "1", "number of top k")
tf.flags.DEFINE_string("summaryDir", "", "summary dir")

FLAGS = tf.flags.FLAGS


def main(_):
    # Data File
    train_file_dir = [FLAGS.tables]
    test_file_dir = FLAGS.test_file_dir
    # Hyper Parameters
    learning_rate = FLAGS.learning_rate
    batch_size = FLAGS.batch_size
    item_embedding_size = FLAGS.item_embedding_size
    cate_embedding_size = FLAGS.cate_embedding_size
    tag_embedding_size = FLAGS.tag_embedding_size
    mode = FLAGS.mode
    output_table = FLAGS.output_table
    saved_model_dir = FLAGS.saved_model_dir
    checkpoint_dir = FLAGS.checkpointDir
    oss_bucket_dir = FLAGS.buckets
    summary_dir = FLAGS.summaryDir
    local = FLAGS.local
    top_k_num = int(FLAGS.top_k_num)
    neg_sample_num = int(FLAGS.neg_sample_num)
    print("train_file_dir: %s" % train_file_dir)
    print("test_file_dir: %s" % test_file_dir)
    print("learning_rate: %f" % learning_rate)
    print("item_embedding_size: %d" % item_embedding_size)
    print("cate_embedding_size: %d" % cate_embedding_size)
    print("tag_embedding_size: %d" % tag_embedding_size)
    print("batch_size: %d" % batch_size)
    print("output table name: %s " % output_table)
    print("checkpoint_dir: %s " % checkpoint_dir)
    print("oss bucket dir: %s" % oss_bucket_dir)
    print("summary dir: %s" % summary_dir)

    if local:
        # summary_dir = "../summary/"
        # recall_cnt_file = "../data/youtube_recall_item_cnt*"
        oss_bucket_dir = ""
    else:
        # summary_dir = oss_bucket_dir + "experiment/summary/"
        # train_file_dir = "oss://ivwen-recsys.oss-cn-shanghai-internal.aliyuncs.com/ctr_data/" + train_file_dir
        # test_file_dir = "oss://ivwen-recsys.oss-cn-shanghai-internal.aliyuncs.com/ctr_data/" + test_file_dir
        # recall_cnt_file = oss_bucket_dir + recall_cnt_file
        # item_tower_file = oss_bucket_dir + item_tower_file
        pass
    saved_model_dir = checkpoint_dir + saved_model_dir

    print("saved_model_dir: %s " % saved_model_dir)
    # GPU config
    # gpu_config = tf.ConfigProto()
    # gpu_config.gpu_options.allow_growth = True
    #

    with tf.Session() as sess:
        # train_files = utils.get_file_name(train_file_dir)
        # print("train files list:")
        # print(train_files)
        # test_files = utils.get_file_name(test_file_dir)
        # print("test files list:")
        # print(test_files)
        two_tower_model = TTFRCTRModelV4(train_file_dir=train_file_dir,
                                         # test_file_dir=test_files,
                                         test_file_dir=train_file_dir,
                                         mode=mode,
                                         item_embedding_size=item_embedding_size,
                                         cate_embedding_size=cate_embedding_size,
                                         tag_embedding_size=tag_embedding_size,
                                         batch_size=batch_size,
                                         learning_rate=learning_rate,
                                         local=local,
                                         output_table=output_table,
                                         top_k_num=top_k_num,
                                         neg_sample_num=neg_sample_num
                                         )
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            loss_sum = 0.0
            sigmoid_loss_sum = 0.0
            pred_step = 1

            if two_tower_model.mode == "train":
                train, losses, sigmoid_losses, lr = two_tower_model.train_model()
                sess.run(tf.global_variables_initializer())
                # sess.run(tf.tables_initializer())
                # sess.run(two_tower_model.training_init_op)
            else:
                two_tower_model.restore_model(sess, checkpoint_dir)
                print("restore model finished!!")

                two_tower_model.evaluate()
                y_true = np.array([])
                y_predict = np.array([])
                # auc_running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="auc")
                # running_vars_initializer = tf.variables_initializer(var_list=auc_running_vars)
                # sess.run(running_vars_initializer)

            while not coord.should_stop():
                if two_tower_model.mode == "train":
                    train_step = two_tower_model.global_step.eval()
                    _, loss, sigmoid_loss, learning_rate = sess.run([train, losses, sigmoid_losses, lr])

                    loss_sum += loss
                    sigmoid_loss_sum += sigmoid_loss
                    if train_step % two_tower_model.PRINT_STEP == 0:
                        if train_step == 0:
                            print(
                                'time: %s\tEpoch: %d\tGlobal_Train_Step: %d\tTrain_loss: %.8f\tsigmoid_loss: %.8f\tLearning_rate:%.8f'
                                % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                   two_tower_model.train_epoches,
                                   train_step, loss_sum, sigmoid_loss_sum, learning_rate))

                        else:
                            print(
                                'time: %s\tEpoch: %d\tGlobal_Train_Step: %d\tTrain_loss: %.8f\tsigmoid_loss: %.8f\tLearning_rate:%.8f'
                                % (
                                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                    two_tower_model.train_epoches,
                                    train_step,
                                    loss_sum / two_tower_model.PRINT_STEP,
                                    sigmoid_loss_sum / two_tower_model.PRINT_STEP,
                                    # auc_sum / two_tower_model.PRINT_STEP,
                                    learning_rate))
                            if train_step % two_tower_model.SAVE_STEP == 0:
                                two_tower_model.save_model(sess=sess, path=checkpoint_dir)
                        loss_sum = 0.0
                        sigmoid_loss_sum = 0.0
                    # 本地测试输出
                    if local == 1:
                        pass
                        # # local print test
                        # user_id, click_label, logits, sigmoid_loss, user_embedding_final, item_embeding_final, uncclick_seq_50size_embed = sess.run(
                        #     [two_tower_model.user_id, two_tower_model.dev_brand,
                        #      two_tower_model.dev_brand_level, two_tower_model.dev_os, two_tower_model.dev_brand_type,
                        #      two_tower_model.dev_type, two_tower_model.dev_net])
                        # print("user_id")
                        # print(user_id)
                        # print("click_label")
                        # print(click_label)
                        # print("logits")
                        # print(logits)
                        # print("sigmoid_loss")
                        # print(sigmoid_loss)
                        # print("user_embedding_final")
                        # print(user_embedding_final)
                        # print("item_embeding_final")
                        # print(item_embeding_final)
                        # print("uncclick_seq_50size_embed")
                        # print(uncclick_seq_50size_embed)
                        # break
                elif two_tower_model.mode == "eval":
                    # sess.run(auc_op)
                    # auc = sess.run(auc_value)

                    # local print test
                    prediction, click_label = sess.run([two_tower_model.prediction, two_tower_model.click_label])
                    # print("prediction")
                    # print(prediction)
                    # print("click_label")
                    # print(click_label)
                    y_true = np.concatenate([y_true, click_label])
                    y_predict = np.concatenate([y_predict, prediction])
                    # y_true.append(list(click_label))
                    # y_predict.append(list(prediction))
                    # print(roc_auc_score(click_label, prediction))
                    # print("logits")
                    # print(logits)
                    # print("sigmoid_loss")
                    # print(sigmoid_loss)
                    # print("user_embedding_final")
                    # print(user_embedding_final)
                    # print("item_embeding_final")
                    # print(item_embeding_final)
                    # print("uncclick_seq_50size_embed")
                    # print(uncclick_seq_50size_embed)
                    #### 输出标记
                    if pred_step % two_tower_model.PRINT_STEP == 0:
                        print("%d finished" % pred_step)
                    #     # print(auc)
                    # pred_step += 1

            # y_true += click_label
            # y_predict += prediction

        except tf.errors.OutOfRangeError:
            print('time: %s\t %d records copied' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                                    two_tower_model.global_step.eval()))
            # if is_train == 1:
            #     two_tower_model.save_model(sess=sess, path=checkpoint_dir)

            # profile_code_builder = option_builder.ProfileOptionBuilder()
            # # profile_code_builder.with_node_names(show_name_regexes=['main.*'])
            # profile_code_builder.with_min_execution_time(min_micros=15)
            # profile_code_builder.select(['micros'])  # 可调整为 'bytes', 'occurrence'
            # profile_code_builder.order_by('micros')
            # profile_code_builder.with_max_depth(6)
            # my_profiler.profile_python(profile_code_builder.build())
            # my_profiler.profile_operations(profile_code_builder.build())
            # my_profiler.profile_name_scope(profile_code_builder.build())
            # my_profiler.profile_graph(profile_code_builder.build())
        finally:
            coord.request_stop()
            coord.join(threads)
        # 保存模型
        if two_tower_model.mode == "train":
            print("time: %s\tckpt model save start..." % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            two_tower_model.save_model(sess=sess, path=checkpoint_dir)
            print("time: %s\tsave_model save start..." % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            two_tower_model.save_model_as_savedmodel(sess=sess,
                                                     dir=saved_model_dir,
                                                     inputs=two_tower_model.saved_model_inputs,
                                                     outputs=two_tower_model.saved_model_outputs)
        # auc评估
        elif two_tower_model.mode == "eval":
            print(y_true.shape)
            print(y_predict.shape)
            auc = roc_auc_score(y_true, y_score=y_predict)
            print("auc:{}".format(auc))


if __name__ == "__main__":
    tf.app.run()
