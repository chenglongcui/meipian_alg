# -*- coding: utf-8 -*-
import tensorflow as tf
from two_tower_model_fr_v2 import TwoTowerModelFRV2
# 可以单独用它生成 timeline，也可以使用下面两个对象生成 timeline
from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import option_builder
from tensorflow.python.client import timeline
import utils
import time

print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

tf.flags.DEFINE_string("train_file_dir", "", "train_file")
tf.flags.DEFINE_string("test_file_dir", "", "test_file_dir")
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
    test_file_dir = FLAGS.test_file_dir
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
    print("test_file_dir: %s" % test_file_dir)
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
        test_file_dir = oss_bucket_dir + test_file_dir
        recall_cnt_file = oss_bucket_dir + recall_cnt_file
        item_tower_file = oss_bucket_dir + item_tower_file
        saved_model_dir = oss_bucket_dir + saved_model_dir

    # get item cnt
    # item_count, cate_count, tag_count = utils.get_item_cnt(recall_cnt_file)
    # item_tower_file = [utils.get_file_name(item_tower_file)]
    # print("item tower file: %s" % item_tower_file)
    # print("item_count: ", item_count)
    # print("cate_count: ", cate_count)
    # print("tag_count: ", tag_count)
    print("saved_model_dir: %s " % saved_model_dir)
    # GPU config
    # gpu_config = tf.ConfigProto()
    # gpu_config.gpu_options.allow_growth = True
    #

    with tf.Session() as sess:
        train_file_name = utils.get_file_name(train_file_dir)
        test_file_dir = utils.get_file_name(test_file_dir)

        two_tower_model = TwoTowerModelFRV2(train_file_dir=train_file_name,
                                            test_file_dir=test_file_dir,
                                            item_tower_file=item_tower_file,
                                            is_train=is_train,
                                            item_embedding_size=item_embedding_size,
                                            cate_embedding_size=cate_embedding_size,
                                            tag_embedding_size=tag_embedding_size,
                                            batch_size=batch_size,
                                            learning_rate=learning_rate,
                                            local=local,
                                            output_table=output_table,
                                            top_k_num=top_k_num,
                                            neg_sample_num=neg_sample_num,
                                            sess=sess
                                            )
        try:
            loss_sum = 0.0
            pred_step = 0

            if is_train == 1:
                train, losses = two_tower_model.train_model()
                sess.run(tf.global_variables_initializer())
                # sess.run(tf.tables_initializer())
                sess.run(tf.local_variables_initializer())
                sess.run(two_tower_model.training_init_op)
            else:
                two_tower_model.restore_model(sess, checkpoint_dir)
                # sess.run(tf.global_variables_initializer())
                # youtube_dnn_model.load_saved_model(sess, savedmodel_dir)
                print("restore model finished!!")

                two_tower_model.predict_topk_score()
                if local == 0:
                    writer = two_tower_model.write_table()

                # # 读取历史参数，保证增量更新
                # if tf.gfile.Exists(checkpoint_dir):
                #     two_tower_model.restore_model(sess, checkpoint_dir)

            while True:
                # # 更新数据到 timeline 的第一种种方式
                # trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                # trace_file = open('timeline.ctf.json', 'w')
                # trace_file.write(trace.generate_chrome_trace_format())

                # 更新数据到profiler
                # my_profiler.add_step(step=int(train_step), run_meta=run_metadata)
                if is_train:
                    train_step = two_tower_model.global_step.eval()

                    # #
                    # _, loss, learning_rate = sess.run([train, losses, two_tower_model.learning_rate],
                    #                                   run_metadata=run_metadata, options=run_options)

                    _, loss, learning_rate = sess.run([train, losses, two_tower_model.learning_rate])
                    loss_sum += loss
                    if train_step % two_tower_model.PRINT_STEP == 0:
                        if train_step == 0:
                            print('time: %s\tEpoch: %d\tGlobal_Train_Step: %d\tTrain_loss: %.8f\tLearning_rate:%.8f'
                                  % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                     two_tower_model.train_epoches,
                                     train_step, loss_sum, learning_rate))

                        else:
                            print('time: %s\tEpoch: %d\tGlobal_Train_Step: %d\tTrain_loss: %.8f\tLearning_rate:%.8f'
                                  % (
                                      time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                      two_tower_model.train_epoches,
                                      train_step, loss_sum / two_tower_model.PRINT_STEP,
                                      learning_rate))
                            if train_step % two_tower_model.SAVE_STEP == 0:
                                two_tower_model.save_model(sess=sess, path=checkpoint_dir)
                        loss_sum = 0.0

                    # local test
                    # user_ids, user_click_hash, target_ids, target_hash = sess.run(
                    #     [
                    #         two_tower_model.user_item_click_avg_embed,
                    #         two_tower_model.gender_one_hot,
                    #         two_tower_model.target_item_list,
                    #         two_tower_model.target_item_idx,
                    #     ])
                    # print(user_ids)
                    # print(user_click_hash)
                    # print(target_ids)
                    # print(target_hash)
                    # print(label.shape)
                    # break
                else:
                    if local == 0:
                        sess.run(writer)
                    else:
                        user_embeddings, item_embeddings, logits = sess.run(
                            [two_tower_model.user_embedding_final,
                             two_tower_model.item_embeding_final,
                             two_tower_model.logits
                             ])
                        print("users:")
                        print(user_embeddings)
                        print("items:")
                        print(item_embeddings)
                        print("logits:")
                        print(logits)
                    if pred_step % 1000 == 0:
                        print("%d finished" % pred_step)
                    pred_step += 1

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

        if is_train == 1:
            print("time: %s\tckpt model save start..." % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            two_tower_model.save_model(sess=sess, path=checkpoint_dir)
            print("time: %s\tsave_model save start..." % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            two_tower_model.save_model_as_savedmodel(sess=sess,
                                                     dir=saved_model_dir,
                                                     inputs=two_tower_model.saved_model_inputs,
                                                     outputs=two_tower_model.saved_model_outputs)


if __name__ == "__main__":
    tf.app.run()
