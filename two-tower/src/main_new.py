# -*- coding: utf-8 -*-
import tensorflow as tf
from youtube_dnn_new_user_model import YoutubeDnnNewUser
import utils

tf.flags.DEFINE_string("tables", "", "tables info")
tf.flags.DEFINE_float("learning_rate", 0.01, "learning_rate")
tf.flags.DEFINE_integer("batch_size", 32, "batch size")
tf.flags.DEFINE_integer("embedding_size", 64, "embedding size")
tf.flags.DEFINE_integer("is_train", 1, "1:training stage 0:predicting stage")
tf.flags.DEFINE_integer("local", 1, "1:local 0:online")
tf.flags.DEFINE_string("output_table", "", "output table name in MaxComputer")
tf.flags.DEFINE_string("checkpointDir", "", "checkpointDir")
tf.flags.DEFINE_string("buckets", "", "oss host")
tf.flags.DEFINE_string("recall_cnt_file", "", "recall_cnt_file")
tf.flags.DEFINE_string("top_k_num", "", "number of top k")
tf.flags.DEFINE_string("neg_sample_num", "", "number of top k")

FLAGS = tf.flags.FLAGS

tf.logging.set_verbosity(tf.logging.DEBUG)


def main(_):
    tables = [FLAGS.tables]
    learning_rate = FLAGS.learning_rate
    batch_size = FLAGS.batch_size
    embedding_size = FLAGS.embedding_size
    is_train = FLAGS.is_train
    output_table = FLAGS.output_table
    checkpoint_dir = FLAGS.checkpointDir
    oss_bucket_dir = FLAGS.buckets
    recall_cnt_file = FLAGS.recall_cnt_file
    local = FLAGS.local
    top_k_num = int(FLAGS.top_k_num)
    neg_sample_num = int(FLAGS.neg_sample_num)

    if local:
        savedmodel_dir = '../savedmodel_path/'
    else:
        savedmodel_dir = oss_bucket_dir + 'savedmodel_path/'
        # summary_dir = oss_bucket_dir + "experiment/summary/"
        recall_cnt_file = oss_bucket_dir + recall_cnt_file
    # get item cnt

    tf.logging.debug("tables: %s" % tables)
    print("is_train: %d" % is_train)
    print("learning_rate: %f" % learning_rate)
    print("embedding_size: %d" % embedding_size)
    print("batch_size: %d" % batch_size)
    print("output table name: %s " % output_table)
    print("checkpointDir: %s" % checkpoint_dir)
    print("oss bucket dir: %s" % oss_bucket_dir)
    print("top_k_num: %d" % top_k_num)

    item_count, brand_count, province_count, city_count = utils.get_item_cnt(recall_cnt_file)
    print("item_count: ", item_count)
    print("brand_count: ", brand_count)
    print("province_count: ", province_count)
    print("city_count: ", city_count)

    # GPU config
    # gpu_config = tf.ConfigProto()
    # gpu_config.gpu_options.allow_growth = True
    #

    with tf.Session() as sess:
        youtube_dnn_model = YoutubeDnnNewUser(tables=tables,
                                              is_train=is_train,
                                              embedding_size=embedding_size,
                                              batch_size=batch_size,
                                              learning_rate=learning_rate,
                                              local=local,
                                              item_count=item_count,
                                              brand_count=brand_count,
                                              province_count=province_count,
                                              city_count=city_count,
                                              output_table=output_table,
                                              top_k_num=top_k_num,
                                              neg_sample_num=neg_sample_num
                                              )

        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            loss_sum = 0.0
            pred_step = 0
            if is_train == 1:
                train, losses = youtube_dnn_model.train_model()
                sess.run(tf.global_variables_initializer())
            else:
                youtube_dnn_model.restore_model(sess, checkpoint_dir)
                # sess.run(tf.global_variables_initializer())
                # youtube_dnn_model.load_saved_model(sess, savedmodel_dir)
                print("restore model finished!!")

                topk_item, topk_score = youtube_dnn_model.predict_topk_score()
                if local == 0:
                    writer = youtube_dnn_model.write_table(topk_item, topk_score)

            while not coord.should_stop():
                if is_train:
                    train_step = youtube_dnn_model.global_step.eval()
                    _, loss, learning_rate = sess.run([train, losses, youtube_dnn_model.learning_rate])
                    loss_sum += loss
                    if train_step % youtube_dnn_model.PRINT_STEP == 0:
                        if train_step == 0:
                            print('Epoch: %d\tGlobal_Train_Step: %d\tTrain_loss: %.8f\tLearning_rate:%.8f'
                                  % (youtube_dnn_model.epoches, train_step, loss_sum, learning_rate))
                        else:
                            print('Epoch: %d\tGlobal_Train_Step: %d\tTrain_loss: %.8f\tLearning_rate:%.8f'
                                  % (youtube_dnn_model.epoches, train_step, loss_sum / youtube_dnn_model.PRINT_STEP,
                                     learning_rate))

                        loss_sum = 0.0
                else:
                    if local:
                        user_id_batch, topk_item_1, topk_score_1 = sess.run(
                            [youtube_dnn_model.user_batch, topk_item, topk_score])
                        print(user_id_batch)
                        print(topk_item_1)
                        print(topk_score_1)
                    else:
                        sess.run(writer)

                    if pred_step % 1000 == 0:
                        print("%d finished" % pred_step)
                    pred_step += 1

        except tf.errors.OutOfRangeError:
            print('%d records copied' % youtube_dnn_model.global_step.eval())

        finally:
            coord.request_stop()
            coord.join(threads)

        if is_train:
            youtube_dnn_model.save_model(sess=sess, path=checkpoint_dir)
            youtube_dnn_model.save_model_as_savedmodel(sess=sess,
                                                       dir=savedmodel_dir,
                                                       inputs=youtube_dnn_model.saved_model_inputs,
                                                       outputs=youtube_dnn_model.saved_model_outputs)


if __name__ == "__main__":
    tf.app.run()
