# -*- coding: utf-8 -*-
from functools import reduce

import tensorflow as tf
import math
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def
from tensorflow.python.saved_model import tag_constants
import tensorflow.feature_column as fc
from tensorflow.python.feature_column.feature_column import _LazyBuilder


class TTFRCTRModelV3:
    def __init__(self,
                 train_file_dir,
                 test_file_dir,
                 mode,
                 item_embedding_size,
                 cate_embedding_size,
                 tag_embedding_size,
                 batch_size,
                 learning_rate,
                 output_table,
                 local,
                 top_k_num,
                 neg_sample_num
                 ):
        self.train_file_dir = train_file_dir
        self.test_file_dir = test_file_dir
        self.mode = mode
        # self.train_file_dir = train_file_dir
        # self.test_file_dir = test_file_dir
        # self.item_tower_file = item_tower_file
        self.item_embedding_size = item_embedding_size
        self.cate_embedding_size = cate_embedding_size
        self.tag_embedding_size = tag_embedding_size
        self.batch_size = batch_size
        self.learning_rate = tf.constant(learning_rate)
        self.table_path = "odps://dpdefault_68367/tables/"
        self.output_path = self.table_path + output_table
        self.local = local
        self.GENDER_CNT = 3
        self.CLIENT_TYPE_CNT = 3
        self.AGE_CNT = 6
        self.CONSUME_LEVEL_CNT = 4
        self.DEV_BRAND_CNT = 240
        self.DEV_BRAND_TYPE_CNT = 24
        self.DEV_TYPE_CNT = 4
        self.DEV_CARRIER_CNT = 5
        self.DEV_NET_CNT = 6
        # self.DEV_BRAND_LEVEL_CNT = 4
        self.DEV_OS_CNT = 700

        self.MEIPIAN_AGE_CNT = 15
        self.NUM_SAMPLED = neg_sample_num
        self.NUM_OF_TABLE_COLUMNS = 3
        self.TOP_K_ITEM_CNT = top_k_num
        self.NUM_OF_EPOCHS = 1
        self._CSV_COLUMNS = ["label", "user_id", "item_id", "class_id", "tag_ids", "client_type", "click_seq_50size",
                             "gender"]
        self._CSV_COLUMN_DEFAULTS = [['x'] for x in self._CSV_COLUMNS]
        self._FEATURES = ["item_id", "class_id", "tag_ids", "client_type", "click_seq_50size", "gender"]
        self._TYPES = [tf.string for _ in self._FEATURES]

        # 预留位置
        self.ITEM_CNT = 1000000
        self.CATE_CNT = 30
        self.TAG_CNT = 300

        # 误差（1e-8）防止除数为0
        self.EPS = tf.constant(1e-8, tf.float32)
        self.saved_model_outputs = {}
        # self.sess = sess

        self.train_epoches = 1
        self.pred_epoches = 1
        if local:
            self.PRINT_STEP = 1
            self.SAVE_STEP = 10
        else:
            self.PRINT_STEP = 1000
            self.SAVE_STEP = 30000
        if self.mode in ["train", "eval"]:
            # read data
            self.click_label, self.user_id, self.item_id, self.class_id, self.tag_ids, self.dev_type, self.dev_brand, self.dev_brand_type, \
            self.dev_os, self.dev_net, self.client_type, self.dev_carrier, self.click_seq_50size, self.gender, self.age, self.consume_level, self.unclick_seq_50size, \
                = self.build_train_batch_data(self.train_file_dir,
                                              self.batch_size,
                                              self.train_epoches)

            tensor_info_user_id = tf.saved_model.utils.build_tensor_info(self.user_id)
            tensor_info_item_id = tf.saved_model.utils.build_tensor_info(self.item_id)
            tensor_info_class_id = tf.saved_model.utils.build_tensor_info(self.class_id)
            tensor_info_tag_ids = tf.saved_model.utils.build_tensor_info(self.tag_ids)
            tensor_info_dev_type = tf.saved_model.utils.build_tensor_info(self.dev_type)
            tensor_info_dev_brand = tf.saved_model.utils.build_tensor_info(self.dev_brand)
            tensor_info_dev_brand_type = tf.saved_model.utils.build_tensor_info(self.dev_brand_type)
            # tensor_info_dev_brand_level = tf.saved_model.utils.build_tensor_info(self.dev_brand_level)
            tensor_info_dev_os = tf.saved_model.utils.build_tensor_info(self.dev_os)
            tensor_info_dev_net = tf.saved_model.utils.build_tensor_info(self.dev_net)
            tensor_info_client_type = tf.saved_model.utils.build_tensor_info(self.client_type)
            tensor_info_dev_carrier = tf.saved_model.utils.build_tensor_info(self.dev_carrier)
            tensor_info_click_seq_50size = tf.saved_model.utils.build_tensor_info(self.click_seq_50size)
            tensor_info_gender = tf.saved_model.utils.build_tensor_info(self.gender)
            tensor_info_age = tf.saved_model.utils.build_tensor_info(self.age)
            tensor_info_consume_level = tf.saved_model.utils.build_tensor_info(self.consume_level)
            tensor_info_unclick_seq_50size = tf.saved_model.utils.build_tensor_info(self.unclick_seq_50size)

            # build saved inputs
            self.saved_model_inputs = {
                "user_id": tensor_info_user_id,
                "item_id": tensor_info_item_id,
                "class_id": tensor_info_class_id,
                "tag_ids": tensor_info_tag_ids,
                "dev_type": tensor_info_dev_type,
                "dev_brand": tensor_info_dev_brand,
                "dev_brand_type": tensor_info_dev_brand_type,
                # "dev_brand_level": tensor_info_dev_brand_level,
                "dev_os": tensor_info_dev_os,
                "dev_net": tensor_info_dev_net,
                "client_type": tensor_info_client_type,
                "dev_carrier": tensor_info_dev_carrier,
                "click_seq_50size": tensor_info_click_seq_50size,
                "gender": tensor_info_gender,
                "age": tensor_info_age,
                "consume_level": tensor_info_consume_level,
                "unclick_seq_50size": tensor_info_unclick_seq_50size,
            }

            # builder = _LazyBuilder(self.features)
            # gender_voclist = ['0', '1', '-1']
            # gender_column = fc.categorical_column_with_vocabulary_list('gender', gender_voclist, num_oov_buckets=0,
            #                                                            dtype=tf.string)
            # gender_column_identy = fc.indicator_column(gender_column)
            # self.gender_tensor = fc.input_layer(self.features, [gender_column_identy])
            #
            # client_type_voclist = ['0', '1', '-1']
            # client_type_column = fc.categorical_column_with_vocabulary_list('client_type', client_type_voclist,
            #                                                                 num_oov_buckets=0,
            #                                                                 dtype=tf.string)
            # client_type_column_identy = fc.indicator_column(client_type_column)
            # self.client_type_tensor = fc.input_layer(self.features, [client_type_column_identy])
            #
            # self.hash_item_id = fc.categorical_column_with_hash_bucket("item_id", hash_bucket_size=self.ITEM_CNT,
            #                                                            dtype=tf.string)
            # self.hash_tag_ids = fc.categorical_column_with_hash_bucket("tag_ids", hash_bucket_size=self.TAG_CNT,
            #                                                            dtype=tf.string)
            # self.hash_class_id = fc.categorical_column_with_hash_bucket("class_id", hash_bucket_size=self.CATE_CNT,
            #                                                             dtype=tf.string)
            # print(self.features.items())
        # elif self.mode == "test":
        #     self.dataset = self.build_user_train_data(self.test_file_dir)
        #     self.train_iterator = self.dataset.make_one_shot_iterator()
        #
        #     self.label, self.user_id, self.item_id, self.class_id, self.tag_ids, self.client_type, self.click_seq_50size, self.gender = self.train_iterator.get_next()
        #     # self.click_seq_50size_len = tf.count_nonzero(self.click_seq_50size, 1)
        #     # self.unclick_seq_50size_len = tf.count_nonzero(self.unclick_seq_50size, 1)
        #
        #     self.training_init_op = self.train_iterator.make_initializer(self.dataset)
        #     # batch_size个数的0(正样本的index)，需要注意最后一个batch是不足s一个batch_size的
        #
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.item_initializer = tf.truncated_normal([self.ITEM_CNT, self.item_embedding_size],
                                                    stddev=1.0 / math.sqrt(self.item_embedding_size))
        self.cate_initializer = tf.truncated_normal([self.CATE_CNT, self.cate_embedding_size],
                                                    stddev=1.0 / math.sqrt(self.cate_embedding_size))
        self.tag_initializer = tf.truncated_normal([self.TAG_CNT, self.tag_embedding_size],
                                                   stddev=1.0 / math.sqrt(self.tag_embedding_size))
        #
        with tf.name_scope('item_embedding'):
            # Variable
            self.item_embedding = tf.Variable(self.item_initializer, name='item_embedding')
            self.cate_embedding = tf.Variable(self.cate_initializer, name='cate_embedding')
            self.tag_embedding = tf.Variable(self.tag_initializer, name='tag_embedding')
        #
        #     # saved_model 格式的输入，用于eas在线预测
        # if self.mode == "train":
        #     # tensor_info_user_id = tf.saved_model.utils.build_tensor_info(self.user_id)
        #     tensor_info_gender = tf.saved_model.utils.build_tensor_info(self.gender)
        #     tensor_info_client_type = tf.saved_model.utils.build_tensor_info(self.client_type)
        #     tensor_info_user_click_item_list = tf.saved_model.utils.build_tensor_info(self.click_seq_50size_input)
        #     tensor_info_item_id = tf.saved_model.utils.build_tensor_info(self.item_id)
        #     tensor_info_class_id = tf.saved_model.utils.build_tensor_info(self.class_id)
        #     tensor_info_tag_ids = tf.saved_model.utils.build_tensor_info(self.tag_ids)
        #

        #
        # self.click_seq_50size_ex = tf.string_to_number(tf.map_fn(fn=lambda x: tf.string_split([x], ',').values,
        #                                                          elems=self.click_seq_50size_input), tf.int32)
        # self.click_seq_50size_len = tf.count_nonzero(self.click_seq_50size_ex, 1)
        # self.build_categorial_features()
        # self.build_sequence_features()
        self.build_categorial_features()
        self.build_sequence_features()
        self.build_label()
        self.build_model()

    def read_train_data(self, file_queue):
        if self.local:
            reader = tf.TextLineReader(skip_header_lines=1)
        else:
            reader = tf.TableRecordReader()
        key, value = reader.read(file_queue)
        defaults = [['0']] * 17
        click_label, user_id, item_id, class_id, tag_ids, dev_type, dev_brand, dev_brand_type, \
        dev_os, dev_net, client_type, dev_carrier, click_seq_50size, gender, age, consume_level, unclick_seq_50size \
            = tf.decode_csv(value, defaults)

        # user_id = tf.cast(user_id, dtype=tf.int64)
        # gender = tf.cast(gender, dtype=tf.int64)
        # client_type = tf.cast(client_type, dtype=tf.int64)
        # tag_ids = tf.string_to_number(tf.string_split([tag_ids], ',').values, tf.int64)
        # click_seq_50size = tf.string_to_number(tf.string_split([click_seq_50size], ',').values, tf.int64)
        # unclick_seq_50size = tf.string_to_number(tf.string_split([unclick_seq_50size], ',').values, tf.int64)
        return click_label, user_id, item_id, class_id, tag_ids, dev_type, dev_brand, dev_brand_type, dev_os, dev_net, client_type, dev_carrier, click_seq_50size, gender, age, consume_level, unclick_seq_50size

    def build_train_batch_data(self, train_data_file, batch_size, num_epochs):
        file_queue = tf.train.string_input_producer(train_data_file, num_epochs=num_epochs)
        click_label, user_id, item_id, class_id, tag_ids, dev_type, dev_brand, dev_brand_type, dev_os, dev_net, client_type, dev_carrier, click_seq_50size, gender, age, consume_level, unclick_seq_50size = self.read_train_data(
            file_queue)
        # min_after_dequeue = 1000
        capacity = 10000
        click_label_batch, user_id_batch, item_id_batch, class_id_batch, tag_ids_batch, dev_type_batch, dev_brand_batch, \
        dev_brand_type_batch, dev_os_batch, dev_net_batch, client_type_batch, dev_carrier_batch, \
        click_seq_50size_batch, gender_batch, age_batch, consume_level_batch, unclick_seq_50size_batch = tf.train.batch(
            [click_label, user_id, item_id, class_id, tag_ids, dev_type, dev_brand, dev_brand_type,
             dev_os, dev_net, client_type, dev_carrier, click_seq_50size, gender, age, consume_level,
             unclick_seq_50size],
            batch_size=batch_size, capacity=capacity,
            num_threads=1, allow_smaller_final_batch=True,
            dynamic_pad=True
            # min_after_dequeue=min_after_dequeue
        )
        return click_label_batch, user_id_batch, item_id_batch, class_id_batch, tag_ids_batch, dev_type_batch, dev_brand_batch, \
               dev_brand_type_batch, dev_os_batch, dev_net_batch, client_type_batch, dev_carrier_batch, \
               click_seq_50size_batch, gender_batch, age_batch, consume_level_batch, unclick_seq_50size_batch

    # 计算序列的embedding(平均或求和)
    # TODO: 后续可以使用序列建模方式抽取序列特征
    def get_seq_embedding(self, item_embedding, user_click_item_list, item_id_list_len_batch, embedding_size, method):

        self.embed_init = tf.nn.embedding_lookup(item_embedding, user_click_item_list)
        self.embedding_mask = tf.sequence_mask(item_id_list_len_batch, tf.shape(user_click_item_list)[1],
                                               dtype=tf.float32)
        self.embedding_mask_expand = tf.expand_dims(self.embedding_mask, -1)
        self.embedding_mask_tile = tf.tile(self.embedding_mask_expand, [1, 1, embedding_size])
        self.embedding_mask_2 = self.embed_init * self.embedding_mask_tile
        self.embedding_sum = tf.reduce_sum(self.embedding_mask_2, 1)

        seq_embedding_final = None
        if method == 'sum':
            seq_embedding_final = self.embedding_sum
        elif method == 'mean':
            # lenth 为0时，分母要加一个很小的数，防止结果为nan
            seq_avg_embedding = tf.div(self.embedding_sum,
                                       tf.cast(tf.tile(tf.expand_dims(item_id_list_len_batch, 1),
                                                       [1, embedding_size]),
                                               tf.float32) + 1e-8)
            seq_embedding_final = seq_avg_embedding

        return seq_embedding_final

    def build_sequence_features(self):
        # build feature
        self.tag_ids_array = tf.reshape(tf.strings.split(self.tag_ids, sep=";").values,
                                        shape=[tf.shape(self.user_id)[0], -1])
        self.unclick_seq_50size_array = tf.reshape(tf.strings.split(self.unclick_seq_50size, sep=";").values,
                                                   shape=[tf.shape(self.user_id)[0], -1])
        self.click_seq_50size_array = tf.reshape(tf.strings.split(self.click_seq_50size, sep=";").values,
                                                 shape=[tf.shape(self.user_id)[0], -1])
        self.click_seq_50size_len = tf.count_nonzero(self.click_seq_50size_array, 1)
        self.tag_ids_len = tf.count_nonzero(self.tag_ids_array, 1)
        self.unclick_seq_50size_len = tf.count_nonzero(self.unclick_seq_50size_array, 1)

        self.click_seq_50size_hash = tf.string_to_hash_bucket_fast(self.click_seq_50size_array, self.ITEM_CNT)
        self.unclick_seq_50size_hash = tf.string_to_hash_bucket_fast(self.unclick_seq_50size_array, self.ITEM_CNT)
        self.tag_ids_hash = tf.string_to_hash_bucket_fast(self.tag_ids_array, self.TAG_CNT)

        # list features embed
        self.click_seq_50size_embed = self.get_seq_embedding(self.item_embedding,
                                                             self.click_seq_50size_hash,
                                                             self.click_seq_50size_len,
                                                             self.item_embedding_size,
                                                             "mean")

        self.uncclick_seq_50size_embed = self.get_seq_embedding(self.item_embedding,
                                                                self.unclick_seq_50size_hash,
                                                                self.unclick_seq_50size_len,
                                                                self.item_embedding_size,
                                                                "mean")

        self.tag_ids_embed = self.get_seq_embedding(self.tag_embedding,
                                                    self.tag_ids_hash,
                                                    self.tag_ids_len,
                                                    self.tag_embedding_size,
                                                    "mean")

    # categorial features embed
    def build_categorial_features(self):
        self.class_id = tf.string_to_number(self.class_id, tf.int32)
        self.class_id_one_hot = tf.one_hot(self.class_id, self.GENDER_CNT)

        self.gender = tf.string_to_number(self.gender, tf.int32)
        self.gender_one_hot = tf.one_hot(self.gender, self.GENDER_CNT)

        self.age = tf.string_to_number(self.age, tf.int32)
        self.age_one_hot = tf.one_hot(self.age, self.AGE_CNT)

        self.consume_level = tf.string_to_number(self.consume_level, tf.int32)
        self.consume_level_one_hot = tf.one_hot(self.consume_level, self.CONSUME_LEVEL_CNT)

        self.client_type = tf.string_to_number(self.client_type, tf.int32)
        self.client_type_one_hot = tf.one_hot(self.client_type, self.CLIENT_TYPE_CNT)

        self.item_id_hash = tf.string_to_hash_bucket_fast(self.item_id, self.ITEM_CNT)
        self.item_id_embed = tf.nn.embedding_lookup(self.item_embedding, self.item_id_hash)

        self.dev_brand_type = tf.string_to_number(self.dev_brand_type, tf.int32)
        self.dev_brand_type_one_hot = tf.one_hot(self.dev_brand_type, self.DEV_BRAND_TYPE_CNT)

        self.dev_type = tf.string_to_number(self.dev_type, tf.int32)
        self.dev_type_one_hot = tf.one_hot(self.dev_type, self.DEV_TYPE_CNT)

        self.dev_carrier = tf.string_to_number(self.dev_carrier, tf.int32)
        self.dev_carrier_one_hot = tf.one_hot(self.dev_carrier, self.DEV_CARRIER_CNT)

        self.dev_net = tf.string_to_number(self.dev_net, tf.int32)
        self.dev_net_one_hot = tf.one_hot(self.dev_net, self.DEV_NET_CNT)

        self.dev_brand_idx = tf.string_to_hash_bucket_fast(self.dev_brand, self.DEV_BRAND_CNT)
        self.dev_brand_one_hot = tf.one_hot(self.dev_brand_idx, self.DEV_BRAND_CNT)

        self.dev_os = tf.string_to_hash_bucket_fast(self.dev_os, self.DEV_OS_CNT)
        self.dev_os_one_hot = tf.one_hot(self.dev_os, self.DEV_OS_CNT)

    def build_label(self):
        self.click_label = tf.string_to_number(self.click_label, tf.float32)

    def build_model(self):

        # User Embedding Layer
        with tf.name_scope("user_tower"):
            # concat embedding
            self.user_embed_concat = tf.concat(
                [self.click_seq_50size_embed, self.uncclick_seq_50size_embed, self.tag_ids_embed,
                 self.gender_one_hot, self.age_one_hot, self.consume_level_one_hot,
                 self.client_type_one_hot, self.dev_brand_type_one_hot, self.dev_type_one_hot, self.dev_carrier_one_hot,
                 self.dev_net_one_hot, self.dev_brand_one_hot, self.dev_os_one_hot],
                axis=-1)
            user_layer_1 = tf.layers.dense(inputs=self.user_embed_concat,
                                           units=1024,
                                           activation=tf.nn.tanh,
                                           name='user_first',
                                           kernel_initializer=tf.initializers.glorot_normal(seed=1234),
                                           use_bias=False
                                           )
            user_layer_2 = tf.layers.dense(inputs=user_layer_1,
                                           units=512,
                                           activation=tf.nn.tanh,
                                           name='user_second',
                                           kernel_initializer=tf.initializers.glorot_normal(seed=1234),
                                           use_bias=False
                                           )
            user_layer_3 = tf.layers.dense(inputs=user_layer_2,
                                           units=128,
                                           activation=tf.nn.tanh,
                                           name='user_final',
                                           kernel_initializer=tf.initializers.glorot_normal(seed=1234),
                                           use_bias=False
                                           )

            #     # user参数写入summary
            #     # user_first_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'user_first')
            #     # user_second_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'user_second')
            #     # user_final_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'user_final')
            #
            #     # tf.summary.histogram("user_layer_1_weights", user_first_vars[0])
            #     # # tf.summary.histogram("user_layer_1_biases", user_first_vars[1])
            #     # tf.summary.histogram("user_layer_1_output", user_layer_1)
            #     #
            #     # tf.summary.histogram("user_layer_2_weights", user_second_vars[0])
            #     # # tf.summary.histogram("user_layer_2_biases", user_second_vars[1])
            #     # tf.summary.histogram("user_layer_2_output", user_layer_2)
            #     # #
            #     # tf.summary.histogram("user_layer_3_weights", user_final_vars[0])
            #     # # tf.summary.histogram("user_layer_3_biases", user_final_vars[1])
            #     # tf.summary.histogram("user_layer_3_output", user_layer_3)
            #
            self.user_embedding_final = user_layer_3

        with tf.name_scope("item_tower"):
            item_embed_concat = tf.concat([self.item_id_embed, self.tag_ids_embed], axis=-1)

            item_layer_1 = tf.layers.dense(inputs=item_embed_concat,
                                           units=128,
                                           activation=tf.nn.tanh,
                                           name='item_first',
                                           kernel_initializer=tf.initializers.glorot_normal(seed=1234),
                                           use_bias=False
                                           )


            #
            # # item参数写入summary
            # # item_first_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'item_first')
            # # tf.summary.histogram("item_layer_1_weights", item_first_vars[0])
            # # # tf.summary.histogram("item_layer_1_biases", item_first_vars[1])
            # # tf.summary.histogram("item_layer_1_output", item_layer_1)
            #
            self.item_embeding_final = item_layer_1

        # 计算logits
        self.logits = tf.reduce_sum(tf.multiply(self.user_embedding_final, self.item_embeding_final), axis=1)
        #
        # # # saved_model 输出
        self.prediction_score = tf.nn.sigmoid(self.logits)
        tensor_info_prediction_score = tf.saved_model.utils.build_tensor_info(self.prediction_score)
        self.saved_model_outputs["prediction_score"] = tensor_info_prediction_score

    def train_model(self):

        self.sigmoid_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.click_label, logits=self.logits))

        # l2_norm loss
        regulation_rate = 0.0001
        l2_norm = tf.add_n([
            tf.reduce_sum(tf.multiply(self.user_embedding_final, self.user_embedding_final)),
            tf.reduce_sum(tf.multiply(self.item_embeding_final, self.item_embeding_final)),
        ])
        l2_loss = regulation_rate * l2_norm
        merge_loss = l2_loss + self.sigmoid_loss

        # tf.summary.scalar('loss_merge', loss_merge)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        # train_op = optimizer.minimize(loss_merge, global_step=self.global_step)

        var_list = tf.trainable_variables()
        gradients = optimizer.compute_gradients(merge_loss, var_list)
        capped_gradients = [(tf.clip_by_value(grad, -500., 500.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients, global_step=self.global_step)

        return train_op, merge_loss, self.sigmoid_loss, self.learning_rate

    def evaluate(self):
        self.prediction = tf.nn.sigmoid(self.logits)

        ## tensorflow 提供的auc计算经过测试发现不靠谱
        # self.click_label = (self.click_label == 1)
        # auc_value, auc_op = tf.metrics.accuracy(labels=self.click_label, predictions=self.prediction, name="auc")
        # return auc_value, auc_op

    # def write_table(self):
    #     writer = tf.TableRecordWriter(self.output_path)
    #     write_to_table = writer.write(range(self.NUM_OF_TABLE_COLUMNS),
    #                                   [tf.as_string(self.user_id),
    #                                    self.user_topk_item,
    #                                    self.user_topk_score])
    #     return write_to_table

    def save_model(self, sess, path):
        saver = tf.train.Saver()
        if not tf.gfile.Exists(path):
            tf.gfile.MkDir(path)
        saver.save(sess, save_path=path + 'model.ckpt')

    def restore_model(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path + 'model.ckpt')

    def save_model_as_savedmodel(self, sess, dir, inputs, outputs):

        if not tf.gfile.Exists(dir):
            print("dir not exists")
            tf.gfile.MkDir(dir)
        tf.gfile.DeleteRecursively(dir)
        builder = tf.saved_model.builder.SavedModelBuilder(dir)

        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs=inputs,
                outputs=outputs,
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    prediction_signature
            }
            # ,legacy_init_op=tf.tables_initializer()
        )

        builder.save()
