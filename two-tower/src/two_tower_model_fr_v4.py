# -*- coding: utf-8 -*-
from functools import reduce

import tensorflow as tf
import math
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def
from tensorflow.python.saved_model import tag_constants


class TwoTowerModelFRV4:
    def __init__(self,
                 train_file_dir,
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
        self.mode = mode
        # self.train_file_dir = train_file_dir
        # self.test_file_dir = test_file_dir
        # self.item_tower_file = item_tower_file
        self.item_embedding_size = item_embedding_size
        self.cate_embedding_size = cate_embedding_size
        self.tag_embedding_size = tag_embedding_size
        # self.item_merge_embedding_size = item_embedding_size + cate_embedding_size + tag_embedding_size
        self.batch_size = batch_size
        self.learning_rate = tf.constant(learning_rate)
        self.table_path = "odps://dpdefault_68367/tables/"
        self.output_path = self.table_path + output_table
        self.local = local
        self.GENDER_CNT = 3
        self.CLIENT_TYPE_CNT = 3
        self.AGE_CNT = 5
        self.CONSUME_LEVEL_CNT = 3
        self.MEIPIAN_AGE_CNT = 15
        self.NUM_SAMPLED = neg_sample_num
        self.NUM_OF_TABLE_COLUMNS = 3
        self.TOP_K_ITEM_CNT = top_k_num
        self.NUM_OF_EPOCHS = 1

        # 预留位置
        # self.NUM_ITEM_OOV_BUCKET = 200000
        # self.NUM_CATE_OOV_BUCKET = 5
        # self.NUM_TAG_OOV_BUCKET = 10
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

        # ## train dataset
        # self.train_dataset = self.build_user_train_data(self.train_file_dir)
        # self.train_iterator = self.train_dataset.make_one_shot_iterator()
        #
        # self.user_id, self.user_click_item_list, self.gender, self.target_item_list, self.target_cate_list, self.target_tag_list = self.train_iterator.get_next()
        # self.user_click_item_list_len = tf.count_nonzero(self.user_click_item_list, 1)
        # self.training_init_op = self.train_iterator.make_initializer(self.train_dataset)
        # self.train_label = tf.zeros(shape=[tf.shape(self.user_id)[0]], dtype=tf.int64)
        #
        # ## test datasets
        # self.test_dataset = self.build_user_test_data(self.test_file_dir)
        # self.test_iterator = self.test_dataset.make_one_shot_iterator()
        #
        # self.test_user_id, self.test_user_click_item_list, self.test_gender, self.true_click = self.test_iterator.get_next()
        # self.testing_init_op = self.test_iterator.make_initializer(self.test_dataset)
        #
        # self.item_vobabulary, self.item_cate_mapping, self.item_tag_mapping, self.cate_vocabulary, \
        # self.tag_vocabulary = self.read_item_train_data()

        if self.mode == "train":
            # odps
            # self.user_id, self.user_click_item_list, self.gender, self.client_type, self.target_item_list, self.target_cate_list, \
            # self.target_tag_list = self.build_train_batch_data(train_data_file=tables,
            #                                                    batch_size=self.batch_size,
            #                                                    num_epochs=self.train_epoches)
            # self.user_click_item_list_len = tf.count_nonzero(self.user_click_item_list, 1)

            # dataset
            self.dataset = self.build_user_train_data(self.train_file_dir)
            self.iterator = self.dataset.make_one_shot_iterator()

            self.user_id, self.user_click_item_list, self.gender, self.client_type, self.target_item_list, self.target_cate_list, \
            self.target_tag_list = self.iterator.get_next()

            # self.click_seq_50size_array = tf.reshape(tf.strings.split(self.user_click_item_list, sep=";").values,
            #                                          shape=[tf.shape(self.user_id)[0], -1])
            # self.user_click_item_list_len = tf.count_nonzero(self.click_seq_50size_array, 1)
            self.user_click_item_list_len = tf.count_nonzero(self.user_click_item_list, 1)
            self.training_init_op = self.iterator.make_initializer(self.dataset)
            # batch_size个数的0，需要注意最后一个batch是不足s一个batch_size的
            self.train_label = tf.zeros(shape=[tf.shape(self.user_id)[0]], dtype=tf.int64)
        else:
            # self.item_list, self.cate_list, self.tag_list = self.read_item_train_data()
            # if self.mode == "pred":
            #     self.user_id, self.user_click_item_list, self.user_click_item_list_len, self.gender, self.client_type \
            #         = self.build_pred_batch_data(train_data_file=tables, batch_size=self.batch_size,
            #                                      num_epochs=self.train_epoches)
            # else:
            #     self.user_id, self.user_click_item_list, self.user_click_item_list_len, self.gender, self.client_type, self.item_click, self.click_len \
            #         = self.build_test_batch_data(train_data_file=tables, batch_size=self.batch_size,
            #                                      num_epochs=self.train_epoches)
            pass

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.item_initializer = tf.truncated_normal([self.ITEM_CNT, self.item_embedding_size],
                                                    stddev=1.0 / math.sqrt(self.item_embedding_size))
        self.cate_initializer = tf.truncated_normal([self.CATE_CNT, self.cate_embedding_size],
                                                    stddev=1.0 / math.sqrt(self.cate_embedding_size))
        self.tag_initializer = tf.truncated_normal([self.TAG_CNT, self.tag_embedding_size],
                                                   stddev=1.0 / math.sqrt(self.tag_embedding_size))

        with tf.name_scope('item_embedding'):
            # Variable
            self.item_embedding = tf.Variable(self.item_initializer, name='item_embedding')
            self.cate_embedding = tf.Variable(self.cate_initializer, name='cate_embedding')
            self.tag_embedding = tf.Variable(self.tag_initializer, name='tag_embedding')
            # get variable
            # self.normal_initializer = tf.random_normal_initializer(0, 1)
            # self.item_embedding = tf.get_variable(name='item_embedding',
            #                                       shape=[self.ITEM_CNT,
            #                                              self.item_embedding_size],
            #                                       initializer=self.normal_initializer)
            # self.item_embedding_b = tf.get_variable("item_b", [self.ITEM_CNT],
            #                                         initializer=tf.constant_initializer(0.0))
            # self.cate_embedding = tf.get_variable(name='cate_embedding',
            #                                       shape=[self.CATE_CNT,
            #                                              self.cate_embedding_size],
            #                                       initializer=self.normal_initializer)
            # self.tag_embedding = tf.get_variable(name='tag_embedding', shape=[self.TAG_CNT,
            #                                                                   self.tag_embedding_size],
            #                                      initializer=self.normal_initializer)
        self.build_model()

        # saved_model 格式的输入，用于eas在线预测
        if self.mode == "train":
            tensor_info_user_id = tf.saved_model.utils.build_tensor_info(self.user_id)
            tensor_info_gender = tf.saved_model.utils.build_tensor_info(self.gender)
            tensor_info_client_type = tf.saved_model.utils.build_tensor_info(self.client_type)
            tensor_info_user_click_item_list = tf.saved_model.utils.build_tensor_info(self.user_click_item_list)
            tensor_info_target_item_list = tf.saved_model.utils.build_tensor_info(self.target_item_list)
            tensor_info_target_cate_list = tf.saved_model.utils.build_tensor_info(self.target_cate_list)
            tensor_info_target_tag_list = tf.saved_model.utils.build_tensor_info(self.target_tag_list)

            # build saved inputs
            self.saved_model_inputs = {
                "user_id": tensor_info_user_id,
                "item_id_list": tensor_info_user_click_item_list,
                "gender": tensor_info_gender,
                "client_type": tensor_info_client_type,
                "target_item_list": tensor_info_target_item_list,
                "target_cate_list": tensor_info_target_cate_list,
                "target_tag_list": tensor_info_target_tag_list
            }

    def decode_train_line(self, line):
        defaults = [[0]] + [['0']] + [[0]] * 2 + [['0']] * 3
        user_id, user_click_item_list, gender, client_type, target_item_list, target_cate_list, target_tag_list = tf.decode_csv(
            line, defaults)

        user_click_item_list = tf.string_to_number(tf.string_split([user_click_item_list], ';').values, tf.int64)
        target_item_list = tf.string_to_number(tf.string_split([target_item_list], ';').values, tf.int64)
        target_cate_list = tf.string_to_number(tf.string_split([target_cate_list], ';').values, tf.int64)
        target_tag_list = tf.string_to_number(tf.string_split([target_tag_list], ';').values, tf.int64)

        user_id = tf.cast(user_id, dtype=tf.int64)
        gender = tf.cast(gender, dtype=tf.int64)
        client_type = tf.cast(client_type, dtype=tf.int64)
        return user_id, user_click_item_list, gender, client_type, target_item_list, target_cate_list, target_tag_list

    def build_user_train_data(self, train_data_file):
        dataset = tf.data.TextLineDataset(train_data_file)
        dataset = dataset.map(self.decode_train_line)
        dataset = dataset.padded_batch(batch_size=self.batch_size, padded_shapes=(
            [], [None], [], [], [None], [None], [None])).repeat(self.train_epoches)

        return dataset

    #
    # def build_user_test_data(self, pred_data_file):
    #     dataset = tf.data.TextLineDataset(pred_data_file)
    #     dataset = dataset.map(self.decode_test_line)
    #     dataset = dataset.padded_batch(batch_size=self.batch_size, padded_shapes=([], [None], [], [None])).repeat(
    #         self.pred_epoches)
    #     return dataset
    #
    # def read_item_train_data(self):
    #     with tf.gfile.Open(self.item_tower_file[0], 'r') as f:
    #         for line in f.readlines():
    #             lines = line.split(',')
    #             item_vocabulary = [int(ele) for ele in lines[0].split(';')]
    #             item_cate_mapping = [int(ele) for ele in lines[1].split(';')]
    #             item_tag_mapping = [int(ele) for ele in lines[2].split(';')]
    #         func = lambda x, y: x if y in x else x + [y]
    #         cate_vocabulary = reduce(func, [[], ] + item_cate_mapping)
    #         tag_vocabulary = reduce(func, [[], ] + item_tag_mapping)
    #         # print(cate_vocabulary)
    #         # print(tag_vocabulary)
    #     return tf.constant(item_vocabulary, dtype=tf.int64), tf.constant(item_cate_mapping, dtype=tf.int64), \
    #            tf.constant(item_tag_mapping, dtype=tf.int64), tf.constant(cate_vocabulary, dtype=tf.int64), tf.constant(
    #         tag_vocabulary, dtype=tf.int64)
    # def read_train_data(self, file_queue):
    #     if self.local:
    #         reader = tf.TextLineReader(skip_header_lines=1)
    #     else:
    #         reader = tf.TableRecordReader()
    #     key, value = reader.read(file_queue)
    #     defaults = [[0]] + [['0']] + [[0]] * 2 + [['0']] * 3
    #     user_id, user_item_click_list, gender, client_type, target_item, target_cate, target_tag = tf.decode_csv(
    #         value, defaults)
    #     user_id = tf.cast(user_id, dtype=tf.int64)
    #     gender = tf.cast(gender, dtype=tf.int64)
    #     client_type = tf.cast(client_type, dtype=tf.int64)
    #     user_item_click_list = tf.string_to_number(tf.string_split([user_item_click_list], ';').values, tf.int64)
    #     target_item = tf.string_to_number(tf.string_split([target_item], ';').values, tf.int64)
    #     target_cate = tf.string_to_number(tf.string_split([target_cate], ';').values, tf.int64)
    #     target_tag = tf.string_to_number(tf.string_split([target_tag], ';').values, tf.int64)
    #     return user_id, user_item_click_list, gender, client_type, target_item, target_cate, target_tag
    #
    # def build_train_batch_data(self, train_data_file, batch_size, num_epochs=None):
    #     file_queue = tf.train.string_input_producer(train_data_file, num_epochs=num_epochs)
    #     user_id, user_item_click_list, gender, client_type, target_item, target_cate, target_tag = self.read_train_data(
    #         file_queue)
    #
    #     # min_after_dequeue = 1000
    #     capacity = 10000
    #     user_id_batch, user_item_click_list_batch, gender_batch, client_type_batch, target_item_batch, target_cate_batch, target_tag_batch \
    #         = tf.train.batch(
    #         [user_id, user_item_click_list, gender, client_type, target_item, target_cate,
    #          target_tag],
    #         batch_size=batch_size, capacity=capacity, num_threads=1, allow_smaller_final_batch=True, dynamic_pad=True
    #         # min_after_dequeue=min_after_dequeue
    #     )
    #
    #     return user_id_batch, user_item_click_list_batch, gender_batch, client_type_batch, target_item_batch, target_cate_batch, target_tag_batch

    # def read_pred_data(self, file_queue):
    #     if self.local:
    #         reader = tf.TextLineReader(skip_header_lines=1)
    #     else:
    #         reader = tf.TableRecordReader()
    #     key, value = reader.read(file_queue)
    #     defaults = [[0]] + [['0']] * 1 + [[0]] * 3
    #     user_id, user_item_click_list, item_click_list_len, gender, client_type = tf.decode_csv(value, defaults)
    #     user_item_click_list = tf.string_to_number(tf.string_split([user_item_click_list], ';').values, tf.int32)
    #     return user_id, user_item_click_list, item_click_list_len, gender, client_type
    #
    # def build_pred_batch_data(self, train_data_file, batch_size, num_epochs=None):
    #     file_queue = tf.train.string_input_producer(train_data_file, num_epochs=num_epochs)
    #     user_id, user_item_click_list, item_click_list_len, gender, client_type = self.read_pred_data(file_queue)
    #
    #     # min_after_dequeue = 1000
    #     capacity = 10000
    #     user_id_batch, user_item_click_list_batch, item_click_list_len_batch, gender_batch, client_type_batch \
    #         = tf.train.batch(
    #         [user_id, user_item_click_list, item_click_list_len, gender, client_type],
    #         batch_size=batch_size, capacity=capacity, num_threads=1, allow_smaller_final_batch=True, dynamic_pad=True
    #         # min_after_dequeue=min_after_dequeue
    #     )
    #
    #     return user_id_batch, user_item_click_list_batch, item_click_list_len_batch, gender_batch, client_type_batch
    #
    # def read_test_data(self, file_queue):
    #     if self.local:
    #         reader = tf.TextLineReader(skip_header_lines=1)
    #     else:
    #         reader = tf.TableRecordReader()
    #     key, value = reader.read(file_queue)
    #     defaults = [[0]] + [['0']] * 1 + [[0]] * 3 + [['0']] + [[0]]
    #     user_id, user_item_click_list, item_click_list_len, gender, client_type, item_click, click_len = tf.decode_csv(
    #         value,
    #         defaults)
    #
    #     user_item_click_list = tf.string_to_number(tf.string_split([user_item_click_list], ';').values, tf.int32)
    #     item_click = tf.string_to_number(tf.string_split([item_click], ';').values, tf.int32)
    #     # user_cate_click_list = tf.string_to_number(tf.string_split([user_cate_click_list], ';').values, tf.int32)
    #     # user_tag_click_list = tf.string_to_number(tf.string_split([user_tag_click_list], ';').values, tf.int32)
    #     return user_id, user_item_click_list, item_click_list_len, gender, client_type, item_click, click_len
    #
    # def build_test_batch_data(self, train_data_file, batch_size, num_epochs=None):
    #     file_queue = tf.train.string_input_producer(train_data_file, num_epochs=num_epochs)
    #     user_id, user_item_click_list, item_click_list_len, gender, client_type, item_click, click_len = self.read_test_data(
    #         file_queue)
    #
    #     # min_after_dequeue = 1000
    #     capacity = 10000
    #     user_id_batch, user_item_click_list_batch, item_click_list_len_batch, gender_batch, client_type_batch, item_click_batch, click_len_batch \
    #         = tf.train.batch(
    #         [user_id, user_item_click_list, item_click_list_len, gender, client_type, item_click, click_len],
    #         batch_size=batch_size, capacity=capacity, num_threads=1, allow_smaller_final_batch=True, dynamic_pad=True
    #         # min_after_dequeue=min_after_dequeue
    #     )
    #
    #     return user_id_batch, user_item_click_list_batch, item_click_list_len_batch, gender_batch, client_type_batch, item_click_batch, click_len_batch

    # def read_item_train_data(self):
    #     item_list = []
    #     cate_list = []
    #     tag_list = []
    #     with tf.gfile.Open(self.item_tower_file[0], 'r') as f:
    #         for line in f.readlines():
    #             lines = line.strip("\n").split(",")
    #             item_list.append(int(lines[0]))
    #             cate_list.append(int(lines[1]))
    #             tag_list.append(int(lines[2]))
    #     return tf.constant(item_list, dtype=tf.int64), tf.constant(cate_list, dtype=tf.int64), tf.constant(tag_list,
    #                                                                                                        dtype=tf.int64)

    # 计算序列的embedding(平均或求和)
    # TODO: 后续优化添加attention层
    def get_seq_embedding(self, item_embedding, user_click_item_list, item_id_list_len_batch, embedding_size, method):
        # user_click_item_list_idx = self.item_table.lookup(self.user_click_item_list)
        embed_init = tf.nn.embedding_lookup(item_embedding, user_click_item_list)
        embedding_mask = tf.sequence_mask(item_id_list_len_batch, tf.shape(user_click_item_list)[1],
                                          dtype=tf.float32)
        embedding_mask = tf.expand_dims(embedding_mask, -1)
        embedding_mask = tf.tile(embedding_mask, [1, 1, embedding_size])
        embedding_mask_2 = embed_init * embedding_mask
        embedding_sum = tf.reduce_sum(embedding_mask_2, 1)

        seq_embedding_final = None
        if method == 'sum':
            seq_embedding_final = embedding_sum
        elif method == 'mean':
            seq_avg_embedding = tf.div(embedding_sum,
                                       tf.cast(tf.tile(tf.expand_dims(item_id_list_len_batch, 1),
                                                       [1, embedding_size]),
                                               tf.float32))
            seq_embedding_final = seq_avg_embedding

        return seq_embedding_final

    def build_model(self):
        user_click_item_list_idx = tf.string_to_hash_bucket_fast(tf.as_string(self.user_click_item_list), self.ITEM_CNT)
        # # User Embedding Layer
        with tf.name_scope("user_tower"):
            with tf.name_scope('user_embedding'):
                user_item_click_sum_embed = self.get_seq_embedding(self.item_embedding,
                                                                   user_click_item_list_idx,
                                                                   self.user_click_item_list_len,
                                                                   self.item_embedding_size,
                                                                   "sum")

                # one-hot feature
                gender_one_hot = tf.one_hot(self.gender, self.GENDER_CNT)
                client_type_one_hot = tf.one_hot(self.client_type, self.CLIENT_TYPE_CNT)

                #   concat embedding

                user_embed_concat = tf.concat(
                    [user_item_click_sum_embed, gender_one_hot, client_type_one_hot], axis=-1)

            with tf.name_scope('layers'):
                user_layer_1 = tf.layers.dense(inputs=user_embed_concat,
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

                # user参数写入summary
                # user_first_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'user_first')
                # user_second_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'user_second')
                # user_final_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'user_final')

                # tf.summary.histogram("user_layer_1_weights", user_first_vars[0])
                # # tf.summary.histogram("user_layer_1_biases", user_first_vars[1])
                # tf.summary.histogram("user_layer_1_output", user_layer_1)
                #
                # tf.summary.histogram("user_layer_2_weights", user_second_vars[0])
                # # tf.summary.histogram("user_layer_2_biases", user_second_vars[1])
                # tf.summary.histogram("user_layer_2_output", user_layer_2)
                # #
                # tf.summary.histogram("user_layer_3_weights", user_final_vars[0])
                # # tf.summary.histogram("user_layer_3_biases", user_final_vars[1])
                # tf.summary.histogram("user_layer_3_output", user_layer_3)

            self.user_embedding_final = user_layer_3

        with tf.name_scope("item_tower"):
            # if self.mode == "train":
            target_item_idx = tf.string_to_hash_bucket_fast(tf.as_string(self.target_item_list), self.ITEM_CNT)
            target_cate_idx = tf.string_to_hash_bucket_fast(tf.as_string(self.target_cate_list), self.CATE_CNT)
            target_tag_idx = tf.string_to_hash_bucket_fast(tf.as_string(self.target_tag_list), self.TAG_CNT)

            item_id_embed = tf.nn.embedding_lookup(self.item_embedding, target_item_idx)
            cate_id_embed = tf.nn.embedding_lookup(self.cate_embedding, target_cate_idx)
            tag_id_embed = tf.nn.embedding_lookup(self.tag_embedding, target_tag_idx)

            target_embed_concat = tf.concat([item_id_embed, cate_id_embed, tag_id_embed], axis=-1)
            with tf.name_scope('item_layers'):
                item_layer_1 = tf.layers.dense(inputs=target_embed_concat,
                                               units=256,
                                               activation=tf.nn.tanh,
                                               name='item_first',
                                               kernel_initializer=tf.initializers.glorot_normal(seed=1234),
                                               use_bias=False
                                               )
                item_layer_2 = tf.layers.dense(inputs=item_layer_1,
                                               units=128,
                                               activation=tf.nn.tanh,
                                               name='item_second',
                                               kernel_initializer=tf.initializers.glorot_normal(seed=1234),
                                               use_bias=False
                                               )

                # item参数写入summary
                # item_first_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'item_first')
                # tf.summary.histogram("item_layer_1_weights", item_first_vars[0])
                # # tf.summary.histogram("item_layer_1_biases", item_first_vars[1])
                # tf.summary.histogram("item_layer_1_output", item_layer_1)

                self.item_embeding_final = item_layer_2

                ## EAS 上运行必须加
                self.user_embedding_final_expand = tf.expand_dims(self.user_embedding_final, 1)
                self.item_embeding_final = tf.transpose(self.item_embeding_final, perm=[0, 2, 1])
                self.logits = tf.squeeze(tf.matmul(self.user_embedding_final_expand, self.item_embeding_final), axis=1)

                # saved_model 输出
                tensor_info_logits = tf.saved_model.utils.build_tensor_info(self.logits)
                self.saved_model_outputs["logits"] = tensor_info_logits
            # else:
            # self.logits = tf.matmul(self.user_embedding_final, self.item_embeding_final,
            #                         transpose_b=True)
            # pass

    def train_model(self):
        # softmax loss

        losses = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.train_label, logits=self.logits))

        softmax_loss = tf.reduce_mean(losses)

        # 计算auc
        auc = self.evaluate()
        # auc = tf.constant(0)
        # l2_norm loss
        regulation_rate = 0.0001
        l2_norm = tf.add_n([
            tf.reduce_sum(tf.multiply(self.user_embedding_final, self.user_embedding_final)),
            tf.reduce_sum(tf.multiply(self.item_embeding_final, self.item_embeding_final)),
        ])
        l2_loss = regulation_rate * l2_norm
        loss_merge = l2_loss + softmax_loss

        # tf.summary.scalar('loss_merge', loss_merge)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        # train_op = optimizer.minimize(loss_merge, global_step=self.global_step)

        var_list = tf.trainable_variables()
        gradients = optimizer.compute_gradients(loss_merge, var_list)
        capped_gradients = [(tf.clip_by_value(grad, -500., 500.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients, global_step=self.global_step)

        return train_op, loss_merge, softmax_loss, auc, self.learning_rate

    def evaluate(self):
        # 模拟二分类方式计算auc
        logit_split_pos, logit_split_neg = tf.split(self.logits, [1, 10], axis=1)
        # 正样本扩展为与负样本一样的维度
        logit_split_pos_expand = tf.tile(logit_split_pos, [1, 10])
        logit_result = logit_split_pos_expand - logit_split_neg
        auc = tf.reduce_mean(tf.to_float(logit_result > 0))
        return auc

    def predict_topk_score(self):

        self.topk_score = tf.nn.top_k(self.logits, self.TOP_K_ITEM_CNT)[0]
        self.topk_idx = tf.nn.top_k(self.logits, self.TOP_K_ITEM_CNT)[1]
        self.user_topk_item = tf.reduce_join(tf.as_string(self.topk_idx), 1, separator=",")
        self.user_topk_score = tf.reduce_join(tf.as_string(self.topk_score), 1, separator=",")

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
