# -*- coding: utf-8 -*-
from functools import reduce

import tensorflow as tf
import math
from tensorflow.python.ops.rnn import dynamic_rnn


class TwoTowerModelFRBPRGRU:
    def __init__(self, train_file_dir,
                 test_file_dir,
                 is_train,
                 item_embedding_size,
                 cate_embedding_size,
                 tag_embedding_size,
                 key_word_embedding_size,
                 batch_size,
                 learning_rate,
                 output_table,
                 local,
                 top_k_num,
                 neg_sample_num, sess):
        self.train_file_dir = train_file_dir
        self.test_file_dir = test_file_dir
        self.is_train = is_train
        self.item_embedding_size = item_embedding_size
        self.cate_embedding_size = cate_embedding_size
        self.tag_embedding_size = tag_embedding_size
        self.key_word_embedding_size = key_word_embedding_size
        self.item_merge_embedding_size = item_embedding_size + cate_embedding_size + tag_embedding_size
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
        self.ITEM_MOD = 600000
        self.CATE_MOD = 30
        self.TAG_MOD = 2000
        self.KEY_WORD_MOD = 800000

        # 误差（1e-8）防止除数为0
        self.EPS = tf.constant(1e-8, tf.float32)
        self.saved_model_outputs = {}
        self.sess = sess

        self.train_epoches = 1
        self.test_epoches = 1
        if local:
            self.PRINT_STEP = 1
            self.SAVE_STEP = 10
        else:
            self.PRINT_STEP = 1000
            self.SAVE_STEP = 30000

        if self.is_train:
            self.dataset = self.build_user_train_data(self.train_file_dir)
            self.iterator = self.dataset.make_one_shot_iterator()

            self.user_id, self.user_click_item_list, self.gender, self.client_type, self.pos_item, self.pos_cate, \
            self.pos_tag, self.pos_key, self.neg_item, self.neg_cate, self.neg_tag, self.neg_key = self.iterator.get_next()

            self.user_click_item_list_len = tf.count_nonzero(self.user_click_item_list, 1)
            self.pos_item_key_len = tf.count_nonzero(self.pos_key, 1)
            self.neg_item_key_len = tf.count_nonzero(self.neg_key, 1)

            self.training_init_op = self.iterator.make_initializer(self.dataset)
            self.train_label = tf.zeros(shape=[tf.shape(self.user_id)[0]], dtype=tf.int64)

        else:
            self.dataset = self.build_user_test_data(self.test_file_dir)
            self.iterator = self.dataset.make_one_shot_iterator()

            self.user_id, self.user_click_item_list, self.gender, self.client_type, self.target_item_list, \
            self.target_cate_list, self.target_tag_list = self.iterator.get_next()

            self.user_click_item_list_len = tf.count_nonzero(self.user_click_item_list, 1)
            self.training_init_op = self.iterator.make_initializer(self.dataset)

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        # self.initializer = tf.glorot_normal_initializer()
        self.selu_initializer = tf.variance_scaling_initializer()
        self.uniform_initializer = tf.random_uniform_initializer(-1, 1)
        # self.normal_initializer = tf.random_normal_initializer(0, 1)
        self.item_initializer = tf.truncated_normal([self.ITEM_MOD, self.item_embedding_size],
                                                    stddev=1.0 / math.sqrt(self.item_embedding_size))
        self.cate_initializer = tf.truncated_normal([self.CATE_MOD, self.cate_embedding_size],
                                                    stddev=1.0 / math.sqrt(self.cate_embedding_size))
        self.tag_initializer = tf.truncated_normal([self.TAG_MOD, self.tag_embedding_size],
                                                   stddev=1.0 / math.sqrt(self.tag_embedding_size))
        self.he_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution='normal')

        with tf.name_scope('item_embedding'):
            self.item_embedding = tf.get_variable(name='item_embedding',
                                                  shape=[self.ITEM_MOD,
                                                         self.item_embedding_size],
                                                  initializer=self.uniform_initializer)
            self.item_embedding_b = tf.get_variable("item_b", [self.ITEM_MOD],
                                                    initializer=tf.constant_initializer(0.0))
            self.cate_embedding = tf.get_variable(name='cate_embedding',
                                                  shape=[self.CATE_MOD,
                                                         self.cate_embedding_size],
                                                  initializer=self.uniform_initializer)
            self.tag_embedding = tf.get_variable(name='tag_embedding', shape=[self.TAG_MOD,
                                                                              self.tag_embedding_size],
                                                 initializer=self.uniform_initializer)
            self.key_word_embedding = tf.get_variable(name='key_word_embedding', shape=[self.KEY_WORD_MOD,
                                                                                        self.key_word_embedding_size],
                                                      initializer=self.uniform_initializer)

        self.build_model()

        # if self.is_train:
        #     tensor_info_user_id = tf.saved_model.utils.build_tensor_info(self.user_id)
        #     tensor_info_gender = tf.saved_model.utils.build_tensor_info(self.gender)
        #     tensor_info_client_type = tf.saved_model.utils.build_tensor_info(self.client_type)
        #     tensor_info_user_click_item_list = tf.saved_model.utils.build_tensor_info(self.user_click_item_list)
        #     tensor_info_target_item_list = tf.saved_model.utils.build_tensor_info(self.target_item_list)
        #     tensor_info_target_cate_list = tf.saved_model.utils.build_tensor_info(self.target_cate_list)
        #     tensor_info_target_tag_list = tf.saved_model.utils.build_tensor_info(self.target_tag_list)
        #
        #     # build saved inputs
        #     self.saved_model_inputs = {
        #         "user_id": tensor_info_user_id,
        #         "item_id_list": tensor_info_user_click_item_list,
        #         "gender": tensor_info_gender,
        #         "client_type": tensor_info_client_type,
        #         "target_item_list": tensor_info_target_item_list,
        #         "target_cate_list": tensor_info_target_cate_list,
        #         "target_tag_list": tensor_info_target_tag_list
        #     }

    def decode_test_line(self, line):
        defaults = [[0]] + [['0']] + [[0]] * 2 + [['0']] * 3
        user_id, user_click_item_list, gender, client_type, pos_item, pos_cate, pos_key, pos_tag, neg_item, neg_cate, neg_tag, neg_key = tf.decode_csv(
            line, defaults)

        user_click_item_list = tf.string_to_number(tf.string_split([user_click_item_list], ';').values, tf.int64)
        # pos_item = tf.string_to_number(tf.string_split([pos_item], ';').values, tf.int64)
        # target_cate = tf.string_to_number(tf.string_split([target_cate], ';').values, tf.int64)
        # target_tag = tf.string_to_number(tf.string_split([target_tag], ';').values, tf.int64)

        user_id = tf.cast(user_id, dtype=tf.int64)
        gender = tf.cast(gender, dtype=tf.int64)
        client_type = tf.cast(client_type, dtype=tf.int64)
        return user_id, user_click_item_list, gender, client_type, pos_item, pos_cate, pos_tag, pos_key, neg_item, neg_cate, neg_tag, neg_key

    def decode_train_line(self, line):
        defaults = [[0]] + [['0']] + [[0]] * 5 + [['0']] + [[0]] * 3 + [['0']]
        user_id, user_click_item_list, gender, client_type, pos_item, pos_cate, pos_tag, pos_key, neg_item, neg_cate, \
        neg_tag, neg_key = tf.decode_csv(line, defaults)

        user_click_item_list = tf.string_to_number(tf.string_split([user_click_item_list], ';').values, tf.int64)
        pos_key = tf.string_to_number(tf.string_split([pos_key], ';').values, tf.int64)
        neg_key = tf.string_to_number(tf.string_split([neg_key], ';').values, tf.int64)
        # target_tag = tf.string_to_number(tf.string_split([target_tag], ';').values, tf.int64)

        user_id = tf.cast(user_id, dtype=tf.int64)
        gender = tf.cast(gender, dtype=tf.int64)
        client_type = tf.cast(client_type, dtype=tf.int64)
        return user_id, user_click_item_list, gender, client_type, pos_item, pos_cate, pos_tag, pos_key, neg_item, \
               neg_cate, neg_tag, neg_key

    def build_user_train_data(self, train_data_file):
        dataset = tf.data.TextLineDataset(train_data_file)
        dataset = dataset.map(self.decode_train_line)
        dataset = dataset.padded_batch(batch_size=self.batch_size, padded_shapes=(
            [], [None], [], [], [], [], [], [None], [], [], [], [None])).repeat(self.train_epoches)
        return dataset

    def build_user_test_data(self, test_data_file):
        dataset = tf.data.TextLineDataset(test_data_file)
        dataset = dataset.map(self.decode_train_line)
        dataset = dataset.padded_batch(batch_size=self.batch_size, padded_shapes=(
            [], [None], [], [], [None], [None], [None])).repeat(self.test_epoches)
        return dataset

    # def read_item_train_data(self):
    #     item_vocabulary = []
    #     item_cate = []
    #     item_tag_mapping = []
    #     with tf.gfile.Open(self.item_tower_file[0], 'r') as f:
    #         for line in f.readlines():
    #             lines = line.split(',')
    #             item_vocabulary = [int(ele) for ele in lines[0].split(';')]
    #             item_cate = [int(ele) for ele in lines[1].split(';')]
    #             item_tag_mapping = [int(ele) for ele in lines[2].split(';')]
    #         # func = lambda x, y: x if y in x else x + [y]
    #         # cate_vocabulary = reduce(func, [[], ] + item_cate)
    #         # tag_vocabulary = reduce(func, [[], ] + item_tag_mapping)
    #         # print(cate_vocabulary)
    #         # print(tag_vocabulary)
    #     return tf.constant(item_vocabulary, dtype=tf.int64), tf.constant(item_cate, dtype=tf.int64), \
    #            tf.constant(item_tag_mapping, dtype=tf.int64)
    #     #    tf.constant(cate_vocabulary, dtype=tf.int64), tf.constant(
    #     # tag_vocabulary, dtype=tf.int64)

    # 计算序列的平均embedding
    def get_seq_embedding(self, item_embedding, user_click_item_list, item_id_list_len_batch, embedding_size, method):
        # user_click_item_list_idx = self.item_table.lookup(self.user_click_item_list)
        embed_init = tf.nn.embedding_lookup(item_embedding, user_click_item_list)
        embedding_mask = tf.sequence_mask(item_id_list_len_batch, tf.shape(user_click_item_list)[1],
                                          dtype=tf.float32)
        embedding_mask = tf.expand_dims(embedding_mask, -1)
        embedding_mask = tf.tile(embedding_mask, [1, 1, embedding_size])
        embedding_mask_2 = embed_init * embedding_mask
        embedding_sum = tf.reduce_sum(embedding_mask_2, 1)

        if method == 'sum':
            return embedding_sum
        elif method == 'avg':
            seq_avg_embedding = tf.div(embedding_sum,
                                       tf.cast(tf.tile(tf.expand_dims(item_id_list_len_batch, 1),
                                                       [1, embedding_size]),
                                               tf.float32) + self.EPS)

            return seq_avg_embedding

    def build_model(self):
        user_click_item_list_idx = tf.string_to_hash_bucket_fast(tf.as_string(self.user_click_item_list),
                                                                 self.ITEM_MOD)

        # embed_init = tf.nn.embedding_lookup(self.item_embedding, user_click_item_list_idx)
        # # User Embedding Layer
        with tf.name_scope("user_tower"):
            with tf.name_scope('user_embedding'):
                user_item_click_avg_embed = self.get_seq_embedding(self.item_embedding,
                                                                   user_click_item_list_idx,
                                                                   self.user_click_item_list_len,
                                                                   self.item_embedding_size,
                                                                   'sum')

                # gru_cell = tf.nn.rnn_cell.GRUCell(self.item_embedding_size)
                #
                # # initial_state = tf.nn.rnn_cell.zero_state(self.batch_size, dtype=tf.float32)
                # output, state = dynamic_rnn(cell=gru_cell, inputs=embed_init, dtype=tf.float32,
                #                             sequence_length=self.user_click_item_list_len)

                gender_one_hot = tf.one_hot(self.gender, self.GENDER_CNT)
                client_type_one_hot = tf.one_hot(self.client_type, self.CLIENT_TYPE_CNT)

                user_embed_concat = tf.concat(
                    [user_item_click_avg_embed, gender_one_hot, client_type_one_hot], axis=-1)

            with tf.name_scope('layers'):
                bn = tf.layers.batch_normalization(inputs=user_embed_concat, name='user_bn', )
                user_layer_1 = tf.layers.dense(bn, 1024, activation=tf.nn.tanh, name='user_first',
                                               kernel_initializer=tf.glorot_normal_initializer())
                user_layer_2 = tf.layers.dense(user_layer_1, 512, activation=tf.nn.tanh, name='user_second',
                                               kernel_initializer=tf.glorot_normal_initializer())
                user_layer_3 = tf.layers.dense(user_layer_2,
                                               self.item_embedding_size,
                                               activation=tf.nn.tanh, name='user_final',
                                               kernel_initializer=tf.glorot_normal_initializer())

            self.user_embedding_final = user_layer_3

        with tf.name_scope("item_tower"):
            # self.item_idx = self.item_table.lookup(self.item_vobabulary)
            # self.item_cate_mapping_idx = self.cate_table.lookup(self.item_cate_mapping)
            # self.item_tag_mapping_idx = self.tag_table.lookup(self.item_tag_mapping)

            pos_item_idx = tf.string_to_hash_bucket_fast(tf.as_string(self.pos_item), self.ITEM_MOD)
            pos_cate_idx = tf.string_to_hash_bucket_fast(tf.as_string(self.pos_cate), self.CATE_MOD)
            pos_tag_idx = tf.string_to_hash_bucket_fast(tf.as_string(self.pos_tag), self.TAG_MOD)

            neg_item_idx = tf.string_to_hash_bucket_fast(tf.as_string(self.neg_item), self.ITEM_MOD)
            neg_cate_idx = tf.string_to_hash_bucket_fast(tf.as_string(self.neg_cate), self.CATE_MOD)
            neg_tag_idx = tf.string_to_hash_bucket_fast(tf.as_string(self.neg_tag), self.TAG_MOD)

            pos_item_id_embed = tf.nn.embedding_lookup(self.item_embedding, pos_item_idx)
            pos_cate_id_embed = tf.nn.embedding_lookup(self.cate_embedding, pos_cate_idx)
            pos_tag_id_embed = tf.nn.embedding_lookup(self.tag_embedding, pos_tag_idx)

            pos_key_word_embed = self.get_seq_embedding(self.key_word_embedding,
                                                        self.pos_key,
                                                        self.pos_item_key_len,
                                                        self.key_word_embedding_size,
                                                        'avg')

            self.pos_embed_concat = tf.concat(
                [pos_item_id_embed, pos_cate_id_embed, pos_tag_id_embed, pos_key_word_embed], axis=-1)

            neg_item_id_embed = tf.nn.embedding_lookup(self.item_embedding, neg_item_idx)
            neg_cate_id_embed = tf.nn.embedding_lookup(self.cate_embedding, neg_cate_idx)
            neg_tag_id_embed = tf.nn.embedding_lookup(self.tag_embedding, neg_tag_idx)
            neg_key_word_embed = self.get_seq_embedding(self.key_word_embedding,
                                                        self.neg_key,
                                                        self.neg_item_key_len,
                                                        self.key_word_embedding_size,
                                                        'avg')

            self.neg_embed_concat = tf.concat(
                [neg_item_id_embed, neg_cate_id_embed, neg_tag_id_embed, neg_key_word_embed], axis=-1)

            self.item_concat = tf.concat([self.pos_embed_concat, self.neg_embed_concat], axis=0)

            with tf.name_scope('item_layers'):
                bn = tf.layers.batch_normalization(inputs=self.item_concat, name='item_bn', )
                item_layer_1 = tf.layers.dense(bn, self.item_embedding_size, activation=tf.nn.tanh, name='item_first',
                                               kernel_initializer=tf.glorot_normal_initializer())
                # item_layer_2 = tf.layers.dense(item_layer_1, self.item_embedding_size, activation=tf.nn.tanh,
                #                                name='item_second',
                #                                kernel_initializer=tf.glorot_normal_initializer())
                # item_layer_3 = tf.layers.dense(item_layer_2, self.item_embedding_size, activation=tf.nn.tanh,
                #                                name='item_third',
                #                                kernel_initializer=tf.glorot_normal_initializer())
            self.item_embed_output = item_layer_1
            self.item_embed_split = tf.split(self.item_embed_output, 2, 0)
            self.pos_embed_final = tf.squeeze(self.item_embed_split[0])
            self.neg_embed_final = tf.squeeze(self.item_embed_split[1])

            # if self.is_train:
            # temp_user_embedding_final = tf.expand_dims(self.user_embedding_final, 1)
            # target_embed_final = tf.transpose(item_embed_output, perm=[0, 2, 1])
            # self.logits = tf.squeeze(tf.matmul(temp_user_embedding_final, target_embed_final), axis=1)
            #
            # tensor_info_logits = tf.saved_model.utils.build_tensor_info(self.logits)
            # self.saved_model_outputs["logits"] = tensor_info_logits

    def train_model(self):

        bpr = tf.reduce_sum(
            tf.multiply(self.user_embedding_final, (self.pos_embed_final - self.neg_embed_final)), 1, keepdims=True)

        auc = tf.reduce_mean(tf.to_float(bpr > 0))

        regulation_rate = 0.0001

        l2_norm = tf.add_n([
            tf.reduce_sum(tf.multiply(self.user_embedding_final, self.user_embedding_final)),
            tf.reduce_sum(tf.multiply(self.pos_embed_final, self.pos_embed_final)),
            tf.reduce_sum(tf.multiply(self.neg_embed_final, self.neg_embed_final))
        ])

        bpr_losses = -tf.reduce_mean(tf.log(tf.clip_by_value(tf.sigmoid(bpr), 1e-8, 1.0)))
        l2_loss = regulation_rate * l2_norm
        loss_merge = l2_loss
        # cost = tf.reduce_mean(bpr_losses)
        # lr = tf.maximum(1e-5,
        #                 tf.train.exponential_decay(self.learning_rate, self.global_step, 40000, 0.9, staircase=True))

        # gradients not clip
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        # train_op = optimizer.minimize(bpr_losses, global_step=self.global_step)

        # gradients clip
        # trainable_params = tf.trainable_variables()
        # gradients = tf.gradients(bpr_losses, trainable_params)
        # clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
        # train_op = optimizer.apply_gradients(
        #     zip(clip_gradients, trainable_params), global_step=self.global_step)

        var_list = tf.trainable_variables()
        gradients = optimizer.compute_gradients(bpr_losses, var_list)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients, global_step=self.global_step)

        return train_op, loss_merge, bpr_losses, auc, self.learning_rate

    def evaluate_auc(self):
        bpr = tf.reduce_sum(
            tf.multiply(self.user_embedding_final, (self.pos_embed_final - self.neg_embed_final)), 1, keepdims=True)

        auc = tf.reduce_mean(tf.to_float(bpr > 0))
        return auc

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
