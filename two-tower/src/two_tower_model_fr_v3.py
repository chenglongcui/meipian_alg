# -*- coding: utf-8 -*-
from functools import reduce

import tensorflow as tf
import math
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def
from tensorflow.python.saved_model import tag_constants


class TwoTowerModelFRV2:
    def __init__(self, train_file_dir, test_file_dir, item_tower_file, is_train, item_embedding_size,
                 cate_embedding_size,
                 tag_embedding_size,
                 batch_size, learning_rate, item_count, cate_count, tag_count, output_table, local, top_k_num,
                 neg_sample_num, sess):
        self.train_file_dir = train_file_dir
        self.test_file_dir = test_file_dir
        self.item_tower_file = item_tower_file
        self.is_train = is_train
        self.item_embedding_size = item_embedding_size
        self.cate_embedding_size = cate_embedding_size
        self.tag_embedding_size = tag_embedding_size
        self.item_merge_embedding_size = item_embedding_size + cate_embedding_size + tag_embedding_size
        self.batch_size = batch_size
        self.learning_rate = tf.constant(learning_rate)
        self.table_path = "odps://dpdefault_68367/tables/"
        self.output_path = self.table_path + output_table
        self.local = local
        self.item_count = item_count
        self.cate_count = cate_count
        self.tag_count = tag_count
        self.GENDER_CNT = 3
        self.AGE_CNT = 5
        self.CONSUME_LEVEL_CNT = 3
        self.MEIPIAN_AGE_CNT = 15
        self.NUM_SAMPLED = neg_sample_num
        self.NUM_OF_TABLE_COLUMNS = 3
        self.TOP_K_ITEM_CNT = top_k_num
        self.NUM_OF_EPOCHS = 1

        # 预留位置
        self.NUM_ITEM_OOV_BUCKET = 500000
        self.NUM_CATE_OOV_BUCKET = 5
        self.NUM_TAG_OOV_BUCKET = 10
        self.ITEM_MOD = 800000
        self.CATE_MOD = 30
        self.TAG_MOD = 300

        # 误差（1e-8）防止除数为0
        self.EPS = tf.constant(1e-8, tf.float32)
        self.saved_model_outputs = {}
        self.sess = sess

        self.train_epoches = 1
        self.pred_epoches = 1
        if local:
            self.PRINT_STEP = 1
            self.SAVE_STEP = 10
        else:
            self.PRINT_STEP = 1000
            self.SAVE_STEP = 30000

        ## train dataset
        self.train_dataset = self.build_user_train_data(self.train_file_dir)
        self.train_iterator = self.train_dataset.make_one_shot_iterator()

        self.user_id, self.user_click_item_list, self.gender, self.target_item_list, self.target_cate_list, self.target_tag_list = self.train_iterator.get_next()
        self.user_click_item_list_len = tf.count_nonzero(self.user_click_item_list, 1)
        self.training_init_op = self.train_iterator.make_initializer(self.train_dataset)
        self.train_label = tf.zeros(shape=[tf.shape(self.user_id)[0]], dtype=tf.int64)

        ## test datasets
        self.test_dataset = self.build_user_test_data(self.test_file_dir)
        self.test_iterator = self.test_dataset.make_one_shot_iterator()

        self.test_user_id, self.test_user_click_item_list, self.test_gender, self.true_click = self.test_iterator.get_next()
        self.testing_init_op = self.test_iterator.make_initializer(self.test_dataset)

        self.item_vobabulary, self.item_cate_mapping, self.item_tag_mapping, self.cate_vocabulary, \
        self.tag_vocabulary = self.read_item_train_data()

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        # self.initializer = tf.glorot_normal_initializer()
        self.selu_initializer = tf.variance_scaling_initializer()
        self.initializer = tf.random_uniform_initializer(0, 1, seed=1234, dtype=tf.float32)
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
                                                  initializer=self.initializer)
            self.item_embedding_b = tf.get_variable("item_b", [self.ITEM_MOD],
                                                    initializer=tf.constant_initializer(0.0))
            self.cate_embedding = tf.get_variable(name='cate_embedding',
                                                  shape=[self.CATE_MOD,
                                                         self.cate_embedding_size],
                                                  initializer=self.initializer)
            self.tag_embedding = tf.get_variable(name='tag_embedding', shape=[self.TAG_MOD,
                                                                              self.tag_embedding_size],
                                                 initializer=self.initializer)

        self.build_model()

        if self.is_train:
            tensor_info_user_id = tf.saved_model.utils.build_tensor_info(self.user_id)
            tensor_info_gender = tf.saved_model.utils.build_tensor_info(self.gender)
            tensor_info_user_click_item_list = tf.saved_model.utils.build_tensor_info(self.user_click_item_list)
            tensor_info_target_item_list = tf.saved_model.utils.build_tensor_info(self.target_item_list)
            tensor_info_target_cate_list = tf.saved_model.utils.build_tensor_info(self.target_cate_list)
            tensor_info_target_tag_list = tf.saved_model.utils.build_tensor_info(self.target_tag_list)

            # build saved inputs
            self.saved_model_inputs = {
                "user_id": tensor_info_user_id,
                "item_id_list": tensor_info_user_click_item_list,
                "gender": tensor_info_gender,
                "target_item_list": tensor_info_target_item_list,
                "target_cate_list": tensor_info_target_cate_list,
                "target_tag_list": tensor_info_target_tag_list
            }

    def decode_test_line(self, line):
        defaults = [[0]] + [['0']] + [[0]] + [['0']]
        user_id, user_click_item_list, gender, true_click = tf.decode_csv(line, defaults)

        item_id_list = tf.string_to_number(tf.string_split([user_click_item_list], ';').values, tf.int64)
        true_click = tf.string_to_number(tf.string_split([true_click], ';').values, tf.int64)
        # user_click_cate_list = tf.string_to_number(tf.string_split([user_click_cate_list], ';').values, tf.int64)
        # user_click_tag_list = tf.string_to_number(tf.string_split([user_click_tag_list], ';').values, tf.int64)
        # label_list = tf.string_to_number(tf.string_split([label_list], ';').values, tf.int64)

        user_id = tf.cast(user_id, dtype=tf.int64)
        gender = tf.cast(gender, dtype=tf.int64)
        # target = tf.cast(target, dtype=tf.int64)
        return user_id, item_id_list, gender, true_click

    def decode_train_line(self, line):
        defaults = [[0]] + [['0']] + [[0]] + [['0']] * 3
        user_id, user_click_item_list, gender, target_item_list, target_cate_list, target_tag_list = tf.decode_csv(
            line, defaults)

        user_click_item_list = tf.string_to_number(tf.string_split([user_click_item_list], ';').values, tf.int64)
        target_item_list = tf.string_to_number(tf.string_split([target_item_list], ';').values, tf.int64)
        target_cate_list = tf.string_to_number(tf.string_split([target_cate_list], ';').values, tf.int64)
        target_tag_list = tf.string_to_number(tf.string_split([target_tag_list], ';').values, tf.int64)

        user_id = tf.cast(user_id, dtype=tf.int64)
        gender = tf.cast(gender, dtype=tf.int64)
        return user_id, user_click_item_list, gender, target_item_list, target_cate_list, target_tag_list

    def build_user_train_data(self, train_data_file):
        dataset = tf.data.TextLineDataset(train_data_file)
        dataset = dataset.map(self.decode_train_line)
        dataset = dataset.padded_batch(batch_size=self.batch_size, padded_shapes=(
            [], [None], [], [None], [None], [None])).repeat(self.train_epoches)
        return dataset

    def build_user_test_data(self, pred_data_file):
        dataset = tf.data.TextLineDataset(pred_data_file)
        dataset = dataset.map(self.decode_test_line)
        dataset = dataset.padded_batch(batch_size=self.batch_size, padded_shapes=([], [None], [], [None])).repeat(
            self.pred_epoches)
        return dataset

    def read_item_train_data(self):
        with tf.gfile.Open(self.item_tower_file[0], 'r') as f:
            for line in f.readlines():
                lines = line.split(',')
                item_vocabulary = [int(ele) for ele in lines[0].split(';')]
                item_cate_mapping = [int(ele) for ele in lines[1].split(';')]
                item_tag_mapping = [int(ele) for ele in lines[2].split(';')]
            func = lambda x, y: x if y in x else x + [y]
            cate_vocabulary = reduce(func, [[], ] + item_cate_mapping)
            tag_vocabulary = reduce(func, [[], ] + item_tag_mapping)
            # print(cate_vocabulary)
            # print(tag_vocabulary)
        return tf.constant(item_vocabulary, dtype=tf.int64), tf.constant(item_cate_mapping, dtype=tf.int64), \
               tf.constant(item_tag_mapping, dtype=tf.int64), tf.constant(cate_vocabulary, dtype=tf.int64), tf.constant(
            tag_vocabulary, dtype=tf.int64)

    # 计算序列的平均embedding
    def get_seq_avg_embedding(self, item_embedding, user_click_item_list, item_id_list_len_batch, embedding_size):
        # user_click_item_list_idx = self.item_table.lookup(self.user_click_item_list)
        embed_init = tf.nn.embedding_lookup(item_embedding, user_click_item_list)
        embedding_mask = tf.sequence_mask(item_id_list_len_batch, tf.shape(user_click_item_list)[1],
                                          dtype=tf.float32)
        embedding_mask = tf.expand_dims(embedding_mask, -1)
        embedding_mask = tf.tile(embedding_mask, [1, 1, embedding_size])
        embedding_mask_2 = embed_init * embedding_mask
        embedding_sum = tf.reduce_sum(embedding_mask_2, 1)

        # seq_avg_embedding = tf.div(embedding_sum,
        #                            tf.cast(tf.tile(tf.expand_dims(item_id_list_len_batch, 1),
        #                                            [1, embedding_size]),
        #                                    tf.float32) + self.EPS)
        return embedding_sum

    def build_model(self):
        # self.mask = self.mask_zero_sequence(self.user_click_item_list)
        # self.user_click_item_list_idx = self.item_table.lookup(self.user_click_item_list)
        # self.target_idx = self.item_table.lookup(self.target)
        # self.target_idx = self.item_table.lookup(self.target)
        item_lenth = tf.count_nonzero(self.user_click_item_list, 1)
        self.user_click_item_list_idx = tf.string_to_hash_bucket_fast(tf.as_string(self.user_click_item_list),
                                                                      self.ITEM_MOD)

        # # User Embedding Layer
        with tf.name_scope("user_tower"):
            with tf.name_scope('user_embedding'):
                self.user_item_click_avg_embed = self.get_seq_avg_embedding(self.item_embedding,
                                                                            self.user_click_item_list_idx, item_lenth,
                                                                            self.item_embedding_size)

                self.gender_one_hot = tf.one_hot(self.gender, self.GENDER_CNT)

                #   concat embedding
                # self.user_embed = tf.concat(
                #     [self.user_item_click_embed, self.user_cate_click_embed, self.user_tag_click_embed, self.gender_one_hot],
                #     axis=1)
                self.gender_embedding = tf.get_variable(name='gender_embedding', shape=[self.GENDER_CNT, 1],
                                                        initializer=self.initializer)
                self.gender_embed = tf.nn.embedding_lookup(self.gender_embedding, self.gender)

                # self.user_embed = tf.concat([self.user_item_click_avg_embed, self.gender_embed], axis=-1)
                # print(tf.shape(gender_one_hot))
                # self.gender_one_hot = tf.squeeze(gender_one_hot)
                # self.user_embed = tf.concat([self.user_item_click_avg_embed, self.gender_one_hot], axis=-1)
                self.user_embed = self.user_item_click_avg_embed

            with tf.name_scope('layers'):
                bn = tf.layers.batch_normalization(inputs=self.user_embed, name='user_bn')
                user_layer_1 = tf.layers.dense(bn, 256, activation=tf.nn.relu, name='user_first')
                user_layer_2 = tf.layers.dense(user_layer_1, 128, activation=tf.nn.relu, name='user_second')
                user_layer_3 = tf.layers.dense(user_layer_2, 64, activation=tf.nn.relu, name='user_third')
                user_layer_4 = tf.layers.dense(user_layer_3, self.item_embedding_size, activation=tf.nn.relu,
                                               name='user_final')

            self.user_embedding_final = user_layer_4

        with tf.name_scope("item_tower"):
            # self.item_idx = self.item_table.lookup(self.item_vobabulary)
            # self.item_cate_mapping_idx = self.cate_table.lookup(self.item_cate_mapping)
            # self.item_tag_mapping_idx = self.tag_table.lookup(self.item_tag_mapping)
            if self.is_train:
                self.target_item_idx = tf.string_to_hash_bucket_fast(tf.as_string(self.target_item_list), self.ITEM_MOD)
                self.target_cate_idx = tf.string_to_hash_bucket_fast(tf.as_string(self.target_cate_list), self.CATE_MOD)
                self.target_tag_idx = tf.string_to_hash_bucket_fast(tf.as_string(self.target_tag_list), self.TAG_MOD)

            else:
                self.target_item_idx = tf.string_to_hash_bucket_fast(tf.as_string(self.item_vobabulary), self.ITEM_MOD)
                self.target_cate_idx = tf.string_to_hash_bucket_fast(tf.as_string(self.item_cate_mapping),
                                                                     self.CATE_MOD)
                self.target_tag_idx = tf.string_to_hash_bucket_fast(tf.as_string(self.item_tag_mapping), self.TAG_MOD)

            self.item_id_embed = tf.nn.embedding_lookup(self.item_embedding, self.target_item_idx)
            self.item_id_bias = tf.nn.embedding_lookup(self.item_embedding_b, self.target_tag_idx)
            self.cate_id_embed = tf.nn.embedding_lookup(self.cate_embedding, self.target_cate_idx)
            self.tag_id_embed = tf.nn.embedding_lookup(self.tag_embedding, self.target_tag_idx)

            self.item_embed_merge = tf.concat(
                [self.item_id_embed, self.cate_id_embed, self.tag_id_embed],
                axis=-1)
            bn = tf.layers.batch_normalization(inputs=self.item_embed_merge, name='item_bn')
            item_layer_1 = tf.layers.dense(bn, 256, activation=tf.nn.relu,
                                           name='item_first')
            item_layer_2 = tf.layers.dense(item_layer_1, 128, activation=tf.nn.relu, name='item_second',
                                           )
            item_layer_3 = tf.layers.dense(item_layer_2, 64, activation=tf.nn.relu,
                                           name='item_third')
            item_layer_4 = tf.layers.dense(item_layer_3, self.item_embedding_size, activation=tf.nn.relu,
                                           name='item_final')

            self.item_embeding_final = item_layer_4
            if self.is_train:
                # self.label_list_idx = self.item_table.lookup(self.label_list)
                self.user_embedding_final = tf.expand_dims(self.user_embedding_final, 1)
                self.item_embeding_final = tf.transpose(self.item_embeding_final, perm=[0, 2, 1])
                self.logits = tf.squeeze(tf.matmul(self.user_embedding_final, self.item_embeding_final),
                                         axis=1) + self.item_id_bias
                tensor_info_logits = tf.saved_model.utils.build_tensor_info(self.logits)
                self.saved_model_outputs["logits"] = tensor_info_logits
            else:
                self.logits = tf.matmul(self.user_embedding_final, self.item_embeding_final,
                                        transpose_b=True) + self.item_id_bias

    def train_model(self):
        # self.user_embedding_final = tf.expand_dims(self.user_embedding_final, 1)
        # self.item_embeding_final = tf.transpose(self.item_embeding_final, perm=[0, 2, 1])
        # self.logits = tf.squeeze(tf.matmul(self.user_embedding_final, self.item_embeding_final), axis=1)
        losses = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.train_label, logits=self.logits))

        cost = tf.reduce_mean(losses)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        # gradients clip
        self.trainable_params = tf.trainable_variables()
        gradients = tf.gradients(cost, self.trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
        train_op = optimizer.apply_gradients(
            zip(clip_gradients, self.trainable_params), global_step=self.global_step)
        return train_op, cost

    def predict_topk_score(self):

        self.topk_score = tf.nn.top_k(self.logits, self.TOP_K_ITEM_CNT)[0]
        self.topk_idx = tf.nn.top_k(self.logits, self.TOP_K_ITEM_CNT)[1]
        self.user_topk_item = tf.reduce_join(tf.as_string(self.topk_idx), 1, separator=",")
        self.user_topk_score = tf.reduce_join(tf.as_string(self.topk_score), 1, separator=",")

    def write_table(self):
        writer = tf.TableRecordWriter(self.output_path)
        write_to_table = writer.write(range(self.NUM_OF_TABLE_COLUMNS),
                                      [tf.as_string(self.user_id),
                                       self.user_topk_item,
                                       self.user_topk_score])
        return write_to_table

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
