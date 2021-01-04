# -*- coding: utf-8 -*-
import tensorflow as tf
import math


class TwoTowerModel:
    def __init__(self, tables, item_tower_file, is_train, item_embedding_size, cate_embedding_size, tag_embedding_size,
                 batch_size, learning_rate, item_count, cate_count, tag_count, output_table, local, top_k_num,
                 neg_sample_num):
        self.tables = tables
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
        self.GENDER_CNT = 2
        self.AGE_CNT = 5
        self.CONSUME_LEVEL_CNT = 3
        self.MEIPIAN_AGE_CNT = 15
        self.NUM_SAMPLED = neg_sample_num
        self.NUM_OF_TABLE_COLUMNS = 3
        self.TOP_K_ITEM_CNT = top_k_num
        self.NUM_OF_EPOCHS = 1
        self.EPS = tf.constant(1e-8, tf.float32)

        self.epoches = 1
        if local:
            self.PRINT_STEP = 1
        else:
            self.PRINT_STEP = 1000

        if self.is_train:
            self.user_batch, self.item_id_list_batch, self.item_id_list_len_batch, self.cate_id_list_batch, self.cate_id_list_len_batch, \
            self.tag_id_list_batch, self.tag_id_list_len_batch, self.gender_batch, self.label_batch = self.get_user_train_batch_data(
                self.tables, self.batch_size, num_epochs=self.NUM_OF_EPOCHS)
            self.item_batch, self.item_cate_id_list_batch, self.item_tag_id_list_batch, = self.read_item_train_data(
                self.item_tower_file)
        else:
            self.user_batch, self.item_id_list_batch, self.item_id_list_len_batch, self.cate_id_list_batch, self.cate_id_list_len_batch, \
            self.tag_id_list_batch, self.tag_id_list_len_batch, self.gender_batch = \
                self.create_test_pipeline(tables, batch_size, num_epochs=self.NUM_OF_EPOCHS)
            self.item_batch, self.item_cate_id_list_batch, self.item_tag_id_list_batch, = self.read_item_train_data(
                self.item_tower_file)

        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.initializer = tf.random_uniform_initializer(-1, 1, seed=1234, dtype=tf.float32)
        self.item_initializer = tf.truncated_normal([self.item_count, self.item_embedding_size],
                                                    stddev=1.0 / math.sqrt(self.item_embedding_size))
        self.cate_initializer = tf.truncated_normal([self.cate_count, self.cate_embedding_size],
                                                    stddev=1.0 / math.sqrt(self.cate_embedding_size))
        self.tag_initializer = tf.truncated_normal([self.tag_count, self.tag_embedding_size],
                                                   stddev=1.0 / math.sqrt(self.tag_embedding_size))
        self.he_initializer = tf.variance_scaling_initializer(mode="fan_avg")

        with tf.name_scope('item_embedding'):
            self.item_embedding = tf.get_variable(name='item_embedding',
                                                  shape=[self.item_count, self.item_embedding_size],
                                                  initializer=self.initializer)
            self.item_embedding_b = tf.get_variable("item_b", [self.item_count],
                                                    initializer=tf.constant_initializer(0.0))
            self.cate_embedding = tf.get_variable(name='cate_embedding',
                                                  shape=[self.cate_count, self.cate_embedding_size],
                                                  initializer=self.initializer)
            self.tag_embedding = tf.get_variable(name='tag_embedding', shape=[self.tag_count, self.tag_embedding_size],
                                                 initializer=self.initializer)
        # self.build_item_tower()
        self.build_user_model()

    def read_user_train_data(self, file_queue):
        if self.local:
            reader = tf.TextLineReader(skip_header_lines=1)
        else:
            reader = tf.TableRecordReader()
        key, value = reader.read(file_queue)
        defaults = [[0]] + [['0']] + [[0]] + [['0']] + [[0]] + [['0']] + [[0]] * 3
        user_id, item_id_list, item_id_list_len, cate_id_list, cat_id_list_len, tag_id_list, tag_id_list_len, gender, \
        label = tf.decode_csv(
            value, defaults)
        item_id_list = tf.string_to_number(tf.string_split([item_id_list], ';').values, tf.int32)
        cate_id_list = tf.string_to_number(tf.string_split([cate_id_list], ';').values, tf.int32)
        tag_id_list = tf.string_to_number(tf.string_split([tag_id_list], ';').values, tf.int32)
        # label_cate_id_lsit = tf.string_to_number(tf.string_split([label_cate_id_lsit], ';').values, tf.int32)
        # label_tag_id_list = tf.string_to_number(tf.string_split([label_tag_id_list], ';').values, tf.int32)
        return user_id, item_id_list, item_id_list_len, cate_id_list, cat_id_list_len, tag_id_list, tag_id_list_len, \
               gender, label

    def read_item_train_data(self, file_queue):
        # reader = tf.TextLineReader()
        # key, value = reader.read(file_queue)
        # # defaults = [[0]] + [['0']] + [[0]] + [['0']] + [[0]]
        # defaults = [['0']] * 3
        # item_id_list, item_cate_id_list, item_tag_id_list, = tf.decode_csv(value, defaults)
        # item_id_list = tf.string_to_number(tf.string_split([item_id_list], ';').values, tf.int32)
        # item_cate_id_list = tf.string_to_number(tf.string_split([item_cate_id_list], ';').values, tf.int32)
        # item_tag_id_list = tf.string_to_number(tf.string_split([item_tag_id_list], ';').values, tf.int32)
        # return item_id_list, item_cate_id_list, item_tag_id_list
        with tf.gfile.Open(self.item_tower_file[0], 'r') as f:
            for line in f.readlines():
                lines = line.split(',')
                item_idx_list = [int(ele) for ele in lines[0].split(';')]
                cate_idx_list = [int(ele) for ele in lines[1].split(';')]
                tag_idx_list = [int(ele) for ele in lines[2].split(';')]
        return tf.constant(item_idx_list), tf.constant(cate_idx_list), tf.constant(tag_idx_list)

    def get_user_train_batch_data(self, tables, batch_size, num_epochs=None):
        file_queue = tf.train.string_input_producer(tables, num_epochs=num_epochs)
        user_id, item_id_list, item_id_list_len, cate_id_list, cat_id_list_len, tag_id_list, tag_id_list_len, gender, label = self.read_user_train_data(
            file_queue)

        # min_after_dequeue = 1000
        capacity = 10000
        user_batch, item_id_list_batch, item_id_list_len_batch, cate_id_list_batch, cate_id_list_len_batch, tag_id_list_batch, tag_id_list_len_batch, gender_batch, label_batch \
            = tf.train.batch(
            [user_id, item_id_list, item_id_list_len, cate_id_list, cat_id_list_len, tag_id_list, tag_id_list_len,
             gender, label],
            batch_size=batch_size, capacity=capacity, num_threads=1, allow_smaller_final_batch=True, dynamic_pad=True
            # min_after_dequeue=min_after_dequeue
        )

        return user_batch, item_id_list_batch, item_id_list_len_batch, cate_id_list_batch, cate_id_list_len_batch, tag_id_list_batch, tag_id_list_len_batch, gender_batch, label_batch

    # def get_item_train_batch_data(self, item_file, batch_size, num_epochs=None):
    #     file_queue = tf.train.string_input_producer(item_file, num_epochs=num_epochs)
    #     item_id, item_cate_id_list, item_tag_id_list = self.read_item_train_data(file_queue)
    #
    #     # min_after_dequeue = 1000
    #     capacity = 10000
    #     item_batch, item_cate_id_list_batch, item_tag_id_list_batch = tf.train.batch(
    #         [item_id, item_cate_id_list, item_tag_id_list],
    #         batch_size=batch_size, capacity=capacity,
    #         num_threads=1,
    #         allow_smaller_final_batch=True,
    #         dynamic_pad=True
    #         # min_after_dequeue=min_after_dequeue
    #     )
    #     return item_batch, item_cate_id_list_batch, item_tag_id_list_batch

    def read_user_test_data(self, file_queue):
        if self.local:
            reader = tf.TextLineReader(skip_header_lines=1)
        else:
            reader = tf.TableRecordReader()
        key, value = reader.read(file_queue)
        defaults = [[0]] + [['0']] + [[0]] + [['0']] + [[0]] + [['0']] + [[0]] * 2
        user_id, item_id_list, item_id_list_len, cate_id_list, cat_id_list_len, tag_id_list, tag_id_list_len, gender = tf.decode_csv(
            value, defaults)
        item_id_list = tf.string_to_number(tf.string_split([item_id_list], ';').values, tf.int32)
        cate_id_list = tf.string_to_number(tf.string_split([cate_id_list], ';').values, tf.int32)
        tag_id_list = tf.string_to_number(tf.string_split([tag_id_list], ';').values, tf.int32)
        return user_id, item_id_list, item_id_list_len, cate_id_list, cat_id_list_len, tag_id_list, tag_id_list_len, gender

    def create_test_pipeline(self, tables, batch_size, num_epochs=None):
        file_queue = tf.train.string_input_producer(tables, num_epochs=num_epochs)
        user_id, item_id_list, item_id_list_len, cate_id_list, cat_id_list_len, tag_id_list, tag_id_list_len, gender = self.read_user_test_data(
            file_queue)
        capacity = 10000
        user_batch, item_id_list_batch, item_id_list_len_batch, cate_id_list_batch, cate_id_list_len_batch, tag_id_list_batch, tag_id_list_len_batch, gender_batch = tf.train.batch(
            tensors=[user_id, item_id_list, item_id_list_len, cate_id_list, cat_id_list_len, tag_id_list,
                     tag_id_list_len, gender],
            batch_size=batch_size, capacity=capacity, num_threads=1, allow_smaller_final_batch=True, dynamic_pad=True
        )
        return user_batch, item_id_list_batch, item_id_list_len_batch, cate_id_list_batch, cate_id_list_len_batch, tag_id_list_batch, tag_id_list_len_batch, gender_batch

    # 计算每个batch平均embedding
    def get_seq_avg_embedding(self, item_embedding, item_id_list_batch, item_id_list_len_batch, embedding_size):
        embed_init = tf.nn.embedding_lookup(item_embedding, item_id_list_batch)
        embedding_mask = tf.sequence_mask(item_id_list_len_batch, tf.shape(item_id_list_batch)[1],
                                          dtype=tf.float32)
        embedding_mask = tf.expand_dims(embedding_mask, -1)
        embedding_mask = tf.tile(embedding_mask, [1, 1, embedding_size])
        embedding_mask_2 = embed_init * embedding_mask
        embedding_sum = tf.reduce_sum(embedding_mask_2, 1)

        seq_avg_embedding = tf.div(embedding_sum,
                                   tf.cast(tf.tile(tf.expand_dims(item_id_list_len_batch, 1),
                                                   [1, embedding_size]),
                                           tf.float32) + self.EPS)
        return seq_avg_embedding

    # def build_item_tower(self):
    #     if self.is_train:
    #         self.item_id_embed = tf.nn.embedding_lookup(self.item_embedding, self.item_batch)
    #         self.cate_id_embed = tf.nn.embedding_lookup(self.cate_embedding, self.item_cate_id_list_batch)
    #         self.tag_id_embed = tf.nn.embedding_lookup(self.tag_embedding, self.item_tag_id_list_batch)
    #         # self.user_embedding =
    #         # self.item_cate_embed = self.get_seq_avg_embedding(self.cate_embedding,
    #         #                                                   self.item_cate_id_list_batch,
    #         #                                                   self.item_cate_id_list_len_batch,
    #         #                                                   self.cate_embedding_size)
    #         # self.item_tag_embed = self.get_seq_avg_embedding(self.tag_embedding,
    #         #                                                  self.item_tag_id_list_batch,
    #         #                                                  self.item_tag_id_list_len_batch,
    #         #                                                  self.tag_embedding_size)
    #
    #         self.item_embed_merge = tf.concat(
    #             [self.item_id_embed, self.cate_id_embed, self.tag_id_embed],
    #             axis=-1)
    #         bn = tf.layers.batch_normalization(inputs=self.item_embed_merge, name='bn')
    #         dense_layer_1 = tf.layers.dense(bn, 512, activation=tf.nn.relu, name='first_dense',
    #                                         kernel_initializer=self.he_initializer)
    #         dense_layer_2 = tf.layers.dense(dense_layer_1, 256, activation=tf.nn.relu, name='second_dense',
    #                                         kernel_initializer=self.he_initializer)
    #         dense_layer_3 = tf.layers.dense(dense_layer_2, self.item_embedding_size, activation=tf.nn.relu,
    #                                         name='item_embed_final', kernel_initializer=self.he_initializer)
    #
    #         self.item_embed_final = dense_layer_3

    def build_user_model(self):
        # Item Embedding Layer
        # with tf.name_scope('item_embedding'):
        #     self.item_embedding = tf.get_variable(name='item_embedding',
        #                                           shape=[self.item_count, self.item_embedding_size],
        #                                           initializer=self.initializer)
        #     self.cate_embedding = tf.get_variable(name='cate_embedding',
        #                                           shape=[self.cate_count, self.cate_embedding_size],
        #                                           initializer=self.initializer)
        #     self.tag_embedding = tf.get_variable(name='tag_embedding', shape=[self.tag_count, self.tag_embedding_size],
        #                                          initializer=self.initializer)
        # item_embedding_b = tf.get_variable("item_b", [self.item_count], initializer=tf.constant_initializer(0.0))
        # if is_train:
        #     item_embed = tf.nn.embedding_lookup(item_embedding, label_batch)
        #     item_embed_b = tf.nn.embedding_lookup(item_embedding_b, label_batch)

        # User Embedding Layer
        with tf.name_scope("user_tower"):
            with tf.name_scope('user_embedding'):
                self.user_item_click_embed = self.get_seq_avg_embedding(self.item_embedding, self.item_id_list_batch,
                                                                        self.item_id_list_len_batch,
                                                                        self.item_embedding_size)
                self.user_cate_click_embed = self.get_seq_avg_embedding(self.cate_embedding, self.cate_id_list_batch,
                                                                        self.cate_id_list_len_batch,
                                                                        self.cate_embedding_size)
                self.user_tag_click_embed = self.get_seq_avg_embedding(self.tag_embedding, self.tag_id_list_batch,
                                                                       self.tag_id_list_len_batch,
                                                                       self.tag_embedding_size)

                # gender embedding
                gender_embedding = tf.get_variable(name='gender_embedding', shape=[self.GENDER_CNT, 1],
                                                   initializer=self.initializer)
                gender_embed = tf.nn.embedding_lookup(gender_embedding, self.gender_batch)

                #   concat embedding
                self.user_embed = tf.concat(
                    [self.user_item_click_embed, self.user_cate_click_embed, self.user_tag_click_embed, gender_embed],
                    axis=1)

            with tf.name_scope('layers'):
                bn = tf.layers.batch_normalization(inputs=self.user_embed, name='b1')
                layer_1 = tf.layers.dense(bn, 512, activation=tf.nn.relu, name='first',
                                          kernel_initializer=self.he_initializer)
                layer_2 = tf.layers.dense(layer_1, 256, activation=tf.nn.relu, name='second',
                                          kernel_initializer=self.he_initializer)
                layer_3 = tf.layers.dense(layer_2, self.item_embedding_size, activation=tf.nn.relu,
                                          name='user_final',
                                          kernel_initializer=self.he_initializer)

            self.user_embedding_final = layer_3

        with tf.name_scope("item_tower"):
            self.item_id_embed = tf.nn.embedding_lookup(self.item_embedding, self.item_batch)
            self.cate_id_embed = tf.nn.embedding_lookup(self.cate_embedding, self.item_cate_id_list_batch)
            self.tag_id_embed = tf.nn.embedding_lookup(self.tag_embedding, self.item_tag_id_list_batch)

            self.item_embed_merge = tf.concat(
                [self.item_id_embed, self.cate_id_embed, self.tag_id_embed],
                axis=-1)
            bn = tf.layers.batch_normalization(inputs=self.item_embed_merge, name='bn')
            dense_layer_1 = tf.layers.dense(bn, 128, activation=tf.nn.relu,
                                            name='first_dense', kernel_initializer=self.he_initializer)
            dense_layer_2 = tf.layers.dense(dense_layer_1, 64, activation=tf.nn.relu, name='second_dense',
                                            kernel_initializer=self.he_initializer)
            dense_layer_3 = tf.layers.dense(dense_layer_2, self.item_embedding_size, activation=tf.nn.relu,
                                            name='item_embed_final', kernel_initializer=self.he_initializer)

            self.item_embed_final = dense_layer_3

            # with tf.name_scope('sampled_weights'):
            #     self.sampled_softmax_weights = tf.Variable(
            #         tf.truncated_normal([self.item_count, self.item_merge_embedding_size], stddev=0.1))
            # with tf.name_scope('sampled_biases'):
            #     self.sampled_softmax_bias = tf.Variable(tf.zeros(self.item_count))

    def train_model(self):
        with tf.name_scope('train'):
            # nce loss
            item_batch = tf.expand_dims(self.label_batch, 1)
            # Construct the variables for the sampled softmax loss

            with tf.name_scope('loss'):
                losses = tf.reduce_mean(
                    tf.nn.sampled_softmax_loss(
                        weights=self.item_embed_final,
                        biases=self.item_embedding_b,
                        labels=item_batch,
                        inputs=self.user_embedding_final,
                        num_sampled=self.NUM_SAMPLED,
                        num_classes=self.item_count))

            cost = tf.reduce_sum(losses) / self.batch_size
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

            # gradients clip
            trainable_params = tf.trainable_variables()
            gradients = tf.gradients(cost, trainable_params)
            clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
            train_op = optimizer.apply_gradients(
                zip(clip_gradients, trainable_params), global_step=self.global_step)
        return train_op, cost

    def predict_topk_score(self):
        logits = tf.matmul(self.user_embedding_final, self.item_embed_final,
                           transpose_b=True) + self.item_embedding_b
        self.topk_score = tf.nn.top_k(logits, self.TOP_K_ITEM_CNT)[0]
        self.topk_idx = tf.nn.top_k(logits, self.TOP_K_ITEM_CNT)[1]
        self.user_topk_item = tf.reduce_join(tf.as_string(self.topk_idx), 1, separator=",")
        self.user_topk_score = tf.reduce_join(tf.as_string(self.topk_score), 1, separator=",")
        # return user_topk_item, user_topk_score

    def write_table(self):
        writer = tf.TableRecordWriter(self.output_path)
        write_to_table = writer.write(range(self.NUM_OF_TABLE_COLUMNS),
                                      [tf.as_string(self.user_batch),
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
