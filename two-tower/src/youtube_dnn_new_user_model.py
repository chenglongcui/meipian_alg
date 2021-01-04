# -*- coding: utf-8 -*-
import tensorflow as tf


class YoutubeDnnNewUser:
    def __init__(self, tables, is_train, embedding_size, batch_size, learning_rate, item_count, brand_count,
                 province_count, city_count, output_table, local, top_k_num, neg_sample_num):

        self.tables = tables
        self.is_train = is_train
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.learning_rate = tf.constant(learning_rate)
        self.table_path = "odps://dpdefault_68367/tables/"
        self.output_path = self.table_path + output_table
        self.local = local
        self.item_count = item_count
        self.brand_count = brand_count
        self.province_count = province_count
        self.city_count = city_count
        self.GENDER_CNT = 2
        self.AGE_CNT = 5
        self.CONSUME_LEVEL_CNT = 3
        self.MEIPIAN_AGE_CNT = 15
        self.NUM_SAMPLED = neg_sample_num
        self.NUM_OF_TABLE_COLUMNS = 3
        self.TOP_K_ITEM_CNT = top_k_num
        self.NUM_OF_EPOCHS = 1

        self.epoches = 1
        if local:
            self.PRINT_STEP = 1
        else:
            self.PRINT_STEP = 1000

        if self.is_train:
            self.user_batch, self.gender_batch, self.age_batch, self.consume_level_batch, self.client_type_batch, \
            self.brand_id_batch, self.province_batch, self.city_batch, self.label_batch = self.get_train_batch_data(
                self.batch_size, num_epochs=self.NUM_OF_EPOCHS)

            self.saved_model_inputs = {"user_id": self.user_batch,
                                       "gender": self.gender_batch,
                                       "age": self.age_batch,
                                       "consume_level": self.consume_level_batch,
                                       "client_type": self.client_type_batch,
                                       "brand_id": self.brand_id_batch,
                                       "province": self.province_batch,
                                       "city": self.city_batch}
        else:
            self.user_batch, self.gender_batch, self.age_batch, self.consume_level_batch, self.client_type_batch, \
            self.brand_id_batch, self.province_batch, self.city_batch = self.create_test_pipeline(tables, batch_size,
                                                                                                  num_epochs=self.NUM_OF_EPOCHS)

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.initializer = tf.random_uniform_initializer(-1, 1, seed=1234, dtype=tf.float32)
        self.he_initializer = tf.variance_scaling_initializer(mode="fan_avg")

        self.build_model()

    def read_train_data(self, file_queue):
        if self.local:
            reader = tf.TextLineReader(skip_header_lines=1)
        else:
            reader = tf.TableRecordReader()
        key, value = reader.read(file_queue)
        defaults = [['0']] * 9
        user_id, gender, age, consume_level, client_type, brand_id, province, city, label = tf.decode_csv(
            value, defaults)
        return user_id, gender, age, consume_level, client_type, brand_id, province, city, label

    def get_train_batch_data(self, batch_size, num_epochs=None):
        file_queue = tf.train.string_input_producer(self.tables, num_epochs=num_epochs)
        user_id, gender, age, consume_level, client_type, brand_id, province, city, label = self.read_train_data(
            file_queue)

        # min_after_dequeue = 1000
        capacity = 10000
        user_batch, gender_batch, age_batch, consume_level_batch, client_type_batch, brand_id_batch, province_batch, city_batch, label_batch \
            = tf.train.batch(
            [user_id, gender, age, consume_level, client_type, brand_id, province, city, label],
            batch_size=batch_size, capacity=capacity, num_threads=1, allow_smaller_final_batch=True, dynamic_pad=True
            # min_after_dequeue=min_after_dequeue
        )

        return user_batch, gender_batch, age_batch, consume_level_batch, client_type_batch, brand_id_batch \
            , province_batch, city_batch, label_batch

    def read_test_data(self, file_queue):
        if self.local:
            reader = tf.TextLineReader(skip_header_lines=1)
        else:
            reader = tf.TableRecordReader()
        key, value = reader.read(file_queue)
        defaults = [['0']] * 8
        user_id, gender, age, consume_level, client_type, brand_id, province, city = tf.decode_csv(value, defaults)

        return user_id, gender, age, consume_level, client_type, brand_id, province, city

    def create_test_pipeline(self, data_file, batch_size, num_epochs=None):
        file_queue = tf.train.string_input_producer(data_file, num_epochs=num_epochs)
        user_id, gender, age, consume_level, client_type, brand_id, province, city = self.read_test_data(file_queue)
        capacity = 10000
        user_batch, gender_batch, age_batch, consume_level_batch, client_type_batch, brand_id_batch \
            , province_batch, city_batch = tf.train.batch(
            tensors=[user_id, gender, age, consume_level, client_type, brand_id, province, city],
            batch_size=batch_size, capacity=capacity, num_threads=1, allow_smaller_final_batch=True, dynamic_pad=True
        )
        return user_batch, gender_batch, age_batch, consume_level_batch, client_type_batch, brand_id_batch \
            , province_batch, city_batch

    # 计算每个序列平均embedding
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
                                           tf.float32))
        return seq_avg_embedding

    def build_model(self):

        self.user_batch = tf.string_to_number(self.user_batch, out_type=tf.int32)
        self.gender_batch = tf.string_to_number(self.gender_batch, out_type=tf.int32)
        self.age_batch = tf.string_to_number(self.age_batch, out_type=tf.int32)
        self.consume_level_batch = tf.string_to_number(self.consume_level_batch, out_type=tf.int32)
        self.client_type_batch = tf.string_to_number(self.client_type_batch, out_type=tf.int32)
        self.brand_id_batch = tf.string_to_number(self.brand_id_batch, out_type=tf.int32)
        self.province_batch = tf.string_to_number(self.province_batch, out_type=tf.int32)
        self.city_batch = tf.string_to_number(self.city_batch, out_type=tf.int32)
        if self.is_train:
            self.label_batch = tf.string_to_number(self.label_batch, out_type=tf.int32)

        # Item Embedding Layer
        # with tf.name_scope('item_embedding'):
        #     item_embedding = tf.get_variable(name='item_embedding', shape=[self.item_count, self.embedding_size],
        #                                      initializer=self.initializer)
        #     item_embedding_b = tf.get_variable("item_b", [self.item_count], initializer=tf.constant_initializer(0.0))
        #     if self.is_train:
        #         item_embed = tf.nn.embedding_lookup(item_embedding, self.label_batch)
        #         item_embed_b = tf.nn.embedding_lookup(item_embedding_b, self.label_batch)

        # User Embedding Layer
        with tf.name_scope('user_embedding'):
            # user_history_click_embed_final = self.get_seq_avg_embedding(item_embedding, self.item_id_list_batch,
            #                                                             self.item_id_list_len_batch,
            #                                                             self.embedding_size)

            # gender embedding
            gender_embedding = tf.get_variable(name='gender_embedding', shape=[self.GENDER_CNT, 1],
                                               initializer=self.initializer)
            gender_embed = tf.nn.embedding_lookup(gender_embedding, self.gender_batch)

            # age embedding
            age_embedding = tf.get_variable(name='age_embedding', shape=[self.AGE_CNT, 3], initializer=self.initializer)
            age_embed = tf.nn.embedding_lookup(age_embedding, self.age_batch)

            # consume_lebel embedding
            consume_level_embedding = tf.get_variable(name='consume_level_embedding', shape=[self.CONSUME_LEVEL_CNT, 2],
                                                      initializer=self.initializer)
            consume_level_embed = tf.nn.embedding_lookup(consume_level_embedding, self.consume_level_batch)

            # client type embedding
            client_type_embedding = tf.get_variable(name='client_type_embedding', shape=[2, 1],
                                                    initializer=self.initializer)
            client_type_embed = tf.nn.embedding_lookup(client_type_embedding, self.client_type_batch)
            #
            # mobile brand embedding
            brand_embedding = tf.get_variable(name='brand_embedding', shape=[self.brand_count, 8],
                                              initializer=self.initializer)
            brand_embed = tf.nn.embedding_lookup(brand_embedding, self.brand_id_batch)

            # location info embedding
            province_embedding = tf.get_variable(name="province_embedding", shape=[self.province_count, 8],
                                                 initializer=self.initializer)
            province_embed = tf.nn.embedding_lookup(province_embedding, self.province_batch)

            city_embedding = tf.get_variable(name="city_embedding", shape=[self.city_count, 16],
                                             initializer=self.initializer)
            city_embed = tf.nn.embedding_lookup(city_embedding, self.city_batch)

            #   concat embedding
            user_embed = tf.concat(
                [gender_embed, age_embed, consume_level_embed, client_type_embed, brand_embed, province_embed,
                 city_embed],
                axis=1)

        with tf.name_scope('layers'):
            bn = tf.layers.batch_normalization(inputs=user_embed, name='b1')
            layer_1 = tf.layers.dense(bn, 512, activation=tf.nn.leaky_relu, name='first',
                                      kernel_initializer=self.he_initializer)
            layer_2 = tf.layers.dense(layer_1, 256, activation=tf.nn.leaky_relu, name='second',
                                      kernel_initializer=self.he_initializer)
            layer_3 = tf.layers.dense(layer_2, self.embedding_size, activation=tf.nn.leaky_relu, name='user_final',
                                      kernel_initializer=self.he_initializer)

        self.user_embedding = layer_3

        with tf.name_scope('sampled_weights'):
            self.sampled_softmax_weights = tf.Variable(
                tf.truncated_normal([self.item_count, self.embedding_size], stddev=0.1))
        with tf.name_scope('sampled_biases'):
            self.sampled_softmax_bias = tf.Variable(tf.zeros(self.item_count))

    def train_model(self):
        with tf.name_scope('train'):
            # nce loss
            item_batch = tf.expand_dims(self.label_batch, 1)
            # Construct the variables for the sampled softmax loss

            with tf.name_scope('loss'):
                losses = tf.reduce_mean(
                    tf.nn.sampled_softmax_loss(
                        weights=self.sampled_softmax_weights,
                        biases=self.sampled_softmax_bias,
                        labels=item_batch,
                        inputs=self.user_embedding,
                        num_sampled=self.NUM_SAMPLED,
                        num_classes=self.item_count))

            cost = tf.reduce_sum(losses) / self.batch_size
            logits = tf.matmul(self.user_embedding, self.sampled_softmax_weights,
                               transpose_b=True) + self.sampled_softmax_bias
            topk_score = tf.nn.top_k(logits, self.TOP_K_ITEM_CNT)[0]
            topk_idx = tf.nn.top_k(logits, self.TOP_K_ITEM_CNT)[1]
            self.saved_model_outputs = {"topk_tags": topk_idx, "topk_score": topk_score}
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

            # gradients clip
            trainable_params = tf.trainable_variables()
            gradients = tf.gradients(cost, trainable_params)
            clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
            train_op = optimizer.apply_gradients(
                zip(clip_gradients, trainable_params), global_step=self.global_step)
        return train_op, cost

    def predict_topk_score(self):
        logits = tf.matmul(self.user_embedding, self.sampled_softmax_weights,
                           transpose_b=True) + self.sampled_softmax_bias
        topk_score = tf.nn.top_k(logits, self.TOP_K_ITEM_CNT)[0]
        topk_idx = tf.nn.top_k(logits, self.TOP_K_ITEM_CNT)[1]
        user_topk_item = tf.reduce_join(tf.as_string(topk_idx), 1, separator=",")
        user_topk_score = tf.reduce_join(tf.as_string(topk_score), 1, separator=",")
        return user_topk_item, user_topk_score

    def write_table(self, user_topk_item, user_topk_score):
        writer = tf.TableRecordWriter(self.output_path)
        write_to_table = writer.write(range(self.NUM_OF_TABLE_COLUMNS),
                                      [tf.as_string(self.user_batch),
                                       user_topk_item,
                                       user_topk_score])
        return write_to_table

    def save_model(self, sess, path):
        print("save begin")
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

        tf.saved_model.simple_save(sess, dir,
                                   inputs=inputs,
                                   outputs=outputs)

    def load_saved_model(self, sess, dir):
        loaded = tf.saved_model.loader.load(sess, ["serve"], dir)
        graph = tf.get_default_graph()
