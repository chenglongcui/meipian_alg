# -*- coding: utf-8 -*-
import tensorflow as tf


class YoutubeDnnAd:
    def __init__(self, tables, is_train, embedding_size, batch_size, learning_rate, item_count, output_table, local,
                 ad_idx, ad_count):
        # self.sess = sess
        self.tables = tables
        self.is_train = is_train
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.learning_rate = tf.constant(learning_rate)
        self.table_path = "odps://dpdefault_68367/tables/"
        self.output_path = self.table_path + output_table
        self.local = local
        self.item_count = item_count
        self.GENDER_CNT = 2
        self.AGE_CNT = 5
        self.CONSUME_LEVEL_CNT = 3
        self.MEIPIAN_AGE_CNT = 15
        self.NUM_SAMPLED = 20
        self.NUM_OF_TABLE_COLUMNS = 3
        self.TOP_K_ITEM_CNT = ad_count
        # self.TOP_K_ITEM_CNT = 1
        self.NUM_OF_EPOCHS = 1
        self.ad_idx = ad_idx
        self.epoches = 1
        if local:
            self.PRINT_STEP = 1
        else:
            self.PRINT_STEP = 1000

        if self.is_train:
            self.user_batch, self.item_id_list_batch, self.item_id_list_len_batch, self.gender_batch, self.age_batch, \
            self.consume_level_batch, self.label_batch = self.get_train_batch_data(self.batch_size,
                                                                                   num_epochs=self.NUM_OF_EPOCHS)
        else:
            self.user_batch, self.item_id_list_batch, self.item_id_list_len_batch, self.gender_batch, self.age_batch, self.consume_level_batch = \
                self.create_test_pipeline(tables, batch_size, num_epochs=self.NUM_OF_EPOCHS)
        # self.checkpoint_dir = args.checkpoint_dir
        # if not os.path.isdir(self.checkpoint_dir):
        #     raise Exception("[!] Checkpoint Dir not found")
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.initializer = tf.random_uniform_initializer(-1, 1, seed=1234, dtype=tf.float32)
        self.he_initializer = tf.variance_scaling_initializer(mode="fan_avg")

        self.build_model()
        # self.sess.run(tf.global_variables_initializer())
        # self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

        # if self.is_training:
        #     return
        #
        # # use self.predict_state to hold hidden states during prediction.
        # self.predict_state = [np.zeros([self.batch_size, self.rnn_size], dtype=np.float32) for _ in xrange(self.layers)]
        # ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        # if ckpt and ckpt.model_checkpoint_path:
        #     self.saver.restore(sess, '{}/gru-model-{}'.format(self.checkpoint_dir, args.test_model)

    def read_train_data(self, file_queue):
        if self.local:
            reader = tf.TextLineReader(skip_header_lines=1)
        else:
            reader = tf.TableRecordReader()
        key, value = reader.read(file_queue)
        defaults = [[0]] + [['0']] + [[0]] * 5
        user_id, item_id_list, item_id_list_len, gender, age, consume_level, label = tf.decode_csv(value, defaults)
        item_id_list = tf.string_to_number(tf.string_split([item_id_list], ';').values, tf.int32)
        return user_id, item_id_list, item_id_list_len, gender, age, consume_level, label

    def get_train_batch_data(self, batch_size, num_epochs=None):
        file_queue = tf.train.string_input_producer(self.tables, num_epochs=num_epochs)
        user_id, item_id_list, item_id_list_len, gender, age, consume_level, label = self.read_train_data(file_queue)

        # min_after_dequeue = 1000
        capacity = 10000
        user_batch, item_id_list_batch, item_id_list_len_batch, gender_batch, age_batch, consume_level_batch, label_batch \
            = tf.train.batch(
            [user_id, item_id_list, item_id_list_len, gender, age, consume_level, label],
            batch_size=batch_size, capacity=capacity, num_threads=1, allow_smaller_final_batch=True, dynamic_pad=True
            # min_after_dequeue=min_after_dequeue
        )

        return user_batch, item_id_list_batch, item_id_list_len_batch, gender_batch, age_batch, consume_level_batch, label_batch

    def read_test_data(self, file_queue):
        if self.local:
            reader = tf.TextLineReader(skip_header_lines=1)
        else:
            reader = tf.TableRecordReader()
        key, value = reader.read(file_queue)
        defaults = [[0]] + [['0']] + [[0]] * 4
        user_id, item_id_list, item_id_list_len, gender, age, consume_level = tf.decode_csv(value, defaults)

        item_id_list = tf.string_to_number(tf.string_split([item_id_list], ';').values, tf.int32)
        return user_id, item_id_list, item_id_list_len, gender, age, consume_level

    def create_test_pipeline(self, data_file, batch_size, num_epochs=None):
        file_queue = tf.train.string_input_producer(data_file, num_epochs=num_epochs)
        user_id, item_id_list, item_id_list_len, gender, age, consume_level = self.read_test_data(file_queue)
        capacity = 10000
        user_batch, item_id_list_batch, item_id_list_len_batch, gender_batch, age_batch, consume_level_batch = tf.train.batch(
            tensors=[user_id, item_id_list, item_id_list_len, gender, age, consume_level],
            batch_size=batch_size, capacity=capacity, num_threads=1, allow_smaller_final_batch=True, dynamic_pad=True
        )
        return user_batch, item_id_list_batch, item_id_list_len_batch, gender_batch, age_batch, consume_level_batch

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
                                           tf.float32))
        return seq_avg_embedding

    def build_model(self):
        # Item Embedding Layer
        with tf.name_scope('item_embedding'):
            item_embedding = tf.get_variable(name='item_embedding', shape=[self.item_count, self.embedding_size],
                                             initializer=self.initializer)
            item_embedding_b = tf.get_variable("item_b", [self.item_count], initializer=tf.constant_initializer(0.0))
            # if is_train:
            #     item_embed = tf.nn.embedding_lookup(item_embedding, label_batch)
            #     item_embed_b = tf.nn.embedding_lookup(item_embedding_b, label_batch)

        # User Embedding Layer
        with tf.name_scope('user_embedding'):
            user_history_click_embed_final = self.get_seq_avg_embedding(item_embedding, self.item_id_list_batch,
                                                                        self.item_id_list_len_batch,
                                                                        self.embedding_size)

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

            #   concat embedding
            user_embed = tf.concat([user_history_click_embed_final, gender_embed, age_embed, consume_level_embed],
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
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

            # gradients clip
            trainable_params = tf.trainable_variables()
            gradients = tf.gradients(cost, trainable_params)
            clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
            train_op = optimizer.apply_gradients(
                zip(clip_gradients, trainable_params), global_step=self.global_step)
        return train_op, cost

    def predict_topk_score(self):
        self.ad_weight = tf.gather(self.sampled_softmax_weights, self.ad_idx)
        self.ad_bias = tf.gather(self.sampled_softmax_bias, self.ad_idx)
        logits = tf.matmul(self.user_embedding, self.ad_weight,
                           transpose_b=True) + self.ad_bias
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
        print("save begin")
        saver = tf.train.Saver()
        if not tf.gfile.Exists(path):
            tf.gfile.MkDir(path)
        saver.save(sess, save_path=path + 'model.ckpt')

    def restore_model(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path + 'model.ckpt')
