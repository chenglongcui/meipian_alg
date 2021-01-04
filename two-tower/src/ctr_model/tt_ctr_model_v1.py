# -*- coding: utf-8 -*-
import tensorflow as tf
import math


class TTFRCTRModelV1:
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
        self.ITEM_CNT = 100000
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

        # build dataset
        if self.mode == "train":
            self.dataset = self.build_user_train_data(self.train_file_dir)
        elif self.mode == "test":
            self.dataset = self.build_user_train_data(self.test_file_dir)
        self.train_iterator = self.dataset.make_one_shot_iterator()

        self.label, self.user_id, self.item_id, self.class_id, self.tag_ids, self.client_type, self.click_seq_50size, \
        self.gender = self.train_iterator.get_next()
        # 以下代码是导致性能下降的原因
        # self.click_seq_50size_array = tf.map_fn(fn=lambda x: tf.string_split([x], ',').values,
        #                                         elems=self.click_seq_50size)

        # self.click_seq_50size_array = tf.strings.split(self.click_seq_50size, sep=",")
        # self.click_seq_50size_len = tf.count_nonzero(self.click_seq_50size, 1)

        # 序列转数组格式(注意最后一个batch数据个数小于batch_size)
        self.click_seq_50size_array = tf.reshape(tf.strings.split(self.click_seq_50size, sep=",").values,
                                                 shape=[tf.shape(self.gender)[0], -1])
        # 获取真实点击序列长度（非0元素）
        self.click_seq_50size_len = tf.count_nonzero(self.click_seq_50size_array, 1)
        self.training_init_op = self.train_iterator.make_initializer(self.dataset)
        self.train_iterator = self.dataset.make_one_shot_iterator()

        # batch_size个数的0(正样本的index)，需要注意最后一个batch是不足s一个batch_size的

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

            # saved_model 格式的输入，用于eas在线预测
            if self.mode == "train":
                # tensor_info_user_id = tf.saved_model.utils.build_tensor_info(self.user_id)
                tensor_info_gender = tf.saved_model.utils.build_tensor_info(self.gender)
                tensor_info_client_type = tf.saved_model.utils.build_tensor_info(self.client_type)
                tensor_info_user_click_item_list = tf.saved_model.utils.build_tensor_info(self.click_seq_50size)
                tensor_info_item_id = tf.saved_model.utils.build_tensor_info(self.item_id)
                tensor_info_class_id = tf.saved_model.utils.build_tensor_info(self.class_id)
                tensor_info_tag_ids = tf.saved_model.utils.build_tensor_info(self.tag_ids)

                # build saved inputs
                self.saved_model_inputs = {
                    "click_seq_50size": tensor_info_user_click_item_list,
                    "gender": tensor_info_gender,
                    "client_type": tensor_info_client_type,
                    "item_id": tensor_info_item_id,
                    "class_id": tensor_info_class_id,
                    "tag_ids": tensor_info_tag_ids
                }
            self.build_categorial_features()
            self.build_sequence_features()
            self.build_model()

    def decode_train_line(self, line):
        defaults = [[0.0]] + [['0']] * 7
        label, user_id, item_id, class_id, tag_ids, client_type, click_seq_50size, gender \
            = tf.decode_csv(line, defaults, field_delim=";")

        # tag只取第一个
        # TODO：后续tag求和或者取平均值
        tag_ids = tf.string_split([tag_ids], ',').values[0]
        return label, user_id, item_id, class_id, tag_ids, client_type, click_seq_50size, gender

    def build_user_train_data(self, train_data_file):
        # padded_shape = ([], [], [], [], [], [], [None], [])
        dataset = tf.data.TextLineDataset(train_data_file)
        dataset = dataset.prefetch(buffer_size=self.batch_size * 100)
        dataset = dataset.map(self.decode_train_line).batch(self.batch_size).repeat(self.train_epoches)
        # dataset = dataset.map(self.decode_train_line).padded_batch(batch_size=self.batch_size,
        #                                                            padded_shapes=padded_shape).repeat(
        #     self.train_epoches)
        return dataset

    # 计算序列的embedding(平均或求和)
    # TODO: 后续可以使用序列建模方式抽取序列特征
    def get_seq_embedding(self, item_embedding, user_click_item_list, item_id_list_len_batch, embedding_size, method):
        # user_click_item_list_idx = self.item_table.lookup(self.user_click_item_list)
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
            seq_avg_embedding = tf.div(self.embedding_sum,
                                       tf.cast(tf.tile(tf.expand_dims(item_id_list_len_batch, 1),
                                                       [1, embedding_size]),
                                               tf.float32))
            seq_embedding_final = seq_avg_embedding

        return seq_embedding_final

    def build_sequence_features(self):
        self.click_seq_50size_idx = tf.string_to_hash_bucket_fast(self.click_seq_50size_array, self.ITEM_CNT)

        self.click_seq_50size_embed = self.get_seq_embedding(self.item_embedding,
                                                             self.click_seq_50size_idx,
                                                             self.click_seq_50size_len,
                                                             self.item_embedding_size,
                                                             "sum")

        # self.unclick_seq_50size_idx = tf.string_to_hash_bucket_fast(tf.as_string(self.unclick_seq_50size),
        #                                                             self.ITEM_CNT)
        #
        # self.unclick_seq_50size_embed = self.get_seq_embedding(self.item_embedding,
        #                                                        self.unclick_seq_50size_idx,
        #                                                        self.unclick_seq_50size_len,
        #                                                        self.item_embedding_size,
        #                                                        "mean")

    def build_categorial_features(self):
        self.client_type_one_hot = tf.one_hot(tf.string_to_hash_bucket_fast(self.client_type,
                                                                            self.CLIENT_TYPE_CNT),
                                              self.CLIENT_TYPE_CNT)

        self.gender_one_hot = tf.one_hot(tf.string_to_hash_bucket_fast(self.client_type,
                                                                       self.GENDER_CNT),
                                         self.GENDER_CNT)

    def build_model(self):

        # # User Embedding Layer
        with tf.name_scope("user_tower"):
            with tf.name_scope('user_embedding'):
                # concat embedding
                user_embed_concat = tf.concat(
                    [self.click_seq_50size_embed, self.gender_one_hot, self.client_type_one_hot], axis=-1)
            with tf.name_scope('layers'):
                user_layer_1 = tf.layers.dense(inputs=user_embed_concat,
                                               units=1024,
                                               activation=tf.nn.tanh,
                                               name='user_first',
                                               kernel_initializer=tf.initializers.glorot_normal(seed=1234),
                                               # use_bias=False
                                               )
                user_layer_2 = tf.layers.dense(inputs=user_layer_1,
                                               units=512,
                                               activation=tf.nn.tanh,
                                               name='user_second',
                                               kernel_initializer=tf.initializers.glorot_normal(seed=1234),
                                               # use_bias=False
                                               )
                user_layer_3 = tf.layers.dense(inputs=user_layer_2,
                                               units=self.item_embedding_size,
                                               activation=tf.nn.tanh,
                                               name='user_final',
                                               kernel_initializer=tf.initializers.glorot_normal(seed=1234),
                                               # use_bias=False
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
            item_idx = tf.string_to_hash_bucket_fast(self.item_id, self.ITEM_CNT)
            cate_idx = tf.string_to_hash_bucket_fast(self.class_id, self.CATE_CNT)
            tag_idx = tf.string_to_hash_bucket_fast(self.tag_ids, self.TAG_CNT)

            item_id_embed = tf.nn.embedding_lookup(self.item_embedding, item_idx)
            cate_id_embed = tf.nn.embedding_lookup(self.cate_embedding, cate_idx)
            tag_id_embed = tf.nn.embedding_lookup(self.tag_embedding, tag_idx)

            target_embed_concat = tf.concat([item_id_embed, cate_id_embed, tag_id_embed], axis=-1)
            with tf.name_scope('item_layers'):
                item_layer_1 = tf.layers.dense(inputs=target_embed_concat,
                                               units=self.item_embedding_size,
                                               activation=tf.nn.tanh,
                                               name='item_first',
                                               kernel_initializer=tf.initializers.glorot_normal(seed=1234),
                                               # use_bias=False
                                               )
                # item_layer_2 = tf.layers.dense(inputs=item_layer_1,
                #                                units=512,
                #                                activation=tf.nn.tanh,
                #                                name='item_second',
                #                                kernel_initializer=tf.initializers.glorot_normal(seed=1234),
                #                                # use_bias=False
                #                                )
                # item_layer_3 = tf.layers.dense(inputs=item_layer_2,
                #                                units=self.item_embedding_size,
                #                                activation=tf.nn.tanh,
                #                                name='item_third',
                #                                kernel_initializer=tf.initializers.glorot_normal(seed=1234),
                #                                # use_bias=False
                #                                )

        # item参数写入summary
        # item_first_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'item_first')
        # tf.summary.histogram("item_layer_1_weights", item_first_vars[0])
        # # tf.summary.histogram("item_layer_1_biases", item_first_vars[1])
        # tf.summary.histogram("item_layer_1_output", item_layer_1)

        self.item_embeding_final = item_layer_1

        # ## EAS 上运行必须加
        # self.user_embedding_final_expand = tf.expand_dims(self.user_embedding_final, 1)
        # self.item_embeding_final = tf.transpose(self.item_embeding_final, perm=[0, 2, 1])
        # self.logits = tf.squeeze(tf.matmul(self.user_embedding_final_expand, self.item_embeding_final), axis=1)
        #

        # 计算logits
        self.logits = tf.nn.sigmoid(
            tf.reduce_sum(tf.multiply(self.user_embedding_final, self.item_embeding_final), axis=1))

        # # saved_model 输出
        ## 输出的分数使用sigmoid归一化
        self.logits_score = tf.nn.sigmoid(self.logits)
        tensor_info_logits = tf.saved_model.utils.build_tensor_info(self.logits_score)
        self.saved_model_outputs["logits"] = tensor_info_logits

    def train_model(self):

        sigmoid_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.logits))

        # l2_norm loss
        regulation_rate = 0.0001
        l2_norm = tf.add_n([
            tf.reduce_sum(tf.multiply(self.user_embedding_final, self.user_embedding_final)),
            tf.reduce_sum(tf.multiply(self.item_embeding_final, self.item_embeding_final)),
        ])
        l2_loss = regulation_rate * l2_norm
        loss_merge = l2_loss + sigmoid_loss

        # tf.summary.scalar('loss_merge', loss_merge)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        # train_op = optimizer.minimize(loss_merge, global_step=self.global_step)

        var_list = tf.trainable_variables()
        gradients = optimizer.compute_gradients(loss_merge, var_list)
        capped_gradients = [(tf.clip_by_value(grad, -500., 500.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients, global_step=self.global_step)

        return train_op, loss_merge, sigmoid_loss, self.learning_rate

    def evaluate(self):
        prediction = tf.nn.sigmoid(self.logits)
        auc_value, auc_op = tf.metrics.auc(labels=self.label, predictions=prediction)
        return auc_value, auc_op

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
