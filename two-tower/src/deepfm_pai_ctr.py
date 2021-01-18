# -*- coding: utf-8 -*-

# import pai
import os
import json
import tensorflow as tf
from datetime import datetime
from tensorflow.contrib import layers
from tensorflow import feature_column
from tensorflow.python.training import queue_runner_impl

os.environ['VAR_PARTITION_THRESHOLD'] = '262144'
tf.logging.set_verbosity(tf.logging.DEBUG)
# tf.get_logger().setLevel('INFO')
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("model", 'deepfm', "model {'wdl', 'deepfm'}")
tf.flags.DEFINE_float("learning_rate", 0.0001, "learning rate")
tf.flags.DEFINE_string("optimizer", 'Adam', "optimizer type {Adam, Adagrad, GD, Momentum}")
tf.flags.DEFINE_integer("num_epochs", None, "Number of epochs")
tf.flags.DEFINE_integer("embedding_size", 16, "Embedding size")

tf.flags.DEFINE_integer("train_batch_size", 1024, "Number of batch size")
tf.flags.DEFINE_integer("val_batch_size", 1024, "Number of batch size")

tf.flags.DEFINE_integer("max_train_step", 10000, "max train step")
tf.flags.DEFINE_integer("save_summary_steps", 1000, "save summary step")
tf.flags.DEFINE_integer("save_checkpoint_and_eval_step", 1000, "save checkpoint and eval step")
tf.flags.DEFINE_integer("every_n_steps", 100, "every n step")
tf.flags.DEFINE_string("task_type", 'train', "task type {train, predict, savemodel}")

tf.flags.DEFINE_string("tables", "", "")
tf.flags.DEFINE_string("outputs", '', "")
tf.flags.DEFINE_string("checkpoint_dir", '', "model checkpoint dir")
tf.flags.DEFINE_string("output_dir", '', "model checkpoint dir")

tf.flags.DEFINE_string("job_name", "", "job name")
tf.flags.DEFINE_integer("task_index", None, "Worker or server index")
tf.flags.DEFINE_string("ps_hosts", "", "ps hosts")
tf.flags.DEFINE_string("worker_hosts", "", "worker hosts")
tf.flags.DEFINE_integer("inter_op_parallelism_threads", 12, "inter_op_parallelism_threads")
tf.flags.DEFINE_integer("intra_op_parallelism_threads", 12, "intra_op_parallelism_threads")


class DeepFM(object):
    def __init__(self):
        self.model = FLAGS.model

        self.lr = FLAGS.learning_rate
        self.optimizer = FLAGS.optimizer
        self.num_epochs = FLAGS.num_epochs
        self.embedding_dim = FLAGS.embedding_size

        self.train_batch_size = FLAGS.train_batch_size
        self.val_batch_size = FLAGS.val_batch_size

        self.output_table = FLAGS.outputs
        table_list = FLAGS.tables.split(',')
        self.train_table = table_list[0]
        self.val_table = table_list[1]
        print("train_table: " + str(self.train_table))
        print("val_table: " + str(self.val_table))

        self.checkpoint_path = FLAGS.checkpoint_dir
        self.output_dir = FLAGS.output_dir
        self.save_checkpoint_and_eval_step = FLAGS.save_checkpoint_and_eval_step
        self.max_train_step = FLAGS.max_train_step
        self.save_summary_steps = FLAGS.save_summary_steps
        self.every_n_steps = FLAGS.every_n_steps

        self.inter_op_parallelism_threads = FLAGS.inter_op_parallelism_threads
        self.intra_op_parallelism_threads = FLAGS.intra_op_parallelism_threads

        self.init_config()
        self.init_variable()
        self.init_distribute()
        self.get_fea_columns()

    # config
    def init_config(self):

        self.feas_name = ['hour', 'c1', 'banner_pos',
                          'site_id', 'site_domain', 'site_category',
                          'app_id', 'app_domain', 'app_category',
                          'device_id', 'device_ip', 'device_model',
                          'device_type', 'device_conn_type',
                          'c14', 'c15', 'c16', 'c17',
                          'c18', 'c19', 'c20', 'c21']

        self.fields_config_dict = {}
        self.fields_config_dict['hour'] = {'field_name': 'field1', 'embedding_dim': self.embedding_dim,
                                           'hash_bucket': 50, 'default_value': '0'}
        self.fields_config_dict['c1'] = {'field_name': 'field2', 'embedding_dim': self.embedding_dim, 'hash_bucket': 10,
                                         'default_value': '0'}
        self.fields_config_dict['banner_pos'] = {'field_name': 'field3', 'embedding_dim': self.embedding_dim,
                                                 'hash_bucket': 10, 'default_value': '0'}
        self.fields_config_dict['site_id'] = {'field_name': 'field4', 'embedding_dim': self.embedding_dim,
                                              'hash_bucket': 10000, 'default_value': '0'}
        self.fields_config_dict['site_domain'] = {'field_name': 'field5', 'embedding_dim': self.embedding_dim,
                                                  'hash_bucket': 10000, 'default_value': '0'}
        self.fields_config_dict['site_category'] = {'field_name': 'field6', 'embedding_dim': self.embedding_dim,
                                                    'hash_bucket': 100, 'default_value': '0'}
        self.fields_config_dict['app_id'] = {'field_name': 'field7', 'embedding_dim': self.embedding_dim,
                                             'hash_bucket': 10000, 'default_value': '0'}
        self.fields_config_dict['app_domain'] = {'field_name': 'field8', 'embedding_dim': self.embedding_dim,
                                                 'hash_bucket': 1000, 'default_value': '0'}
        self.fields_config_dict['app_category'] = {'field_name': 'field9', 'embedding_dim': self.embedding_dim,
                                                   'hash_bucket': 100, 'default_value': '0'}
        self.fields_config_dict['device_id'] = {'field_name': 'field10', 'embedding_dim': self.embedding_dim,
                                                'hash_bucket': 100000, 'default_value': '0'}
        self.fields_config_dict['device_ip'] = {'field_name': 'field11', 'embedding_dim': self.embedding_dim,
                                                'hash_bucket': 100000, 'default_value': '0'}
        self.fields_config_dict['device_model'] = {'field_name': 'field12', 'embedding_dim': self.embedding_dim,
                                                   'hash_bucket': 10000, 'default_value': '0'}
        self.fields_config_dict['device_type'] = {'field_name': 'field13', 'embedding_dim': self.embedding_dim,
                                                  'hash_bucket': 10, 'default_value': '0'}
        self.fields_config_dict['device_conn_type'] = {'field_name': 'field14', 'embedding_dim': self.embedding_dim,
                                                       'hash_bucket': 10, 'default_value': '0'}
        self.fields_config_dict['c14'] = {'field_name': 'field15', 'embedding_dim': self.embedding_dim,
                                          'hash_bucket': 500, 'default_value': '0'}
        self.fields_config_dict['c15'] = {'field_name': 'field16', 'embedding_dim': self.embedding_dim,
                                          'hash_bucket': 500, 'default_value': '0'}
        self.fields_config_dict['c16'] = {'field_name': 'field17', 'embedding_dim': self.embedding_dim,
                                          'hash_bucket': 500, 'default_value': '0'}
        self.fields_config_dict['c17'] = {'field_name': 'field18', 'embedding_dim': self.embedding_dim,
                                          'hash_bucket': 500, 'default_value': '0'}
        self.fields_config_dict['c18'] = {'field_name': 'field19', 'embedding_dim': self.embedding_dim,
                                          'hash_bucket': 500, 'default_value': '0'}
        self.fields_config_dict['c19'] = {'field_name': 'field20', 'embedding_dim': self.embedding_dim,
                                          'hash_bucket': 500, 'default_value': '0'}
        self.fields_config_dict['c20'] = {'field_name': 'field21', 'embedding_dim': self.embedding_dim,
                                          'hash_bucket': 500, 'default_value': '0'}
        self.fields_config_dict['c21'] = {'field_name': 'field22', 'embedding_dim': self.embedding_dim,
                                          'hash_bucket': 500, 'default_value': '0'}

        self.category_feas_num = len(self.fields_config_dict)
        print('category_feas_num: ' + str(self.category_feas_num))

        self.feas_type = []
        self.record_defaults = [[0]]
        for fea in self.feas_name:
            default = self.fields_config_dict[fea]['default_value']
            self.record_defaults.append([default])
            field_type = None
            if type(default) == type(0):
                field_type = tf.int32
            elif type(default) == type(''):
                field_type = tf.string
            self.feas_type.append(field_type)

    def init_variable(self):
        dnn_hidden_1 = 128
        dnn_hidden_2 = 32
        dnn_hidden_3 = 16
        dnn_output_dim = 1
        self.dnn_dims = [dnn_hidden_1, dnn_hidden_2, dnn_hidden_3, dnn_output_dim]

    # config for distribution
    def init_distribute(self):
        self.task_index = FLAGS.task_index
        ps_hosts = FLAGS.ps_hosts.split(",")
        worker_hosts = FLAGS.worker_hosts.split(",")
        self.ps_num = len(ps_hosts)

        if FLAGS.task_type == 'predict':
            self.worker_num = len(worker_hosts)

        else:
            print('old task_index: ' + str(FLAGS.task_index))
            if FLAGS.task_index > 1:
                self.task_index = FLAGS.task_index - 1
            print('new task_index: ' + str(self.task_index))

            self.worker_num = len(worker_hosts) - 1
            if len(worker_hosts):
                cluster = {"chief": [worker_hosts[0]], "ps": ps_hosts, "worker": worker_hosts[2:]}
                if FLAGS.job_name == "ps":
                    os.environ['TF_CONFIG'] = json.dumps(
                        {'cluster': cluster, 'task': {'type': FLAGS.job_name, 'index': FLAGS.task_index}})
                elif FLAGS.job_name == "worker":
                    if FLAGS.task_index == 0:
                        os.environ['TF_CONFIG'] = json.dumps(
                            {'cluster': cluster, 'task': {'type': "chief", 'index': 0}})
                    elif FLAGS.task_index == 1:
                        os.environ['TF_CONFIG'] = json.dumps(
                            {'cluster': cluster, 'task': {'type': "evaluator", 'index': 0}})
                    else:
                        os.environ['TF_CONFIG'] = json.dumps(
                            {'cluster': cluster, 'task': {'type': FLAGS.job_name, 'index': FLAGS.task_index - 2}})
            if 'TF_CONFIG' in os.environ:
                print(os.environ['TF_CONFIG'])
            os.environ['TASK_INDEX'] = str(FLAGS.task_index)
            os.environ['JOB_NAME'] = str(FLAGS.job_name)

            self.is_chief = False
            if self.task_index == 0 and FLAGS.job_name == "worker":
                print("This is chief")
                self.is_chief = True

            self.hook_sync_replicas = None
            self.sync_init_op = None

    def hash_embedding(self, hash_bucket, embedding_dim, name):
        cate_feature = feature_column.categorical_column_with_hash_bucket(name,
                                                                          hash_bucket,
                                                                          dtype=tf.string)
        emb_col = feature_column.embedding_column(
            cate_feature,
            dimension=embedding_dim,
            combiner='mean'
        )
        ind_col = feature_column.indicator_column(cate_feature)
        return emb_col, ind_col

    def get_fea_columns(self):
        self.feature_columns_dict = {
            'wide_feas': [],
            'fm_and_dnn_feas': [],
        }

        for field_idx in range(len(self.feas_name)):
            fea_name = self.feas_name[field_idx]
            config = self.fields_config_dict[fea_name]

            embeded_fea, ind_fea = self.hash_embedding(config['hash_bucket'], config['embedding_dim'], fea_name)
            self.feature_columns_dict['fm_and_dnn_feas'].append(embeded_fea)
            self.feature_columns_dict['wide_feas'].append(ind_fea)

    # print(self.feature_columns_dict)

    def _parse_batch_for_tabledataset(self, *args):
        label = tf.reshape(args[0], [-1])
        fields = [tf.reshape(v, [-1]) for v in args[1:]]
        return dict(zip(self.feas_name, fields)), label

    def train_input_fn_from_odps(self, data_path, epoch=10, batch_size=1024, slice_id=0, slice_count=1):
        with tf.device('/cpu:0'):
            dataset = tf.data.TableRecordDataset([data_path], record_defaults=self.record_defaults,
                                                 slice_count=slice_count, slice_id=slice_id)
            dataset = dataset.batch(batch_size).repeat(epoch)
            dataset = dataset.map(self._parse_batch_for_tabledataset, num_parallel_calls=8).prefetch(100)
            return dataset

    def val_input_fn_from_odps(self, data_path, epoch=1, batch_size=1024, slice_id=0, slice_count=1):
        with tf.device('/cpu:0'):
            dataset = tf.data.TableRecordDataset([data_path], record_defaults=self.record_defaults,
                                                 slice_count=slice_count, slice_id=slice_id)
            dataset = dataset.batch(batch_size).repeat(epoch)
            dataset = dataset.map(self._parse_batch_for_tabledataset, num_parallel_calls=8).prefetch(100)
            return dataset

    def serving_input_receiver_fn(self):
        feature_map = {}

        for fea_name, fea_type in zip(self.feas_name, self.feas_type):
            tensor = tf.placeholder(
                fea_type,
                shape=[None],
                name=fea_name)
            feature_map[fea_name] = tensor

        inputs = dict(feature_map)
        return tf.estimator.export.ServingInputReceiver(feature_map, inputs)

    # linear
    def linear(self, net):
        # net = tf.concat([cate_wide_feas], axis=1, name='linear_input')
        net = tf.layers.dense(net, units=1, activation=tf.nn.tanh, name='wide')
        linear_v = tf.reshape(net, [-1])
        return linear_v

    # fm
    def fm(self, net):
        # net = tf.concat([category_feas], axis=1, name='fm_input')
        fea_count = int(int(net.shape[-1]) / self.embedding_dim)
        # embeddings = tf.concat(category_feas, axis=1, name='concat_fm_input')   # None, n*k
        embeddings = tf.reshape(net, [-1, fea_count, self.embedding_dim])  # None, n, k
        # print(embeddings.shape)
        sum_square = tf.square(tf.reduce_sum(embeddings, 1))
        # print(sum_square.shape)
        square_sum = tf.reduce_sum(tf.square(embeddings), 1)
        # print(square_sum.shape)
        # y_v = 0.5 * tf.reduce_sum(tf.subtract(sum_square, square_sum), 1)  # None,
        y_v = 0.5 * tf.subtract(sum_square, square_sum)  # None, k
        # print(y_v.shape)
        y_v = tf.reduce_sum(y_v, 1)
        return y_v

    # dnn
    def dnn(self, net):
        net = tf.layers.batch_normalization(inputs=net, name='concat_bn', reuse=tf.AUTO_REUSE)
        for idx, units in enumerate(self.dnn_dims):
            net = tf.layers.dense(net, units=units, activation=tf.nn.relu, name='dnn_' + str(idx))
        dnn_v = tf.reshape(net, [-1])
        return dnn_v

    def model_fn(self, features, labels, mode, params):
        # forward
        wide_fea_cols = params['feature_columns']['wide_feas']
        fm_and_dnn_fea_cols = params['feature_columns']['fm_and_dnn_feas']

        wide_net = tf.feature_column.input_layer(features, wide_fea_cols)
        fm_and_dnn_net = tf.feature_column.input_layer(features, fm_and_dnn_fea_cols)

        y_pred = tf.Variable(0.0, name='y_pred')
        y_pred += self.linear(wide_net)
        y_pred += self.dnn(fm_and_dnn_net)
        if self.model == 'deepfm':
            y_pred += self.fm(fm_and_dnn_net)
        y_pred = tf.sigmoid(y_pred)

        # predict
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'probabilities': y_pred,
                'logits': y_pred
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        loss = tf.losses.log_loss(labels=labels, predictions=y_pred)

        # eval
        auc_ = tf.metrics.auc(labels=labels, predictions=y_pred, name='auc_op')
        metrics = {'auc': auc_}
        tf.summary.scalar('auc', auc_[1])
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

        # train
        if params['optimizer'] == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=params['lr'], beta1=0.9, beta2=0.999, epsilon=1e-8)
        elif params['optimizer'] == 'Adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate=params['lr'], initial_accumulator_value=1e-8)
        elif params['optimizer'] == 'Momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate=params['lr'], momentum=0.95)
        elif params['optimizer'] == 'ftrl':
            optimizer = tf.train.FtrlOptimizer(params['lr'])
        else:
            optimizer = tf.train.GradientDescentOptimizer(params['lr'])

        # SyncReplicasOptimizer for distribution
        optimizer = tf.train.SyncReplicasOptimizer(optimizer,
                                                   replicas_to_aggregate=self.worker_num,
                                                   total_num_replicas=self.worker_num,
                                                   use_locking=False)
        print("hook_sync_replicas is set")
        self.hook_sync_replicas = optimizer.make_session_run_hook(is_chief=self.is_chief, num_tokens=0)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        chief_queue_runner = optimizer.get_chief_queue_runner()
        queue_runner_impl.add_queue_runner(chief_queue_runner)
        self.sync_init_op = optimizer.get_init_tokens_op()
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[self.hook_sync_replicas])

    def run(self):
        session_config = tf.ConfigProto()
        session_config.intra_op_parallelism_threads = self.intra_op_parallelism_threads
        session_config.inter_op_parallelism_threads = self.inter_op_parallelism_threads
        session_config.allow_soft_placement = True
        session_config.gpu_options.allow_growth = True

        classifier = tf.estimator.Estimator(
            model_fn=self.model_fn,
            params={
                'feature_columns': self.feature_columns_dict,
                'lr': self.lr,
                'optimizer': self.optimizer,
            },
            config=tf.estimator.RunConfig(
                session_config=session_config,
                model_dir=self.checkpoint_path,
                tf_random_seed=2020,
                save_summary_steps=self.save_summary_steps,
                save_checkpoints_steps=self.save_checkpoint_and_eval_step,
                keep_checkpoint_max=1000)
        )

        if FLAGS.task_type == 'train':
            print("......................Start training......................")
            hooks = []
            if self.is_chief:
                # hook_profiler = tf.train.ProfilerHook(save_steps=self.every_n_steps, output_dir=self.output_dir)
                hook_counter = tf.train.StepCounterHook(output_dir=self.output_dir, every_n_steps=self.every_n_steps)
                # hooks.append(hook_profiler)
                hooks.append(hook_counter)

            train_spec = tf.estimator.TrainSpec(
                input_fn=lambda: self.train_input_fn_from_odps(
                    self.train_table,
                    epoch=self.num_epochs,
                    batch_size=self.train_batch_size,
                    slice_id=self.task_index,
                    slice_count=self.worker_num),
                max_steps=self.max_train_step,
                hooks=hooks)

            eval_spec = tf.estimator.EvalSpec(
                input_fn=lambda: self.val_input_fn_from_odps(
                    self.val_table,
                    batch_size=self.val_batch_size,
                    slice_id=0,
                    slice_count=1),
                steps=None,
                start_delay_secs=60,
                throttle_secs=60)

            tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

        elif FLAGS.task_type == 'predict':
            print("......................Start predict......................")
            predictions = classifier.predict(
                input_fn=lambda: self.val_input_fn_from_odps(
                    self.val_table,
                    epoch=1,
                    batch_size=self.val_batch_size,
                    slice_id=self.task_index,
                    slice_count=self.worker_num),
                predict_keys=None,
                checkpoint_path=None,
                yield_single_examples=False
            )
            # predict_result = next(predictions)
            # predict_result = list(zip(predict_result['probabilities'],
            # 	predict_result['logits']))
            # print(predict_result)

            idx = 0
            self.cur_time = datetime.now()
            start_time = self.cur_time
            try:
                writer = tf.python_io.TableWriter(self.output_table, slice_id=self.task_index)
                while True:
                    predict_result = next(predictions)
                    if not predict_result \
                            or 'probabilities' not in predict_result \
                            or 'logits' not in predict_result:
                        break
                    predict_result = list(zip(predict_result['probabilities'],
                                              predict_result['logits']))

                    writer.write(predict_result, indices=range(2))
                    idx += len(predict_result)
                    if (idx % (len(predict_result) * 100) == 0):
                        span = datetime.now() - self.cur_time
                        print("idx: {0}, time consume: {1}".format(idx, span.seconds))
                        self.cur_time = datetime.now()
            except StopIteration:
                print(
                    "predict finish..., idx: {0}, time consume: {1}".format(idx, (datetime.now() - start_time).seconds))

        elif FLAGS.task_type == 'savemodel':
            if self.is_chief:
                print("......................Start savemodel......................")
                classifier.export_savedmodel(self.output_dir, self.serving_input_receiver_fn, strip_default_attrs=True)


def main(_):
    model = DeepFM()
    model.run()


if __name__ == "__main__":
    print(datetime.now().strftime('%Y-%m-%d , %H:%M:%S'))
    print("task_type: " + str(FLAGS.task_type))
    print("learning_rate: " + str(FLAGS.learning_rate))
    print("optimizer: " + str(FLAGS.optimizer))
    print("num_epochs: " + str(FLAGS.num_epochs))
    print("embedding_size: " + str(FLAGS.embedding_size))
    print("train_batch_size: " + str(FLAGS.train_batch_size))
    print("val_batch_size: " + str(FLAGS.val_batch_size))
    print("max_train_step: " + str(FLAGS.max_train_step))
    print("save_checkpoint_and_eval_step: " + str(FLAGS.save_checkpoint_and_eval_step))
    print("output: " + str(FLAGS.outputs))
    print("checkpoint_dir: " + str(FLAGS.checkpoint_dir))

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
