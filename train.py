# coding=utf-8
import os
import datetime
import random
import tensorflow as tf

from simpleLSTM import IDSNet

# TF log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# =========================================================================
flags = tf.flags
logging = tf.logging
# Data params
# ========================================================================
flags.DEFINE_string("Input_data", './data/train.csv','Data for training')
flags.DEFINE_string('Test_data', './data/test.csv', 'Data for test')
# Model params
# =========================================================
flags.DEFINF_integer("first_layer_node", 64, 'nodes in first layer')
flags.DEFINF_integer("second_layer_node", 50, 'nodes in first layer(if necessary)')
flags.DEFINE_integer("num_classes", 7, "Number of authors(default: 7")
flags.DEFINE_integer("flow_length", 10, "Number of flows in each sample")
flags.DEFINE_float("dropout_keep_prob", 1.0, "FC layer dropout keep probability (default: 1.0)")
# Training parameters
# =================================================
flags.DEFINE_float("learning_rate", 0.003, "Learning rate (default: 0.003)")
flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
# ===========================================================
FLAGS = flags.FLAGS

print("\nParameters:")
for attr, value in sorted(FLAGS.flag_values_dict().items()):
    print("{}={}".format(attr.upper(), value))
print("="*30)

def load_data():
"""
load data from train and test
return: tf.data
"""
def read_batch(input, batch_size):
    batch = []
    for element in input:
        batch.append(element)
        if len(batch) == batch_size:
            yield batch
            batch = []
    yield batch

def train(input_data):
    with tf.Graph().as_default():
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        with tf.Session(config=sess_config).as_default() as sess:
            ids = IDSNet(
                FLAGS.dropout_keep_prob,
                FLAGS.num_classes,
                FLAGS.flow_length
            )
            print('=' * 30)
            global_step = tf.Variable(0, name = 'global_step')
            learning_rate = tf.train.exponential_decay(
                learning_rate = FLAGS.learning_rate,
                global_step = global_step,
                decay_step = 100,
                decay_rate = 0.99,
                staircase = True,
                name = 'rl_decay'
            )
            optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(ids.loss)

            sess.run(tf.global_variables_initializer())

            def train_step(flow_train, y):
                feed_dict = {
                    ids.x_flow: flow_train,
                    ids.y: y
                }
                _, step, loss, accuracy = sess.run(
                    [train_op, global_step,  ids.loss, ids.acc],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

            for epoch in range(FLAGS.num_epochs):
                print('waiting for generating train data')
                train_data = read_batch(input, FLAGS.batch_size)
                train_data_len = len(train_data)
                num_batches = int(train_data_len/FLAGS.batch_size)
                print("{} mini batches per epoch".format(num_batches))
                for batch in range(num_batches):
                    train_data_feed = train_data.__next__()
                    train_data_x = train_data_feed[:,0:-1,:]
                    train_data_y = train_data_feed[:,-1,:]
                    train_step(train_data_x, train_data_y)


