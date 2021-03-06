# coding=utf-8
# !/usr/bin/python3
import os
import datetime
import random
import tensorflow as tf
import csv
import sklearn as sk
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np

from simpleLSTM import IDSNet
from simpleGRU import IDSGRUNet
from hstateLSTM import HIDSNet

# TF log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# =========================================================================
flags = tf.flags
# logging = tf.logging
# Data params
# ========================================================================
flags.DEFINE_string("Input_data", './data/test5ss.csv', 'Data for training')
flags.DEFINE_string('Test_data', './data/test5ss.csv', 'Data for test')
flags.DEFINE_integer('Feature_num', 78, 'numbers of features in one flow (defult:79)')
# Model params
# =========================================================
flags.DEFINE_integer("LSTM_hidden_node", 512, 'nodes in lstm layer')
flags.DEFINE_integer("fcmiddle_node", 0, 'nodes in second layer(if necessary)')
flags.DEFINE_integer("num_classes", 8, "Number of authors(default: 8")
flags.DEFINE_integer("flow_length", 5, "Number of flows in each sample")
flags.DEFINE_float("dropout_keep_prob", 1.0, "FC layer dropout keep probability (default: 1.0)")
flags.DEFINE_integer("Attentions", 10 ,"attention on the chosen flow")
# Training parameters
# =================================================
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate (default: 0.003)")
flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
flags.DEFINE_integer("num_epochs", 40, "Number of training epochs (default: 200)")
flags.DEFINE_boolean("is_verbose", False, "Print loss (default: True)")
flags.DEFINE_integer("evaluate_every", 2000, "when to dev")
# ===========================================================
FLAGS = flags.FLAGS




def load_data():
    """
    load data from train and test

    list1 = ["0" for i in range(20)]
    list2 = ['1' for i in range(20)]
    list3 = ['0' for i in range(10)] + ['1' for i in range(10)]
    print("list1:{}\nlist2:{}\nlist3:{}\n".format(list1, list2, list3))
    list4, list5, list6 = [], [], []
    for i in range(10):
        list4.append(list1)
        list5.append(list2)
        list6.append(list3)
    list4.append('1')
    list5.append('2')
    list6.append('3')
    list7 = []
    for i in range(100):
        list7.append(list4)
        list7.append(list5)
        list7.append(list6)
    #   print(list7)
    list8 = list7
    return list7, list8
    """
    with open(FLAGS.Input_data, 'r') as train_data:
        reader = csv.reader(train_data)
        table = [row for row in reader]
#        feature = [row[:-1] for row in table]
        label = [row[-1] for row in table]
        print("train label over")
        train_table = []
        for i in range(len(table)):
            features = []
            for j in range(FLAGS.flow_length):
                features.append(table[i][j*FLAGS.Feature_num:(j+1)*FLAGS.Feature_num])
            features.append(label[i])
            train_table.append(features)
	table = []
	print('Training set over')
    
    with open(FLAGS.Test_data, 'r') as train_data:
        reader = csv.reader(train_data)
        table = [row for row in reader]
#        feature = [row[:-1] for row in table]
        label = [row[-1] for row in table]
        print("test label over")
        test_table = []
        for i in range(len(table)):
            features = []
            for j in range(FLAGS.flow_length):
                features.append(table[i][j*FLAGS.Feature_num:(j+1)*FLAGS.Feature_num])
            features.append(label[i])
            test_table.append(features)
        table = []
	print('test set over')
    
#    test_table = train_table
#    print(train_table[0])
#    print(len(test_table[0]))
    return train_table, test_table







def read_batch(input, batch_size):
    batch = []
    for element in input:
        batch.append(element)
        if len(batch) == batch_size:
            yield batch
            batch = []
    yield batch


def train(input_data_train, input_data_test):
    with tf.Graph().as_default():
        sess_config = tf.ConfigProto()
#	device_count = {"CPU": 30},
#	inter_op_parallelism_threads = 2,
#	intra_op_parallelism_threads = 2)
#        sess_config.gpu_options.allow_growth = True
        n = 3
        with tf.Session(config=sess_config).as_default() as sess:
            if n == 1:
                ids = IDSGRUNet(
                    FLAGS.dropout_keep_prob,
                    FLAGS.num_classes,
                    FLAGS.flow_length,
                    FLAGS.batch_size
                )
            if n == 2:
                ids = IDSNet(
		    FLAGS.LSTM_hidden_node,
                    FLAGS.dropout_keep_prob,
                    FLAGS.num_classes,
                    FLAGS.flow_length,
                    FLAGS.batch_size
                )
            if n == 3:
                ids = HIDSNet(
		    FLAGS.LSTM_hidden_node,
		    FLAGS.fcmiddle_node,
		    FLAGS.Attentions,
                    FLAGS.dropout_keep_prob,
                    FLAGS.num_classes,
                    FLAGS.flow_length,
                    FLAGS.batch_size
                )
            print('=' * 30)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            learning_rate = tf.train.exponential_decay(
                learning_rate=FLAGS.learning_rate,
                global_step=global_step,
                decay_steps=100,
                decay_rate=0.99,
                staircase=True,
                name="rl_decay"
            )
            optimizer = tf.train.AdamOptimizer(learning_rate)
            grads = optimizer.compute_gradients(ids.loss)
            train_op = optimizer.apply_gradients(grads, global_step=global_step)

            sess.run(tf.global_variables_initializer())

            def train_step(flow_train, y):
                feed_dict = {
                    ids.x_flow: flow_train,
                    ids.y: y
                }
                _, step, loss, accuracy = sess.run(
                    [train_op, global_step, ids.loss, ids.acc],
                    feed_dict)
                if FLAGS.is_verbose:
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

            def dev_step(flow_test, y):
                feed_dict = {
                    ids.x_flow: flow_test,
                    ids.y: y
                }
                step, loss, accuracy, pre_y = sess.run(
                    [global_step, ids.loss, ids.acc, ids.pre_y],
                    feed_dict)
                return loss, accuracy, pre_y

            for epoch in range(FLAGS.num_epochs):
                print('waiting for generating train data')
                train_data = read_batch(input_data_train, FLAGS.batch_size)
                #                print(len(input_data_train))
                #                print(input_data_train[1])
                train_data_len = len(input_data_train)
                num_batches = int(train_data_len / FLAGS.batch_size)
                print("{} mini batches per epoch".format(num_batches))
                for batch in range(num_batches):
                    train_data_feed = next(train_data)
                    #                   print(len(train_data_feed))
                    #                   print(train_data_feed[:][:-1])
                    train_data_x = []
                    train_data_y = []
                    for i in range(FLAGS.batch_size):
                        train_data_x.append(train_data_feed[i][:-1])
                        train_data_y.append(train_data_feed[i][-1])
                    train_step(train_data_x, train_data_y)
                    current_step = tf.train.global_step(sess, global_step)

                    if current_step % FLAGS.evaluate_every == 0:
                        print("\nEvaluation:")
                        dev_num_batches = int(len(list(input_data_test)) / FLAGS.batch_size)
                        print("awaiting for generating dev data...")
                        test_data = read_batch(input_data_test, FLAGS.batch_size)

                        dev_loss = []
                        dev_acc = []
                        label_y = []
                        pred_y = []
                        for dev_batch in range(dev_num_batches):
                            test_data_feed = next(test_data)
                            test_data_x = []
                            test_data_y = []
                            for i in range(FLAGS.batch_size):
                                test_data_x.append(test_data_feed[i][:-1])
                                test_data_y.append(test_data_feed[i][-1])
                                label_y.append(test_data_feed[i][-1])
                            loss1, acc1, pre_y = dev_step(test_data_x, test_data_y)
                            dev_loss.append(loss1)
                            dev_acc.append(acc1)
                            pred_y = pred_y + pre_y.tolist()[0]
 #                       print("Recall", sk.metrics.recall_score(label_y, pred_y))
  #                      print("f1_score", sk.metrics.f1_score(label_y, pred_y))
                        time_str = datetime.datetime.now().isoformat()
                        print("dev:{} step: {}, loss: {:g}, acc: {:g}".
                              format(time_str, current_step, sum(dev_loss) / len(dev_loss),
                                     sum(dev_acc) / len(dev_loss)))
			pred_y =  np.array(pred_y, dtype = 'int_')
			label_y = np.array(label_y, dtype = 'int_')
#			print(pred_y[:10])
#			print(label_y[:10])
			print(classification_report(label_y, pred_y))
			print(confusion_matrix(label_y, pred_y))


                        print("\n" + "=" * 30)


def main(_):
    print("Load data...\n")
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__dict__['__flags'].items()):
        print("{}={}".format(attr.upper(), value))
    print("=" * 30)
    input_train, input_test = load_data()
    print('Complete\n')
    print('Model Start...\n')
    train(input_train, input_test)


if __name__ == "__main__":
    tf.app.run()



