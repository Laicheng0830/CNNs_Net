"""
!/usr/bin/env python
-*- coding:utf-8 -*-
Author: eric.lai
Time: 2019/3/1 11:32
"""

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from CNNs_OPS import *

class InceptionV1_Mode():
    """ create InceptionV1 network use tensorflow
        InceptionV1 network structure:
        (conv 7x7 2 64)==>(maxpool 3x3 2)
        (conv 3x3 1 192)==>(maxpool 3x3 2)
        (Inception 64 96 128 16 32 32)==>(Inception 128 128 192 32 96 64)==>(maxpool 3x3 2)
        (Inception 192 96 208 16 48 64)==>(Inception 160 112 224 24 64 64)==>(Inception 128 128 256 24 64 64)==>(Inception 112 114 288 32 64 64)
        (Inception 256 160 320 32 128 128)==>(maxpool 3x3 2)
        (Inception 256 160 320 32 128 128)==>(Inception 384 192 384 48 128 128)
        (avgpool 7x7 1)==>drop(0.4)==>(fc 1000)==>(finalout nclass)
    """
    def model_bulid(self, height, width, channel,classes):
        """bulid"""
        x = tf.placeholder(dtype= tf.float32, shape = [None,height,width,channel])
        y = tf.placeholder(dtype= tf.float32 ,shape=[None,classes])

        # (conv 7x7 2 64)==>(maxpool 3x3 2)
        conv1_1 = tf.nn.relu(conv_layer(x,ksize=[3,3,channel,64],stride=[1,2,2,1],padding='SAME',name='conv1_1'))
        pool1_1 = pool_layer(conv1_1,ksize=[1,2,2,1],stride=[1,2,2,1],name='pool1_1')

        # (conv 3x3 1 192)==>(maxpool 3x3 2)
        conv2_1 = tf.nn.relu(conv_layer(pool1_1,ksize=[3,3,64,192],stride=[1,1,1,1],padding='SAME',name='conv2_1'))
        pool2_1 = pool_layer(conv2_1,ksize=[1,2,2,1],stride=[1,2,2,1],name='pool2_1')

        # (Inception 64 96 128 16 32 32)==>(Inception 128 128 192 32 96 64)==>(maxpool 3x3 2)
        inception3_1 = inception_model(pool2_1,64,96,128,16,32,32,name='inception3_1')
        inception3_2 = inception_model(inception3_1,128,128,192,32,96,64,name='inception3_2')
        pool3_1 = pool_layer(inception3_2,ksize=[1,2,2,1],stride=[1,2,2,1],name='pool3_1')

        # (Inception 192 96 208 16 48 64)==>(Inception 160 112 224 24 64 64)==>(Inception 128 128 256 24 64 64)==>(Inception 112 114 288 32 64 64)
        # (Inception 256 160 320 32 128 128)==>(maxpool 3x3 2)
        inception4_1 = inception_model(pool3_1,192,96,208,16,48,64,name='inception4_1')
        inception4_2 = inception_model(inception4_1,160,112,224,24,64,64,name='inception4_2')
        inception4_3 = inception_model(inception4_2,128,128,256,24,64,64,name='inception4_3')
        inception4_4 = inception_model(inception4_3,112,114,288,32,64,64,name='inception4_4')
        inception4_5 = inception_model(inception4_4,256,160,320,32,128,128,name='inception4_5')
        pool4_1 = pool_layer(inception4_5,ksize=[1,2,2,1],stride=[1,2,2,1],name='pool4_1')

        # (Inception 256 160 320 32 128 128) == > (Inception 384 192 384 48 128 128)
        inception5_1 = inception_model(pool4_1,256,160,320,32,128,128,name='inception5_1')
        inception5_2 = inception_model(inception5_1,384,192,384,48,128,128,name='inception5_2')

        # (avgpool 7x7 1) == > drop(0.4)
        pool6_1 = avg_pool_layer(inception5_2,ksize=[1,7,7,1],stride=[1,1,1,1],name='pool6_1')
        dp = dropout(pool6_1,keeppro=0.4)

        # Flatten
        ft = flatten(dp)

        # (fc 1000) == > (finalout nclass)
        fc1 = fc_layer(ft,fc_dims=1000,name="fc1")
        finaloutput = finlaout_layer(fc1,fc_dims=10,name="final")

        # cost
        loss = tf.losses.softmax_cross_entropy(y,finaloutput)

        # optimize
        LEARNING_RATE_BASE = 0.0001
        LEARNING_RATE_DECAY = 0.1
        LEARNING_RATE_STEP = 300
        gloabl_steps = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE
                                                   , gloabl_steps,
                                                   LEARNING_RATE_STEP,
                                                   LEARNING_RATE_DECAY,
                                                   staircase=True)
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            optimize = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        # prediction
        prediction_label = finaloutput
        correct_prediction = tf.equal(tf.argmax(prediction_label, 1), tf.argmax(y, 1))
        accurary = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))
        correct_times_in_batch = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.int32))
        return dict(
            x=x,
            y=y,
            optimize=optimize,
            correct_prediction=prediction_label,
            correct_times_in_batch=correct_times_in_batch,
            cost=loss,
            accurary=accurary
        )

    def init_sess(self):
        init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
        self.sess = tf.Session()
        self.sess.run(init)

    def train_network(self,graph,x_train,y_train):
        # Tensorfolw Adding more and more nodes to the previous graph results in a larger and larger memory footprint
        # reset graph
        tf.reset_default_graph()
        self.sess.run(graph['optimize'],feed_dict={graph['x']:x_train, graph['y']:y_train})

    def load_data(self):
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        g = self.model_bulid(28, 28, 1, 10)
        self.init_sess()
        for i in range(1000):
            batch_xs, batch_ys = mnist.train.next_batch(500)
            batch_xs = np.reshape(batch_xs,[-1,28,28,1])
            self.train_network(g, batch_xs, batch_ys)
            print("cost: ", self.sess.run(g['cost'], feed_dict={g['x']: batch_xs, g['y']: batch_ys}), "accurary: ",
                  self.sess.run(g['accurary'], feed_dict={g['x']: batch_xs, g['y']: batch_ys}))

    def load_CIFAR_data(self):
        import pickle
        g = self.model_bulid(224, 224, 3, 10)
        self.init_sess()
        """Enter data shape(None,224,224,3)"""

InceptionV1 = InceptionV1_Mode()
# InceptionV1.load_data()
InceptionV1.load_CIFAR_data()
