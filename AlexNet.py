"""
!/usr/bin/env python
-*- coding:utf-8 -*-
Author: eric.lai
Time: 2019/2/22 11:49
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from CNNs_OPS import *


class AlexNet_Mode():
    """ create AlexNet network use tensorflow
        AlexNet network structure:
        (conv 11x11 96)=>( lrn )==>(maxpool 3x3 stride 2)
        (conv 5x5 256 grops=2)=>( lrn )==>(maxpoool 3x3 stride 2)
        (conv 3x3 384)=>(conv 3x3 384 groups=2)==>(conv 3x3 256 groups=2)==>(maxpool 3x3 stride=2)
        (ft fc 4096)=>(droupout)=>(fc 4096)==>(droupout)==>(fc classes)
    """
    def model_bulid(self, height, width, channel,classes):
        x = tf.placeholder(dtype= tf.float32, shape = [None,height,width,channel])
        y = tf.placeholder(dtype= tf.float32 ,shape=[None,classes])

        # (conv 11x11 96)=>( lrn )==>(maxpool 3x3 stride 2)
        conv1_1 = tf.nn.relu(conv_layer(x,ksize=[11,11,channel,96],stride=[1,1,1,1],name='conv1_1',padding='SAME'))
        lrn_1 = lrn(conv1_1,R=2,alpha=2e-5,beta=0.75)
        pool1_1 = pool_layer(lrn_1,ksize=[1,3,3,1],stride=[1,2,2,1],name='pool1_1') # 232x64x96

        # (conv 5x5 256 grops=2)=>( lrn )==>(maxpoool 3x3 stride 2)
        conv2_1 = tf.nn.relu(conv_layer(pool1_1,ksize=[5,5,96,256],stride=[1,1,1,1],name='conv2_1',padding='SAME'))
        lrn_2 = lrn(conv2_1,R=2,alpha=2e-5,beta=0.75)
        pool2_1 = pool_layer(lrn_2,ksize=[1,3,3,1],stride=[1,2,2,1],name='pool2_1') # 116x32x256

        # (conv 3x3 384)=>(conv 3x3 384 groups=2)==>(conv 3x3 256 groups=2)==>(maxpool 3x3 stride=2)
        conv3_1 = tf.nn.relu(conv_layer(pool2_1,ksize=[3,3,256,384],stride=[1,1,1,1],padding="SAME",name="conv3_1"))
        conv3_2 = tf.nn.relu(conv_layer(conv3_1,ksize=[3,3,384,384],stride=[1,1,1,1],padding="SAME",name="conv3_2"))
        conv3_3 = tf.nn.relu(conv_layer(conv3_2,ksize=[3,3,384,256],stride=[1,1,1,1],padding='SAME',name='conv3_3'))
        pool3_1 = pool_layer(conv3_3,ksize=[1,3,3,1],stride=[1,2,2,1],name="pool3_1") # Nx58x16x256

        # Flatten
        ft = flatten(pool3_1)

        # (ft fc 4096)=>(droupout)=>(fc 4096)==>(droupout)==>(fc classes)
        fc1 = fc_layer(ft,fc_dims=4096,name="fc1")
        dt1 = dropout(fc1)
        fc2 = fc_layer(dt1,fc_dims=4096,name="fc2")
        finaloutput = finlaout_layer(fc2,fc_dims=10,name="final")

        # cost
        loss = tf.losses.softmax_cross_entropy(y,finaloutput)

        # optimize
        LEARNING_RATE_BASE = 0.001
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

AlexNet = AlexNet_Mode()
AlexNet.load_data()
