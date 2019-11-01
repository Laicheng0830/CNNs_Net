"""
!/usr/bin/env python
-*- coding:utf-8 -*-
Author: eric.lai
Created on 2019/10/31 16:02
"""

from CNNs_OPSv2 import *
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


class ResNet50(object):
    def build_model(self, height, width, channel,num_classes=10, is_training=True,
                 scope="resnet50"):
        self.is_training = is_training
        self.num_classes = num_classes
        inputs = tf.placeholder(dtype= tf.float32, shape = [None,height,width,channel])
        y = tf.placeholder(dtype=tf.float32, shape=[None, num_classes])

        with tf.variable_scope(scope):
            # construct the model
            net = conv_layer(inputs, 64, 7, 2, name="conv1") # -> [batch, 112, 112, 64]
            net = tf.nn.relu(batch_norm(net, is_training=self.is_training, scope="bn1"))
            net = pool_layer(net, 3, 2, name="maxpool1")  # -> [batch, 56, 56, 64]
            net = self._block(net, 256, 3, init_stride=1, is_training=self.is_training,
                              scope="block2")           # -> [batch, 56, 56, 256]
            net = self._block(net, 512, 4, is_training=self.is_training, scope="block3")
                                                        # -> [batch, 28, 28, 512]
            net = self._block(net, 1024, 6, is_training=self.is_training, scope="block4")
                                                        # -> [batch, 14, 14, 1024]
            net = self._block(net, 2048, 3, is_training=self.is_training, scope="block5")
                                                        # -> [batch, 7, 7, 2048]
            net = avg_pool_layer(net, 7, 1, name="avgpool5")    # -> [batch, 1, 1, 2048]
            net = tf.squeeze(net, [1, 2], name="SpatialSqueeze") # -> [batch, 2048]
            self.logits = fc_layer(net, self.num_classes, "fc6")       # -> [batch, num_classes]
            self.predictions = tf.nn.softmax(self.logits)
            self.loss = tf.losses.softmax_cross_entropy(self.predictions, y)
            optimize = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.loss)

            correct_prediction = tf.equal(tf.argmax(self.predictions, 1), tf.argmax(y, 1))
            accurary = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))
        return dict(
                x=inputs,
                y=y,
                optimize=optimize,
                correct_prediction=self.predictions,
                cost=self.loss,
                accurary = accurary
            )


    def _block(self, x, n_out, n, init_stride=2, is_training=True, scope="block"):
        with tf.variable_scope(scope):
            h_out = n_out // 4
            out = self._bottleneck(x, h_out, n_out, stride=init_stride,
                                   is_training=is_training, scope="bottlencek1")
            for i in range(1, n):
                out = self._bottleneck(out, h_out, n_out, is_training=is_training,
                                       scope=("bottlencek%s" % (i + 1)))
            return out

    def _bottleneck(self, x, h_out, n_out, stride=None, is_training=True, scope="bottleneck"):
        """ A residual bottleneck unit"""
        n_in = x.get_shape()[-1]
        if stride is None:
            stride = 1 if n_in == n_out else 2

        with tf.variable_scope(scope):
            h = conv_layer(x, h_out, 1, stride=stride, name="conv_1")
            h = batch_norm(h, is_training=is_training, scope="bn_1")
            h = tf.nn.relu(h)
            h = conv_layer(h, h_out, 3, stride=1, name="conv_2")
            h = batch_norm(h, is_training=is_training, scope="bn_2")
            h = tf.nn.relu(h)
            h = conv_layer(h, n_out, 1, stride=1, name="conv_3")
            h = batch_norm(h, is_training=is_training, scope="bn_3")

            if n_in != n_out:
                shortcut = conv_layer(x, n_out, 1, stride=stride, name="conv_4")
                shortcut = batch_norm(shortcut, is_training=is_training, scope="bn_4")
            else:
                shortcut = x
            return tf.nn.relu(shortcut + h)

    def init_sess(self):
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess = tf.Session()
        self.sess.run(init)

    def train_network(self, graph, x_train, y_train):
        # Tensorfolw Adding more and more nodes to the previous graph results in a larger and larger memory footprint
        # reset graph
        tf.reset_default_graph()
        self.sess.run(graph['optimize'], feed_dict={graph['inputs']: x_train, graph['y']: y_train})

    def load_data(self):
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

        g = self.build_model(28, 28, 1, 10)
        self.init_sess()
        # Build the model first, then initialize it, just once
        for i in range(1000):
            batch_xs, batch_ys = mnist.train.next_batch(1000)
            batch_xs = np.reshape(batch_xs,[-1,28,28,1])
            self.train_network(g,batch_xs,batch_ys)
            if i%5==0:
                print("cost: ",self.sess.run(g['cost'],feed_dict={g['inputs']:batch_xs, g['y']:batch_ys}),"accurary: ",self.sess.run(g['accurary'],feed_dict={g['x']:batch_xs, g['y']:batch_ys}))
                # print("correct_prediction",self.sess.run(g['correct_prediction'],feed_dict={g['x']:batch_xs,g['y']:batch_ys}))


if __name__ == "__main__":
    resnet50 = ResNet50()
    resnet50.load_data()