"""
!/usr/bin/env python
-*- coding:utf-8 -*-
Author: eric.lai
Time: 2019/2/26 14:38
"""
import numpy as np
import tensorflow as tf

def conv_layer(data, ksize, stride, name, w_biases=False, padding="SAME"):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        w_init = tf.contrib.layers.xavier_initializer()
        w = tf.get_variable(name=name, shape=ksize, initializer=w_init)
        biases = tf.Variable(tf.constant(0.0, shape=[ksize[3]], dtype=tf.float32), 'biases')
    if w_biases == False:
        cov = tf.nn.conv2d(input=data, filter=w, strides=stride, padding=padding)
    else:
        cov = tf.nn.conv2d(input=data, filter=w, stride=stride, padding=padding) + biases
    return cov

def convLayer_nGPU(x, kHeight, kWidth, strideX, strideY,
              featureNum, name, padding = "SAME", groups = 1):
    """convolution and relu activate ,GPU Split into two runs"""
    channel = int(x.get_shape()[-1])
    conv = lambda a, b: tf.nn.conv2d(a, b, strides = [1, strideY, strideX, 1], padding = padding)
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape = [kHeight, kWidth, channel/groups, featureNum])
        b = tf.get_variable("b", shape = [featureNum])

        xNew = tf.split(value = x, num_or_size_splits = groups, axis = 3)
        wNew = tf.split(value = w, num_or_size_splits = groups, axis = 3)

        featureMap = [conv(t1, t2) for t1, t2 in zip(xNew, wNew)]
        mergeFeatureMap = tf.concat(axis = 3, values = featureMap)
        # print mergeFeatureMap.shape
        out = tf.nn.bias_add(mergeFeatureMap, b)
        return tf.nn.relu(tf.reshape(out, mergeFeatureMap.get_shape().as_list()), name = scope.name)


def pool_layer(data, ksize, stride, name, padding='VALID'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        max_pool = tf.nn.max_pool(value=data, ksize=ksize, strides=stride, padding=padding)
    return max_pool


def flatten(data):
    [a, b, c, d] = data.shape
    ft = tf.reshape(data, [-1, b * c * d])
    return ft


def fc_layer(data, name, fc_dims, w_biases = 'True'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        data_shape = data.get_shape().as_list()
        w_init = tf.contrib.layers.xavier_initializer()
        w = tf.get_variable(shape=[data_shape[1],fc_dims],name= 'w',initializer=w_init)
        biases = tf.Variable(tf.constant(0.0, shape=[fc_dims], dtype=tf.float32), 'biases')
        if w_biases == 'True':
            fc = tf.nn.relu(tf.matmul(data, w) + biases)
        else:
            fc = tf.nn.relu(tf.matmul(data,w))
    return fc


def finlaout_layer(data, name, fc_dims, w_biases = 'Flase'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        w_init = tf.contrib.layers.xavier_initializer()
        w = tf.get_variable(shape=[data.shape[1], fc_dims], name='w', initializer=w_init)
        biases = tf.Variable(tf.constant(0.0, shape=[fc_dims], dtype=tf.float32), 'biases')
        # fc = tf.nn.softmax(tf.matmul(data,w)+ biases)
        if w_biases == 'Flase':
            finlaout = tf.matmul(data, w)
        else:
            finlaout = tf.matmul(data,w)+ biases
    return finlaout

def one_hot(data, nclass):
    data_one_hot = np.zeros(shape=(len(data),nclass),dtype=np.float32)
    for i in range(len(data)):
        data_one_hot[i][data[i]] = 1
    return data_one_hot

def dropout(data, keeppro = 0.1,name = None):
    return tf.nn.dropout(data,keeppro,name)

def lrn(data, R, alpha, beta, name = None, bias = 1.0):
    return tf.nn.lrn(data, depth_radius=R, alpha=alpha, beta=beta, bias=bias, name=name)
