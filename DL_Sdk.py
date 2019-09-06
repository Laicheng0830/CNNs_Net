# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np


#def leaky_relu(x, alpha=0.3):    
#    return tf.maximum(x, alpha * x)

def leaky_relu(x, leak=0.2, name="lrelu"):
    """
    :param x:Tensor,typr = tf.float32
    :param leak:float
    :return:Tensor
    """
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        relu = f1 * x + f2 * abs(x)
        return relu

def activation_function(x,activation_function_type):
    """
    :param x:Tensor
    :param activation_function_type:str activation function type,'lrelu','tanh','sigmoid','relu','linear'
    :return:Tensor
    """
    if(activation_function_type=='lrelu'):
        h = leaky_relu(x)
    if(activation_function_type=='tanh'):
        h = tf.tanh(x)
    if(activation_function_type=='sigmoid'):
        h = tf.sigmoid(x)
    if(activation_function_type=='relu'):
        h = tf.nn.relu(x)
    if(activation_function_type=='linear'):
        h = x        
    return h

def batch_norm(layer,center=True,scale=True,training=True,name='bn'):
    """
    :param layer:Tensor
    :param center:True or False ， If True, add offset of beta to normalized tensor. If False, beta is ignored.
    :param scale:True or False ,  If True, multiply by gamma. If False, gamma is not used.
    :param training:bool , Whether to return the output in training mode or in inference mode.
    :param name:String, layery name
    :return:Tensor
    """
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        bn = tf.layers.batch_normalization(layer,center=center,scale=scale,training=training,name=name)
    return bn
    
def fully_connected_layer(data, layer_dims, w_init = None, b_init = None,bias = True,activation_function_type='lrelu',
                          data_type=tf.float32,name='fcl',keep_prob=1):
    """
    :param data:Tensor
    :param layer_dims:int
    :param w_init:default
    :param b_init:default
    :param activation_function_type:str activation function type,'lrelu','tanh','sigmoid','relu','linear'
    :param data_type:tf.float
    :param name:String
    :param keep_prob:Prevention of overfitting, 0-1
    :return:Tensor
    """
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):

        data_shape = np.shape(data)
        
        if w_init is None:
#            w_init = tf.contrib.layers.xavier_initializer()
            w_init = tf.truncated_normal_initializer(mean=0,stddev=0.01)
            
        if b_init is None:
            b_init = tf.constant_initializer(0.0)
    
        w = tf.get_variable(shape=[data_shape[1],layer_dims], initializer=w_init, name="w", dtype=data_type)
        
        if bias == True:
            b = tf.get_variable(shape=[layer_dims], initializer=b_init, name="b", dtype=data_type)
            h = activation_function(tf.matmul(data, w) + b,activation_function_type)
        else:
            h = activation_function(tf.matmul(data, w),activation_function_type)
            
        if((keep_prob<1) and keep_prob>0):
            out = tf.nn.dropout(h,keep_prob)
        else:
            out = h
     
#     return out   
    return out

def conv2d_encoder(data, kenerl_size, strides=[1, 1, 1, 1], w_init = None,norm = True,is_training = True, padding='SAME',
                 activation_function_type='lrelu', stddev=0.05, data_type=tf.float32,name="conv2en",keep_prob=1, with_w=False):
    """
    :param data:Tensor,must 4-D tensor
    :param kenerl_size:list ,kenerl_size of the new or existing variable.
    :param strides:A list of ints. 1-D tensor of length 4.
    :param w_init:An optional. Defaults to None.
    :param norm:An optional bool. Defaults to True.
    :param is_training:An optional bool. Defaults to True.
    :param padding: A string from: "SAME", "VALID". The type of padding algorithm to use.
    :param activation_function_type:str activation function type,'lrelu','tanh','sigmoid','relu','linear'
    :param keep_prob:Restrictions in the 0-1,Prevention of overfitting
    :param with_w:A optional,return out or return out,w
    :return:Tensor
    """
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        if w_init is None:
            w_init=tf.truncated_normal_initializer(mean=0,stddev=stddev)
        w = tf.get_variable(shape=kenerl_size, initializer=w_init, name="w", dtype=data_type)
        
        cov = tf.nn.conv2d(data, w, strides=strides, padding=padding)
        if norm == True:
            bn = batch_norm(cov,training=is_training)
        else:
            bn = cov
            
        if((keep_prob<1) and keep_prob>0):
            cov_drop = tf.nn.dropout(bn,keep_prob)
        else:
            cov_drop = bn
            
        out = activation_function(cov_drop,activation_function_type)
        
    if(with_w == False):    
        return out
    else:
        return out,w
        
def con2d_decoder(data, kenerl_size,output_shape, strides=[1, 1, 1, 1], k_init = None,norm = True,is_training = True, padding='SAME',
                 activation_function_type='lrelu', stddev=0.05, data_type=tf.float32,name="conv2de",keep_prob=1, with_w=False):
    """
    :param data:Tensor
    :param kenerl_size:kenerl_size of the new or existing variable.
    :param output_shape:A 1-D Tensor representing the output shape of the deconvolution op.
    :param strides: A list of ints. The stride of the sliding window for each dimension of the input tensor.
    :param padding:A string, either 'VALID' or 'SAME'. The padding algorithm.
    :param activation_function_type:Str activation function type,'lrelu','tanh','sigmoid','relu','linear'
    :param with_w:True or False
    :return:Tensor
    """
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        
        if k_init is None:
            k_init = tf.random_normal_initializer(mean=0,stddev=stddev)
#            k_init = tf.truncated_normal_initializer(mean=0,stddev=stddev)
            
        kenerl = tf.get_variable(shape=kenerl_size, initializer=k_init, name="w", dtype=data_type) 
        
        cov = tf.nn.conv2d_transpose(data,kenerl,output_shape,strides, padding=padding,name=name)
        
        if norm == True:
            bn = batch_norm(cov,training=is_training)
        else:
            bn = cov
            
        if((keep_prob<1) and keep_prob>0):
            cov_drop = tf.nn.dropout(bn,keep_prob)
        else:
            cov_drop = bn
            
        out = activation_function(cov_drop,activation_function_type)
    
    if(with_w == False):    
        return out
    else:
        return out,kenerl


def separable_conv2d_encoder(data, depthwise_size,pointwise_size, d_init = None, p_init = None, strides=[1, 1, 1, 1], norm = True,
                             is_training = True, padding='SAME',activation_function_type='lrelu',stddev = 0.5, 
                             data_type=tf.float32,name="spconv2",keep_prob=1):
    """
   

    """
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        
        if d_init is None:
            d_init = tf.truncated_normal_initializer(mean=0,stddev=stddev)            
        if p_init is None:
            p_init = tf.truncated_normal_initializer(mean=0,stddev=stddev)
        
        depthwise_filter = tf.get_variable(shape=depthwise_size, initializer=d_init, name="depthwise_filter", dtype=data_type) 
        pointwise_filter = tf.get_variable(shape=pointwise_size, initializer=p_init, name="pointwise_filter", dtype=data_type)
        
        cov = tf.nn.separable_conv2d(data,depthwise_filter,pointwise_filter,strides,padding)
        
        if norm == True:
            bn = batch_norm(cov,training=is_training)
        else:
            bn = cov
            
        if((keep_prob<1) and keep_prob>0):
            cov_drop = tf.nn.dropout(bn,keep_prob)
        else:
            cov_drop = bn
            
        out = activation_function(cov_drop,activation_function_type)
        
    return out



def conv2d_layer(data, kenerl_size, strides=[1, 1, 1, 1], w_init = None, b_init = None,padding='SAME',
                 activation_function_type='lrelu', data_type=tf.float32,name="conv2",keep_prob=1):
    """
    :param data:Tensor
    :param kenerl_size:kenerl_size of the new or existing variable.
    :param strides:A string, either 'VALID' or 'SAME'. The padding algorithm.
    :param padding:A string, either 'VALID' or 'SAME'. The padding algorithm.
    :param activation_function_type:Str activation function type,'lrelu','tanh','sigmoid','relu','linear'
    :return:Tensor
    """
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):   
        if w_init is None:
            w_init = tf.contrib.layers.xavier_initializer()            
        if b_init is None:
            b_init = tf.constant_initializer(0.0)
        
        w = tf.get_variable(shape=kenerl_size, initializer=w_init, name="w", dtype=data_type) 
        b = tf.get_variable(shape=[kenerl_size[3]], initializer=b_init, name="b", dtype=data_type)

        cov = tf.nn.conv2d(data, w, strides=strides, padding=padding)
        h = activation_function(cov + b,activation_function_type)
        
        if((keep_prob<1) and keep_prob>0):
            out = tf.nn.dropout(h,keep_prob)
        else:
            out = h
    return out        
#    return out,w,b
    
def max_pooling_layer(x, kenerl_size,strides, padding='SAME',name="maxpl"):
    """
    :param x:Tensor
    :param kenerl_size:kenerl_size of the new or existing variable.
    :param strides:A string, either 'VALID' or 'SAME'. The padding algorithm.
    :param padding:A string, either 'VALID' or 'SAME'. The padding algorithm.
    :return:Tensor
    """
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        return tf.nn.max_pool(x, ksize=kenerl_size,strides=strides, padding=padding)

def upconv2d_layer(data, kenerl,output_shape ,strides=[1, 1, 1, 1], padding='SAME',name="upcv2"):
    """
    :param data:Tensor
    :param kenerl:A 4-D Tensor with the same type as value and shape [height, width, output_channels, in_channels].
    :param output_shape:A 1-D Tensor representing the output shape of the deconvolution op.
    :param strides:A list of ints. The stride of the sliding window for each dimension of the input tensor.
    :param padding: A string, either 'VALID' or 'SAME'. The padding algorithm.
    :return:Tensor
    """
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        return tf.nn.conv2d_transpose(data,kenerl,output_shape,strides, padding=padding,name=name)

def model_optimizer(lrate=1e-20,beta1=0.9, beta2=0.999,momentum_value=0.9,optimizer_type='opt'):
    """
    :param lrate:A Tensor or a floating point value. The learning rate.
    :param beta1:A float value or a constant float tensor. The exponential decay rate for the 1st moment estimates.
    :param beta2:A float value or a constant float tensor. The exponential decay rate for the 2nd moment estimates.
    :param momentum_value:A Tensor or a floating point value. The momentum.
    :param optimizer_type: An optional.'adam','momentum','sgd'
    :return:A list of variables.
    """
    if(optimizer_type=='adam'):
        optimizer = tf.train.AdamOptimizer(learning_rate=lrate,beta1=0.9, beta2=0.999)
    if(optimizer_type=='momentum'):
        optimizer = tf.train.MomentumOptimizer(learning_rate=lrate,momentum=momentum_value)
    if(optimizer_type=='sgd'):
        optimizer = tf.train.GradientDescentOptimizer(lrate)
    
    return optimizer

def cost_function(predict_data,label,cost_type='mse'):
    """
    :param predict_data:Tensor
    :param label:Tensor
    :param cost_type:An optional.'l1','mes','ce','hub'
    :return:loss
    """
    if(cost_type=='l1'):
        loss = tf.reduce_mean(tf.abs(predict_data - label))    
    if(cost_type=='mse'):
        loss = tf.reduce_mean(tf.square(predict_data - label))
    if(cost_type=='ce'):
        loss = -tf.reduce_mean(label * tf.log(tf.clip_by_value(predict_data, 1e-10, 1.0)))
    if(cost_type=='hub'):
        loss = tf.losses.huber_loss(label,predict_data)
    if(cost_type=='softmax'):
        loss = tf.losses.softmax_cross_entropy(label,predict_data)
    return loss
