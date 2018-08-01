#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 09:35:30 2018

@author: RWH
@mail:rwhcartman@163.com
"""

import warnings
import tensorflow as tf
import numpy as np
warnings.filterwarnings('ignore')


class AlexNet:
    def __init__(self, input_x, keep_prob,
                 num_classes, skip_layer, weights_path='Default'):
        ## 初始化参数
        self.input_x = input_x
        self.keep_prob = keep_prob  # drop_out参数
        self.skip_layer = skip_layer  # 模型参数加载跳过
        if weights_path == 'Default':
            self.weights_path = 'bvlc_alexnet.npy'
        else:
            self.weights_path = weights_path
        self.num_classes = num_classes  # 有几类
        # Create the AlexNet Network Define
        self.create()
        ## 在这一步的位置tfVariable已经全部构成了
        #print(tf.trainable_variables())
            
    def conv(self, x, kernel_height, num_kernels, stride, \
             name, padding='SAME', padding_num=0, \
             groups=1):
        print ('name is {} np.shape(input) {}'.format(name, np.shape(input)))
        input_channels = int(x.get_shape()[-1])
        if padding_num != 0:
            ## padings 也是一个张量，代表每一维填充多少行/列，
            ## 但是有一个要求它的rank一定要和tensor的rank是一样的
            x = tf.pad(x, [[0, 0], [padding_num, padding_num], \
                           [padding_num, padding_num], [0, 0]])
            '''
            tf.pad的使用，第一个是填充0，后面两个是复制前几行或者列
            [1,1],[2,2]
            [1,1]指的是向上扩充一行，向下扩充一行
            [2,2]指的是向左扩充2列，向右扩充2列
            '''
        ## i ->x k -> conv-kernal
        convolve = lambda i, k: tf.nn.conv2d(i, k, strides=[1,stride,stride, 1], padding=padding)
        with tf.variable_scope(name) as scope:
            ## get_variable创建共享变量
            weights = tf.get_variable('weights', \
                                      shape=[kernel_height, kernel_height, \
                                             input_channels/groups, num_kernels])
            biases = tf.get_variable('biases', shape=[num_kernels])
        if groups == 1:
            conv = convolve(x, weights)
        else:
            '''axis = 3 是 channel的axis'''
            input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
            weights_groups = tf.split(axis=3, num_or_size_splits= \
                groups,value = weights)
            output_groups = [convolve(i, k) for i, k in \
                                             zip(input_groups, weights_groups)]
            conv = tf.concat(axis=3, values = output_groups)
        withBias = tf.reshape(tf.nn.bias_add(conv, biases),conv.get_shape().as_list())
        relu = tf.nn.relu(withBias)
        return relu

    def maxPooling(self, input, filter_size, stride, name, padding='SAME'):
        print ('name is {} np.shape(input) {}'.format(name, np.shape(input)))
        return tf.nn.max_pool(input, ksize=[1, filter_size, filter_size, 1],
                              strides=[1, stride, stride, 1], padding=padding, name=name)

    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        print ('name is {} np.shape(input) {}'.format(name, np.shape(input)))
        return tf.nn.local_response_normalization(input, depth_radius=radius,
                                                  alpha=alpha, beta=beta, bias=bias, name=name)

    def fc(self, input, num_in, num_out, name, drop_ratio=0, relu=True):
        print ('name is {} np.shape(input) {}'.format(name, np.shape(input)))
        with tf.variable_scope(name) as scope:
            # trainable: If True also add the variable to the graph collection
            weights = tf.get_variable('weights', \
                                      shape=[num_in, num_out], trainable=True)
            biases = tf.get_variable('biases', \
                                     shape=[num_out], trainable=True)
            ## Linear
            act = tf.nn.xw_plus_b(input, weights, biases, name=scope.name)
            if relu == True:
                relu = tf.nn.relu(act)
                if drop_ratio == 0:
                    return relu
                else:
                    return tf.nn.dropout(relu, 1.0 - drop_ratio)
            else:
                if drop_ratio == 0:
                    return act
                else:
                    return tf.nn.dropout(act, 1.0 - drop_ratio)

    ## 加载预训练的参数
    def load_weights(self, session):
        ## weights_dict is a dict var:params
        weights_dict = np.load(self.weights_path, encoding='bytes').item()
        for op_name in weights_dict:  ## 遍历参数
            if op_name not in self.skip_layer:
                with tf.variable_scope(op_name, reuse=True):
                    for data in weights_dict[op_name]:
                        print('op_name:',op_name,'weights:',data)
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases', trainable=False)
                            session.run(var.assign(data))
                        else:
                            var = tf.get_variable('weights', trainable=False)
                            session.run(var.assign(data))
                            
        ## 搭网络
    def create(self):
        # layer1
        conv1 = self.conv(self.input_x, 11, 96, 4, name='conv1',padding='VALID')
        pool1 = self.maxPooling(conv1, filter_size=3, stride=2, name='pool1',padding='VALID')
        norm1 = self.lrn(pool1, 2, 2e-05, 0.75, name='norm1')
        # layer2
        conv2 = self.conv(norm1, 5, 256, 1, name='conv2', padding_num=0, groups=2)
        pool2 = self.maxPooling(conv2, filter_size=3, stride=2, name='pool2',
                                padding='VALID')
        norm2 = self.lrn(pool2, 2, 2e-05, 0.75, name='norm2')
        # layer3
        conv3 = self.conv(norm2, 3, 384, 1, name='conv3')
        # layer4
        conv4 = self.conv(conv3, 3, 384, 1, name='conv4', groups=2)
        # layer5
        conv5 = self.conv(conv4, 3, 256, 1, name='conv5', groups=2)
        pool5 = self.maxPooling(conv5, filter_size=3, stride=2,
                                name='pool5', padding='VALID')
        # layer6
        flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])
        fc6 = self.fc(input=flattened, num_in=6 * 6 * 256, \
                      num_out=4096, name='fc6', drop_ratio=1.0 - self.keep_prob, \
                      relu=True)
        # layer7
        fc7 = self.fc(input=fc6, num_in=4096, num_out=4096, name='fc7', \
                      drop_ratio=1.0 - self.keep_prob, relu=True)
        # layer8
        self.fc8 = self.fc(input=fc7, num_in=4096, num_out=self.num_classes, \
                           name='fc8', drop_ratio=0, relu=False)