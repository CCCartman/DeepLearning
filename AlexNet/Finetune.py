#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 11:26:58 2018

@author: RWH
@mail:rwhcartman@163.com
"""
import os
import numpy as np
import cv2
import tensorflow as tf
from datetime import datetime
from AlexNet import AlexNet
from Datagenerator import ImageDataGenerator


def fineTunes():
    tf.reset_default_graph()
    train_file = 'catdog-train.txt'  #####
    val_file = 'catdog-test.txt'  #####
    learning_rate = 0.01  ## 学习率
    num_epochs = 5
    batch_size = 1
    dropout_rate = 0.5
    num_class = 2  #####
    train_layers = ['fc8', 'fc7']
    display_step = 10
    filewreiter_path = 'filewriter/catdogs'  #####
    checkpoint_path = 'filewriter/'  #####

    X = tf.placeholder(tf.float32, shape=[batch_size, 227, 227, 3])
    y = tf.placeholder(tf.float32, shape=[None, num_class])
    keep_prob = tf.placeholder(tf.float32)

    model = AlexNet(input_x=X, keep_prob=keep_prob,
                    num_classes=num_class, skip_layer=train_layers,
                    weights_path='Default')
    score = model.fc8
    ## 'fc8/weighs' 'fc8/bias'
    ## <tf.Variable 'conv1/weights:0' shape=(11, 11, 3, 96) dtype=float32_ref>
    ## .name 'conv1/weights:0'
    var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] \
                in train_layers]
    '''重新训练模型参数保存'''
    layer_list = list(set([v.name.split('/')[0] for v in tf.trainable_variables()]))
    params_dict = {_: [] for _ in layer_list}
    with tf.name_scope('cross_ent'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( \
            logits=score, labels=y))
    with tf.name_scope('train'):
        gradients = tf.gradients(ys=loss, xs=var_list)
        gradients = list(zip(gradients, var_list))
        optimizer = tf.train.GradientDescentOptimizer( \
            learning_rate=learning_rate)
        train_op = optimizer.apply_gradients(grads_and_vars=gradients)
        '''
        for gradient,var in gradients:
            tf.summary.histogram(var.name+'/gradient',gradient)
        for var in var_list:
            tf.summary.histogram(var.name,var)
        '''
        tf.summary.scalar('cross_entropy', loss)
        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(tf.arg_max(score, 1), tf.arg_max(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        tf.summary.scalar('accuracy', accuracy)
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(filewreiter_path)

        saver = tf.train.Saver()
        '''class_list, n_class, batch_size = 1, flip = True, \
                shuffle = False, mean = np.array([104., 117., 124.]), scale_size = (227,227)
        '''
        train_generator = ImageDataGenerator(class_list=train_file, \
                                             n_class=num_class, batch_size=batch_size, flip=True, \
                                             shuffle=True)
        val_generator = ImageDataGenerator(class_list=val_file, \
                                           n_class=num_class, batch_size=1, shuffle=False, \
                                           flip=False)
        ## 每一个epoch里训练\测试多少图片
        train_batchs_per_epochs = np.floor(
            train_generator.data_size / batch_size).astype(np.int16)
        print('train_batchs_per_epochs:', train_batchs_per_epochs)
        val_batchs_per_epochs = np.floor(
            val_generator.data_size / batch_size).astype(np.int16)
        print('val_batchs_per_epochs:', val_batchs_per_epochs)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            writer.add_graph(sess.graph)
            model.load_weights(sess)

            print('{} start training ...'.format(datetime.now()))
            print('{} open TensorBoard at --logdir {}'.format(datetime.now(),
                                                              filewreiter_path))

            for epoch in range(num_epochs):
                step = 1
                while step < train_batchs_per_epochs:
                    ## images one-hot labels
                    batch_xs, batch_ys = train_generator.getNext_batch()
                    sess.run(train_op, feed_dict={X: batch_xs,
                                                  y: batch_ys, keep_prob: dropout_rate})
                    if step % display_step == 0:  ## 打印
                        s, get_loss = sess.run([merged_summary, loss], \
                                               feed_dict={X: batch_xs, y: batch_ys, keep_prob: 0.5})
                        writer.add_summary(s, epoch * train_batchs_per_epochs + step)
                        print('{} steps number:{} loss is {}'.format(datetime.now(),
                                                                     epoch * train_batchs_per_epochs + step, get_loss))
                    step += 1
                print('{} start validation'.format(datetime.now()))
                test_acc = 0.
                test_count = 0
                for j in range(val_batchs_per_epochs):
                    batch_xs, batch_ys = val_generator.getNext_batch()
                    acc = sess.run(accuracy, feed_dict={X: batch_xs, y: batch_ys, keep_prob: 1.})
                    print('{} steps number:{} validation acc is {}'.format(datetime.now(),j,acc))
                    test_acc += acc
                    test_count += 1
                test_acc /= test_count
                print('{} validation Accuracy = {:.4f}'.format(datetime.now(),
                                                               test_acc))
                val_txt = open('validation_acc.txt','a')
                print('{} validation Accuracy = {:.4f}'.format(datetime.now(),
                                                               test_acc), file = val_txt)
                val_txt.close()
                val_generator.reset_pointer()
                train_generator.reset_pointer()
                ## 保存模型 每个epoch保存一次
                print('{} saving checkpoint of model ...'.format(
                    datetime.now()))

                checkpoint_name = os.path.join(checkpoint_path,'model_epoch' + str(epoch + 1) + '.ckpt')
                saver.save(sess,checkpoint_name,global_step = epoch)
                print('{} Model checkpoint saved at {}'.format(datetime.now(), checkpoint_name))

                ## 保存.npy形式参数
                # layer_list = list(set([v.name.split('/')[0] for v in tf.trainable_variables()]))
                # params_dict = {_:[] for _ in layer_list}
                w_b = sess.run(tf.global_variables())
                params =  [_.name.split('/')[0] for _ in tf.global_variables()]
                for _layer, _weights in zip(params, w_b):
                    if len(_weights.shape) > 1:
                        params_dict[_layer] = [_weights]
                    else:
                        params_dict[_layer].append(_weights)
                np.save('catdog_alexnet' + str(epoch) + '.npy' , params_dict)


if __name__ == '__main__':
    fineTunes()


