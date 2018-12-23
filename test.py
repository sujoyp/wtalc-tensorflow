import os
import numpy as np
import tensorflow as tf
import time
import random
import scipy.io as sio
from video_dataset import Dataset
import utils
import options
from classificationMAP import getClassificationMAP as cmAP
from detectionMAP import getDetectionMAP as dmAP

def _variable_with_weight_decay(name, shape, wd):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def test(dataset, args, itr):

    # Placeholders
    feature_seq = tf.placeholder(tf.float32, [1, args.max_seqlen, args.feature_size])
    seq_len = tf.cast(tf.reduce_sum(tf.sign(tf.reduce_max(tf.abs(feature_seq), axis=2)), axis=1), tf.int32)

    fseq = feature_seq[:,:tf.reduce_max(seq_len),:]
    sgn = tf.sign(tf.reduce_sum(tf.abs(fseq),keep_dims=True,axis=2))
    seq_len = tf.cast(tf.reduce_sum(tf.sign(tf.reduce_max(tf.abs(fseq), axis=2)), axis=1), tf.int32)
    k = tf.cast(tf.ceil(tf.cast(seq_len, tf.float32)/8), tf.int32)

    with tf.device('/gpu:0'):
        with tf.variable_scope('Fully_Connected', reuse=True):
            fc_W = _variable_with_weight_decay('fc_w', [args.feature_size, args.feature_size], 0.0005)
            fc_b = _variable_with_weight_decay('fc_b', [args.feature_size], 0.0000)
            feature = tf.matmul(fseq, tf.tile(tf.expand_dims(fc_W,0),[1,1,1])) + fc_b
            feature = tf.nn.relu(feature)
            feature = tf.nn.dropout(feature, 1.0)

        with tf.variable_scope('Attention', reuse=True) as an:
            atn_W = _variable_with_weight_decay('atn_w', [args.feature_size, args.num_class], 0.0005)
            atn_b = _variable_with_weight_decay('atn_b', [args.num_class], 0.0000)
            temporal_logits = tf.matmul(feature, tf.tile(tf.expand_dims(atn_W,0),[1,1,1])) + atn_b

    # Initialize everything
    init = tf.global_variables_initializer()
    config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(init)
    saver = tf.train.Saver()
    saver.restore(sess,tf.train.latest_checkpoint('./ckpt/' + args.model_name + '/'))
  
    # Test
    element_logit_stack = []
    instance_logit_stack = []
    label_stack = []
    done = False

    while not done:
        features, label, done = dataset.load_data(is_training=False)
        element_logit = []
        features = np.concatenate([features, np.zeros((args.max_seqlen - len(features) % args.max_seqlen, args.feature_size))], axis=0)
        for i in range(0, len(features), args.max_seqlen):
            tmp = sess.run(temporal_logits, feed_dict={feature_seq: np.expand_dims(features[i:i+args.max_seqlen], axis=0)})
            if len(element_logit) == 0:
                element_logit = np.squeeze(tmp)
            else:
                element_logit = np.concatenate([element_logit, np.squeeze(tmp)], axis=0)

        element_logit = np.array(element_logit)
        instance_logit = np.mean(np.sort(element_logit, axis=0)[::-1][:max(1,int(np.shape(element_logit)[0]/8)),:], axis=0)
        instance_logit = np.exp(instance_logit); instance_logit /= np.sum(instance_logit)
        instance_logit_stack.append(instance_logit)  
        element_logit_stack.append(element_logit)
        label_stack.append(label)

    instance_logit_stack = np.array(instance_logit_stack)
    label_stack = np.array(label_stack)

    dmap, iou = dmAP(element_logit_stack, dataset.path_to_annotations)
    
    if args.dataset_name == 'Thumos14':
        test_set = sio.loadmat('test_set_meta.mat')['test_videos'][0]
        for i in range(np.shape(label_stack)[0]):
            if test_set[i]['background_video'] == 'YES':
                label_stack[i,:] = np.zeros_like(label_stack[i,:])

    cmap = cmAP(instance_logit_stack, label_stack)
    np.save('package-1.npy', (instance_logit_stack, label_stack))
    print('Classification map %f' %cmap)
    print('Detection map @ %f = %f' %(iou[0], dmap[0]))
    print('Detection map @ %f = %f' %(iou[1], dmap[1]))
    print('Detection map @ %f = %f' %(iou[2], dmap[2]))
    print('Detection map @ %f = %f' %(iou[3], dmap[3]))
    print('Detection map @ %f = %f' %(iou[4], dmap[4]))
        
    utils.write_to_file(args.dataset_name, dmap, cmap, itr)

