import numpy as np
import tensorflow as tf
import time
import random
import scipy.io as sio
from video_dataset import Dataset
import utils
import options
from test import test 
import os

args = options.parser.parse_args()

if not os.path.exists('./ckpt/' + args.model_name):
    os.makedirs('./ckpt/' + args.model_name)

def _variable_with_weight_decay(name, shape, wd):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def main():

    # Placeholders
    learning_rate = tf.placeholder(tf.float32)
    feature_seq = tf.placeholder(tf.float32, [args.batch_size, args.max_seqlen, args.feature_size])
    labels = tf.placeholder(tf.float32, [args.batch_size, args.num_class])

    seq_len = tf.cast(tf.reduce_sum(tf.sign(tf.reduce_max(tf.abs(feature_seq), axis=2)), axis=1), tf.int32)
    fseq = feature_seq[:,:tf.reduce_max(seq_len),:]

    sgn = tf.sign(tf.reduce_sum(tf.abs(fseq),keep_dims=True,axis=2))
    seq_len = tf.cast(tf.reduce_sum(tf.sign(tf.reduce_max(tf.abs(fseq), axis=2)), axis=1), tf.int32)
    k = tf.cast(tf.ceil(tf.cast(seq_len, tf.float32)/8), tf.int32)
  
    # Model
    with tf.device('/gpu:0'):
        with tf.variable_scope('Fully_Connected'):
            fc_W = _variable_with_weight_decay('fc_w', [args.feature_size, args.feature_size], 0.0005)
            fc_b = _variable_with_weight_decay('fc_b', [args.feature_size], 0.0000)
            feature = tf.matmul(fseq, tf.tile(tf.expand_dims(fc_W,0),[args.batch_size,1,1])) + fc_b
            feature = tf.nn.relu(feature)
            feature = tf.nn.dropout(feature, 0.3)

        with tf.variable_scope('Attention') as an:
            atn_W = _variable_with_weight_decay('atn_w', [args.feature_size, args.num_class], 0.0005)
            atn_b = _variable_with_weight_decay('atn_b', [args.num_class], 0.0000)
            temporal_logits = tf.matmul(feature, tf.tile(tf.expand_dims(atn_W,0),[args.batch_size,1,1])) + atn_b
    
        # MILL
        logits = []
        for i in range(args.batch_size):
            tmp, _ = tf.nn.top_k(tf.transpose(temporal_logits[i,:seq_len[i],:],[1,0]), k=k[i])
            logits.append(tf.reduce_mean(tf.transpose(tmp,[1,0]), axis=0))
        logits = tf.stack(logits)
        mill = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        tf.add_to_collection('losses', mill * args.Lambda)
        tf.summary.scalar('MILL', mill)

        # CASL
        tmp = tf.exp(temporal_logits)*sgn
        attention = tf.div(tmp, tf.reduce_sum(tmp,axis=1,keep_dims=True))
        attn_classwise_feat = tf.matmul(tf.transpose(feature,[0,2,1]),attention)      
        norm_comp_attention = sgn*(1-attention)/tf.cast(tf.expand_dims(tf.expand_dims(tf.maximum(seq_len-1,1),axis=1),axis=1), tf.float32)
        comp_attn_classwise_feat = tf.matmul(tf.transpose(feature,[0,2,1]),norm_comp_attention)
        casl, n_tmp = 0., 0.
        for i in range(0, args.num_similar*2, 2):
            f1 = attn_classwise_feat[i,:,:]
            f2 = attn_classwise_feat[i+1,:,:]
            f3 = comp_attn_classwise_feat[i,:,:]
            f4 = comp_attn_classwise_feat[i+1,:,:]
            d1 = 1 - tf.reduce_sum(f1*f2,axis=0)/(tf.norm(f1,axis=0)*tf.norm(f2,axis=0))
            d2 = 1 - tf.reduce_sum(f1*f4,axis=0)/(tf.norm(f1,axis=0)*tf.norm(f4,axis=0))
            d3 = 1 - tf.reduce_sum(f2*f3,axis=0)/(tf.norm(f2,axis=0)*tf.norm(f3,axis=0))
            casl = casl + tf.reduce_sum(tf.maximum(0., d1-d2+0.5)*0.5*tf.cast(tf.greater(labels[i,:],0),tf.float32)*tf.cast(tf.greater(labels[i+1,:],0),tf.float32))
            casl = casl + tf.reduce_sum(tf.maximum(0., d1-d3+0.5)*0.5*tf.cast(tf.greater(labels[i,:],0),tf.float32)*tf.cast(tf.greater(labels[i+1,:],0),tf.float32))
            n_tmp = n_tmp + tf.reduce_sum(tf.cast(tf.greater(labels[i,:],0),tf.float32)*tf.cast(tf.greater(labels[i+1,:],0),tf.float32))
        casl = casl / n_tmp
        tf.add_to_collection('losses', casl * (1-args.Lambda))
        tf.summary.scalar('CASL', casl)

        total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        tf.summary.scalar('Total Loss', total_loss)
    
        apply_gradient_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss)
  

    # Initialize tensorflow graph
    init = tf.global_variables_initializer()
    config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(init)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./tensorboards/' + args.model_name, sess.graph)
    saver = tf.train.Saver(max_to_keep=200)

    # Start from scratch or load model
    if args.pretrained_ckpt is None:
        iter_num = 0
    else:  
        iter_num = np.load('iter_num.npy')
        saver.restore(sess,tf.train.latest_checkpoint('./ckpt/' + args.pretrained_ckpt + '/'))

    # Initialize dataset
    dataset = Dataset(args)

    #Start training
    for i in range(iter_num, args.max_iter):
   
        # Train
        batch_feature_seq, batch_labels = dataset.load_data(n_similar=args.num_similar)
        batch_labels = batch_labels / np.sum(batch_labels, axis=1, keepdims=True)
        _, cost, sumry = sess.run([apply_gradient_op, total_loss, merged], feed_dict={feature_seq:batch_feature_seq, labels:batch_labels, learning_rate: args.lr})
        train_writer.add_summary(sumry, i)

        print('Iteration: %d, Loss: %.5f' %(i, cost))

        if i % 500 == 0:
            #sumry = sess.run(merged, feed_dict={feature_seq: batch_feature_seq, labels:batch_labels, learning_rate: lr, keep_prob: None})
            #train_writer.add_summary(sumry, i)
            np.save('iter_num.npy', i)
            saver.save(sess, './ckpt/' + args.model_name + '/model', global_step=i)
            test(dataset, args, i)     

if __name__ == "__main__":
   main()
