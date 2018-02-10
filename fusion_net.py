# Created by Weizhi Li
# MV3D network
from __future__ import print_function
from os.path import expanduser
import tensorflow as tf
import numpy as np

import TensorflowUtils as utils
import datetime
import random
import pdb
import model

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-16.mat'
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_bool('debug', "True", "Debug mode: True/ False")
tf.flags.DEFINE_float("learning_rate", "1e-3", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_integer("batch_size", "1", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "MV3D_logs/", "path to logs directory")
tf.flags.DEFINE_bool('regularization', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")



# training loss
def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)

# main function
def train_network(train_data, val_data, MAX_EPOCH, weight, reg_weight, keep_prob):
    # keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")

    data_birdview = train_data[0]
    data_frontview = train_data[1]
    data_rgbview = train_data[2]
    data_birdview_rois = train_data[3]
    data_frontview_rois = train_data[4]
    data_rgbview_rois = train_data[5]

    data_birdview_box_ind = train_data[6]
    data_frontview_box_ind = train_data[7]
    data_rgbview_box_ind = train_data[8]

    data_ROI_labels = train_data[9]
    data_ROI_regs = train_data[10]
    data_cls_mask = train_data[11]
    data_reg_mask = train_data[12]

    val_data_birdview = val_data[0]
    val_data_frontview = val_data[1]
    val_data_rgbview = val_data[2]

    val_data_birdview_rois = val_data[3]
    val_data_frontview_rois = val_data[4]
    val_data_rgbview_rois = val_data[5]

    val_data_birdview_box_ind = val_data[6]
    val_data_frontview_box_ind = val_data[7]
    val_data_rgbview_box_ind = val_data[8]

    val_data_ROI_labels = val_data[9]
    val_data_ROI_regs = val_data[10]
    val_data_cls_mask = val_data[11]
    val_data_reg_mask = val_data[12]

    BATCHES = np.shape(data_birdview)[0]
    BATCHES1 = np.shape(val_data_birdview)[0]
    NUM_OF_REGRESSION_VALUE = np.shape(data_ROI_regs)[2]
    IMAGE_SIZE_BIRDVIEW = [data_birdview.shape[1], data_birdview.shape[2]]
    IMAGE_SIZE_FRONTVIEW = [data_frontview.shape[1], data_frontview.shape[2]]
    IMAGE_SIZE_RGB = [data_rgbview.shape[1], data_rgbview.shape[2]]
    BIRDVIEW_CHANNEL = data_birdview.shape[3]
    FRONTVIEW_CHANNEL = data_frontview.shape[3]
    RGB_CHANNEL = data_rgbview.shape[3]
    NUM_OF_ROI = 200
    ROI_H = 10
    ROI_W = 2

    with tf.name_scope("Input"):
        birdview = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, IMAGE_SIZE_BIRDVIEW[0], IMAGE_SIZE_BIRDVIEW[1], BIRDVIEW_CHANNEL], name="birdview")
        frontview = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, IMAGE_SIZE_FRONTVIEW[0], IMAGE_SIZE_FRONTVIEW[1], FRONTVIEW_CHANNEL], name="frontview")
        rgbview = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, IMAGE_SIZE_RGB[0], IMAGE_SIZE_RGB[1], RGB_CHANNEL], name="rgbview")
 

        frontview_rois = tf.placeholder(tf.float32, shape=[FLAGS.batch_size * NUM_OF_ROI, 4], name = "birdview_rois") # rois: [upleft corner y, upleft corner x, down_right corner y, down_right corner x]
        rgbview_rois = tf.placeholder(tf.float32, shape=[FLAGS.batch_size * NUM_OF_ROI, 4], name = "frontview_rois")
        birdview_rois = tf.placeholder(tf.float32, shape=[FLAGS.batch_size * NUM_OF_ROI, 4], name = "rgbview_rois")  

        frontview_box_ind = tf.placeholder(tf.int32, shape=[FLAGS.batch_size * NUM_OF_ROI], name = "frontview_box_ind") # rois: [index of feat]
        rgbview_box_ind = tf.placeholder(tf.int32, shape=[FLAGS.batch_size * NUM_OF_ROI], name = "rgbview_box_ind")
        birdview_box_ind = tf.placeholder(tf.int32, shape=[FLAGS.batch_size * NUM_OF_ROI], name = "birdview_box_ind")  

        before_sample_cls_mask = tf.placeholder(tf.int32, shape=[FLAGS.batch_size * NUM_OF_ROI, 1], name = "before_sample_cls_mask")
        after_sample_cls_mask = tf.placeholder(tf.int32, shape=[FLAGS.batch_size * NUM_OF_ROI, 1], name = "after_sample_cls_mask")
        reg_mask = tf.placeholder(tf.int32, shape=[FLAGS.batch_size * NUM_OF_ROI, 1], name = "reg_mask")
        gt_ROI_labels = tf.placeholder(tf.int64, shape=[FLAGS.batch_size, NUM_OF_ROI], name = "gt_ROI_labels")
        gt_ROI_regs = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, NUM_OF_ROI, NUM_OF_REGRESSION_VALUE], name = "gt_ROI_regs")


    with tf.name_scope("MV3D"):
        Overall_loss, classification_loss, regression_loss, logits_cls, logits_reg = model.MV3D(birdview, frontview, rgbview, after_sample_cls_mask, reg_mask, gt_ROI_labels, gt_ROI_regs, 
                                                                                                birdview_rois, frontview_rois, rgbview_rois, birdview_box_ind, frontview_box_ind, rgbview_box_ind, 
                                                                                                ROI_H, ROI_W, NUM_OF_REGRESSION_VALUE, FLAGS.model_dir, weight, reg_weight, FLAGS.debug, keep_prob )
        
        before_smaple_accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(logits_cls, dimension=1), tf.reshape(gt_ROI_labels, [-1])), tf.float32) * tf.cast(tf.reshape(before_sample_cls_mask, [-1]), tf.float32)) / tf.reduce_sum(tf.cast(tf.reshape(before_sample_cls_mask, [-1]), tf.float32))
        after_smaple_accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(logits_cls, dimension=1), tf.reshape(gt_ROI_labels, [-1])), tf.float32) * tf.cast(tf.reshape(after_sample_cls_mask, [-1]), tf.float32)) / tf.reduce_sum(tf.cast(tf.reshape(after_sample_cls_mask, [-1]), tf.float32))

    with tf.name_scope("summary"):
        print("Setting up summary op...")
        tf.summary.image("bird_view_intensity", birdview[:, :, :, BIRDVIEW_CHANNEL - 2 : BIRDVIEW_CHANNEL - 1], max_outputs =2)
        tf.summary.image("frontview", frontview, max_outputs =2)
        tf.summary.image("rgbview", rgbview, max_outputs =2)
        trainable_var = tf.trainable_variables()
        tf.summary.scalar("Overall_loss", Overall_loss)
        tf.summary.scalar("regression_loss", regression_loss)
        tf.summary.scalar("classification_loss", classification_loss)
        if FLAGS.debug:
            for var in trainable_var:
                utils.add_to_regularization_and_summary(var)
        summary_op = tf.summary.merge_all()

    with tf.name_scope("Train"):
        train_op = train(Overall_loss, trainable_var)


        # uncomment BELOW TO RUNNING ON CPU
        config = tf.ConfigProto(device_count = {'GPU': 0})
        sess = tf.Session(config=config)  
        # uncomment to run on GPU   
        # sess = tf.Session()
        ###############################
        
        print("Setting up Saver...")
        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)

        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored...")


        total_iter = 0
        batch1 = 0
        for epoch in range (MAX_EPOCH):
                # train only in single image
                # adjust values in range() to train on multiple images
            for batch in range(0, BATCHES / FLAGS.batch_size):
            # for batch in range(5, 6):

                # for train
                birdview_batch = data_birdview[batch * FLAGS.batch_size : batch * FLAGS.batch_size + FLAGS.batch_size]
                frontview_batch = data_frontview[batch * FLAGS.batch_size : batch * FLAGS.batch_size + FLAGS.batch_size] 
                rgbview_batch = data_rgbview[batch * FLAGS.batch_size : batch * FLAGS.batch_size + FLAGS.batch_size]
                
                birdview_rois_batch = data_birdview_rois[batch * FLAGS.batch_size * NUM_OF_ROI : batch * FLAGS.batch_size * NUM_OF_ROI+ NUM_OF_ROI * FLAGS.batch_size]
                frontview_rois_batch = data_frontview_rois[batch * FLAGS.batch_size * NUM_OF_ROI : batch * FLAGS.batch_size *NUM_OF_ROI + NUM_OF_ROI * FLAGS.batch_size]
                rgbview_rois_batch = data_rgbview_rois[batch * FLAGS.batch_size * NUM_OF_ROI : batch * FLAGS.batch_size * NUM_OF_ROI+ NUM_OF_ROI * FLAGS.batch_size] 

                birdview_box_ind_batch = data_birdview_box_ind[batch * FLAGS.batch_size * NUM_OF_ROI : batch * FLAGS.batch_size * NUM_OF_ROI+ NUM_OF_ROI * FLAGS.batch_size]
                frontview_box_ind_batch = data_frontview_box_ind[batch * FLAGS.batch_size * NUM_OF_ROI : batch * FLAGS.batch_size *NUM_OF_ROI + NUM_OF_ROI * FLAGS.batch_size]
                rgbview_box_ind_batch = data_rgbview_box_ind[batch * FLAGS.batch_size * NUM_OF_ROI : batch * FLAGS.batch_size * NUM_OF_ROI+ NUM_OF_ROI * FLAGS.batch_size] 

                cls_mask_batch = data_cls_mask[batch * FLAGS.batch_size * NUM_OF_ROI : batch * FLAGS.batch_size * NUM_OF_ROI+ NUM_OF_ROI * FLAGS.batch_size]
                reg_mask_batch = data_reg_mask[batch * FLAGS.batch_size * NUM_OF_ROI : batch * FLAGS.batch_size * NUM_OF_ROI+ NUM_OF_ROI * FLAGS.batch_size]
                ROI_labels_batch = data_ROI_labels[batch * FLAGS.batch_size  : batch * FLAGS.batch_size + FLAGS.batch_size]
                ROI_regs_batch = data_ROI_regs[batch * FLAGS.batch_size  : batch * FLAGS.batch_size + FLAGS.batch_size]

                positive_index = np.where(reg_mask_batch == 1)

                # for validation
                val_birdview_batch = val_data_birdview[batch1 * FLAGS.batch_size : batch1 * FLAGS.batch_size + FLAGS.batch_size]
                val_frontview_batch = val_data_frontview[batch1 * FLAGS.batch_size : batch1 * FLAGS.batch_size + FLAGS.batch_size] 
                val_rgbview_batch = val_data_rgbview[batch1 * FLAGS.batch_size : batch1 * FLAGS.batch_size + FLAGS.batch_size] / 255.

                val_birdview_rois_batch = val_data_birdview_rois[batch1 * FLAGS.batch_size * NUM_OF_ROI : batch1 * FLAGS.batch_size * NUM_OF_ROI+ NUM_OF_ROI * FLAGS.batch_size]
                val_frontview_rois_batch = val_data_frontview_rois[batch1 * FLAGS.batch_size * NUM_OF_ROI : batch1 * FLAGS.batch_size *NUM_OF_ROI + NUM_OF_ROI * FLAGS.batch_size]
                val_rgbview_rois_batch = val_data_rgbview_rois[batch1 * FLAGS.batch_size * NUM_OF_ROI : batch1 * FLAGS.batch_size * NUM_OF_ROI+ NUM_OF_ROI * FLAGS.batch_size] 

                val_birdview_box_ind_batch = val_data_birdview_box_ind[batch1 * FLAGS.batch_size * NUM_OF_ROI : batch1 * FLAGS.batch_size * NUM_OF_ROI+ NUM_OF_ROI * FLAGS.batch_size]
                val_frontview_box_ind_batch = val_data_frontview_box_ind[batch1 * FLAGS.batch_size * NUM_OF_ROI : batch1 * FLAGS.batch_size *NUM_OF_ROI + NUM_OF_ROI * FLAGS.batch_size]
                val_rgbview_box_ind_batch = val_data_rgbview_box_ind[batch1 * FLAGS.batch_size * NUM_OF_ROI : batch1 * FLAGS.batch_size * NUM_OF_ROI+ NUM_OF_ROI * FLAGS.batch_size] 

                val_cls_mask_batch = val_data_cls_mask[batch1 * FLAGS.batch_size * NUM_OF_ROI : batch1 * FLAGS.batch_size * NUM_OF_ROI+ NUM_OF_ROI * FLAGS.batch_size]
                val_reg_mask_batch = val_data_reg_mask[batch1 * FLAGS.batch_size * NUM_OF_ROI : batch1 * FLAGS.batch_size * NUM_OF_ROI+ NUM_OF_ROI * FLAGS.batch_size]
                val_ROI_labels_batch = val_data_ROI_labels[batch1 * FLAGS.batch_size  : batch1 * FLAGS.batch_size + FLAGS.batch_size]
                val_ROI_regs_batch = val_data_ROI_regs[batch1 * FLAGS.batch_size  : batch1 * FLAGS.batch_size + FLAGS.batch_size]
              
                val_positive_index = np.where(val_reg_mask_batch == 1)

                
                if len(positive_index[0]) > 0:
                    
                    negative_index = np.where(np.logical_and(cls_mask_batch == 1, reg_mask_batch == 0))
                    neg_sample_index = random.sample(range(len(negative_index[0])), len(positive_index[0]))
                    after_sample_cls_mask_batch = np.zeros(cls_mask_batch.shape)
                    after_sample_cls_mask_batch[positive_index[0], positive_index[1]] = 1
                    after_sample_cls_mask_batch[negative_index[0][neg_sample_index], negative_index[1][neg_sample_index]] = 1

                    feed_dict = {birdview: birdview_batch,  frontview: frontview_batch, rgbview: rgbview_batch, 
                                 birdview_rois: birdview_rois_batch, frontview_rois: frontview_rois_batch, 
                                 rgbview_rois: rgbview_rois_batch, birdview_box_ind: birdview_box_ind_batch,
                                 frontview_box_ind:frontview_box_ind_batch, rgbview_box_ind: rgbview_box_ind_batch,
                                 before_sample_cls_mask:cls_mask_batch, after_sample_cls_mask: after_sample_cls_mask_batch, 
                                 reg_mask:reg_mask_batch, gt_ROI_labels: ROI_labels_batch, gt_ROI_regs: ROI_regs_batch}

             
                    val_negative_index = np.where(np.logical_and(val_cls_mask_batch == 1, val_reg_mask_batch == 0))
                    val_neg_sample_index = random.sample(range(len(val_negative_index[0])), len(val_positive_index[0]))
                    val_after_sample_cls_mask_batch = np.zeros(val_cls_mask_batch.shape)
                    val_after_sample_cls_mask_batch[val_positive_index[0], val_positive_index[1]] = 1
                    val_after_sample_cls_mask_batch[val_negative_index[0][val_neg_sample_index], val_negative_index[1][val_neg_sample_index]] = 1

                    feed_dict1 = {birdview: val_birdview_batch,  frontview: val_frontview_batch, rgbview: val_rgbview_batch, 
                                 birdview_rois: val_birdview_rois_batch, frontview_rois: val_frontview_rois_batch, 
                                 rgbview_rois: val_rgbview_rois_batch, birdview_box_ind: val_birdview_box_ind_batch,
                                 frontview_box_ind: val_frontview_box_ind_batch, rgbview_box_ind: val_rgbview_box_ind_batch,
                                 before_sample_cls_mask: val_cls_mask_batch, after_sample_cls_mask: val_after_sample_cls_mask_batch, 
                                 reg_mask: val_reg_mask_batch, gt_ROI_labels: val_ROI_labels_batch, gt_ROI_regs: val_ROI_regs_batch}

                    batch1 += 1
                    if batch1 >= BATCHES1:
                        batch1 = 0

                    if total_iter % 10 == 0:       
                        train_Overall_loss, train_classification_loss, train_regression_loss, train_nonsample_accuracy, train_sample_accuracy, summary_str = sess.run([Overall_loss, classification_loss, regression_loss, 
                                                                                                                                                                  before_smaple_accuracy, after_smaple_accuracy, summary_op],
                                                                                                                                                                  feed_dict=feed_dict)

                        print("Iter: %d, Num of postives: %d, Num of negatives: %d, Train_Overall_loss:%g, Train_classification_loss:%g, \
                              Train_regression_loss:%g, Sample_training_accuracy: %g, Nonsample_training_accuracy: %g" % (total_iter, len(positive_index[0]), len(negative_index[0]), train_Overall_loss,
                              train_classification_loss, train_regression_loss, train_sample_accuracy, train_nonsample_accuracy))              
                        summary_writer.add_summary(summary_str, total_iter)
                        saver.save(sess, FLAGS.logs_dir + "model.ckpt", total_iter)

                    if total_iter % 50 == 0:       
                        val_Overall_loss, val_classification_loss, val_regression_loss, val_nonsample_accuracy, val_sample_accuracy = sess.run([Overall_loss, classification_loss, regression_loss, 
                                                                                                                                                                  before_smaple_accuracy, after_smaple_accuracy],
                                                                                                                                                                  feed_dict= feed_dict1)

                        print("Iter: %d, Num of postives: %d, Num of negatives: %d, Val_Overall_loss:%g, Val_classification_loss:%g, \
                              Val_regression_loss:%g, Sample_training_accuracy: %g, Nonsample_training_accuracy: %g" % (total_iter, len(positive_index[0]), len(negative_index[0]), val_Overall_loss,
                              val_classification_loss, val_regression_loss, val_sample_accuracy, val_nonsample_accuracy))              
                 
                    sess.run(train_op, feed_dict=feed_dict)
                    total_iter += 1

def prediction_network(data):
    # keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")

    data_birdview = data[0]
    data_frontview = data[1]
    data_rgbview = data[2]

    data_birdview_rois = data[3]
    data_frontview_rois = data[4]
    data_rgbview_rois = data[5]

    data_birdview_box_ind = data[6]
    data_frontview_box_ind = data[7]
    data_rgbview_box_ind = data[8]

    IMAGE_SIZE_BIRDVIEW = [data_birdview.shape[1], data_birdview.shape[2]]
    IMAGE_SIZE_FRONTVIEW = [data_frontview.shape[1], data_frontview.shape[2]]
    IMAGE_SIZE_RGB = [data_rgbview.shape[1], data_rgbview.shape[2]]
    BIRDVIEW_CHANNEL = data_birdview.shape[3]
    FRONTVIEW_CHANNEL = data_frontview.shape[3]
    RGB_CHANNEL = data_rgbview.shape[3]
    NUM_OF_ROI = 200
    ROI_H = 10
    ROI_W = 2
    label = []
    reg = []

    with tf.name_scope("Input"):
        birdview = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, IMAGE_SIZE_BIRDVIEW[0], IMAGE_SIZE_BIRDVIEW[1], BIRDVIEW_CHANNEL], name="birdview")
        frontview = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, IMAGE_SIZE_FRONTVIEW[0], IMAGE_SIZE_FRONTVIEW[1], FRONTVIEW_CHANNEL], name="frontview")
        rgbview = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, IMAGE_SIZE_RGB[0], IMAGE_SIZE_RGB[1], RGB_CHANNEL], name="rgbview")
 

        frontview_rois = tf.placeholder(tf.float32, shape=[FLAGS.batch_size * NUM_OF_ROI, 4], name = "birdview_rois") # rois: [upleft corner y, upleft corner x, down_right corner y, down_right corner x]
        rgbview_rois = tf.placeholder(tf.float32, shape=[FLAGS.batch_size * NUM_OF_ROI, 4], name = "frontview_rois")
        birdview_rois = tf.placeholder(tf.float32, shape=[FLAGS.batch_size * NUM_OF_ROI, 4], name = "rgbview_rois")  

        frontview_box_ind = tf.placeholder(tf.int32, shape=[FLAGS.batch_size * NUM_OF_ROI], name = "frontview_box_ind") # rois: [index of feat]
        rgbview_box_ind = tf.placeholder(tf.int32, shape=[FLAGS.batch_size * NUM_OF_ROI], name = "rgbview_box_ind")
        birdview_box_ind = tf.placeholder(tf.int32, shape=[FLAGS.batch_size * NUM_OF_ROI], name = "birdview_box_ind")  

        before_sample_cls_mask = tf.placeholder(tf.int32, shape=[FLAGS.batch_size * NUM_OF_ROI, 1], name = "before_sample_cls_mask")
        after_sample_cls_mask = tf.placeholder(tf.int32, shape=[FLAGS.batch_size * NUM_OF_ROI, 1], name = "after_sample_cls_mask")
        reg_mask = tf.placeholder(tf.int32, shape=[FLAGS.batch_size * NUM_OF_ROI, 1], name = "reg_mask")
        gt_ROI_labels = tf.placeholder(tf.int64, shape=[FLAGS.batch_size, NUM_OF_ROI], name = "gt_ROI_labels")
        gt_ROI_regs = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, NUM_OF_ROI, 24], name = "gt_ROI_regs")


    with tf.name_scope("MV3D"):
        Overall_loss, classification_loss, regression_loss, logits_cls, logits_reg = model.MV3D(birdview, frontview, rgbview, after_sample_cls_mask, reg_mask, gt_ROI_labels, gt_ROI_regs, 
                                                                                                birdview_rois, frontview_rois, rgbview_rois, birdview_box_ind, frontview_box_ind, rgbview_box_ind, 
                                                                                                ROI_H, ROI_W, 24, FLAGS.model_dir, 1, 0, FLAGS.debug)
        


    with tf.name_scope("Prediction"):


        # uncomment BELOW TO RUNNING ON CPU
        config = tf.ConfigProto(device_count = {'GPU': 0})
        sess = tf.Session(config=config)  
        # uncomment to run on GPU   
        # sess = tf.Session()
        ###############################
        
        print("Setting up Saver...")
        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored...")

        for i in range (len(data_birdview)):
            birdview_batch = data_birdview[i : i + 1]
            frontview_batch = data_frontview[i : i + 1]
            rgbview_batch = data_rgbview[i : i + 1]

            birdview_rois_batch = data_birdview_rois[i * NUM_OF_ROI : i * NUM_OF_ROI + NUM_OF_ROI]
            frontview_rois_batch = data_frontview_rois[i * NUM_OF_ROI : i * NUM_OF_ROI + NUM_OF_ROI]
            rgbview_rois_batch = data_rgbview_rois[i * NUM_OF_ROI : i * NUM_OF_ROI + NUM_OF_ROI]

            birdview_box_ind_batch = data_birdview_box_ind[i * NUM_OF_ROI : i * NUM_OF_ROI + NUM_OF_ROI]
            frontview_box_ind_batch = data_frontview_box_ind[i * NUM_OF_ROI : i * NUM_OF_ROI + NUM_OF_ROI]
            rgbview_box_ind_batch = data_rgbview_box_ind[i * NUM_OF_ROI : i * NUM_OF_ROI + NUM_OF_ROI]


            feed_dict = {birdview: birdview_batch,  frontview: frontview_batch, rgbview: rgbview_batch, 
                         birdview_rois: birdview_rois_batch, frontview_rois: frontview_rois_batch, 
                         rgbview_rois: rgbview_rois_batch, birdview_box_ind: birdview_box_ind_batch,
                         frontview_box_ind:frontview_box_ind_batch, rgbview_box_ind: rgbview_box_ind_batch}

             
            _logits_cls, _logits_reg = sess.run([logits_cls, logits_reg], feed_dict=feed_dict)
            _logits_cls = np.argmax(_logits_cls, 1)
            label.append(_logits_cls)
            reg.append(_logits_reg)
            print('prediction %d'%i)

