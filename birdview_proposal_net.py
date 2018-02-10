# 3D proposal networks
from __future__ import print_function
from os.path import expanduser
import tensorflow as tf
import numpy as np
import model

import TensorflowUtils as utils
import datetime
import random
import pdb

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-16.mat'
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_bool('debug', "True", "Debug mode: True/ False")
tf.flags.DEFINE_float("learning_rate", "1e-3", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_integer("batch_size", "1", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "Birdview_proposal_logs/", "path to logs directory")
tf.flags.DEFINE_bool('regularization', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")



# training loss
def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)

# main function
# train_data[0] = birdview; train_data[1] = anchor_label; 
# train_data[2] = anchor_label_mask; train_data[3] = anchor_reg;
# train_data[4] = anchor_reg_mask; 
# val_data has same structure as train data
# MAX_EPOCH: training epochs
# weight: weight of classification loss
# reg_weight: weight of l2 regularization term 

def train_network(train_data, val_data, MAX_EPOCH, weight, reg_weight, keep_prob):
    # data preparation
    tr_birdview = train_data[0]
    tr_anchor_label = train_data[1]
    tr_anchor_reg = train_data[2]
    tr_anchor_label_mask = train_data[3]
    tr_anchor_reg_mask = train_data[4]

    val_birdview = val_data[0]
    val_anchor_label = val_data[1]
    val_anchor_reg = val_data[2]
    val_anchor_label_mask = val_data[3]
    val_anchor_reg_mask = val_data[4]

    BATCHES = tr_birdview.shape[0]
    BATCHES1 = val_birdview.shape[0]
    BIRDVIEW_CHANNEL = tr_birdview.shape[3]
    IMAGE_SIZE = [tr_birdview.shape[1], tr_birdview.shape[2]]
    OUTPUT_IMAGE_SIZE = [tr_birdview.shape[1] / 4, tr_birdview.shape[2] / 4]
    NUM_OF_REGRESSION_VALUE = tr_anchor_reg.shape[4]
    NUM_OF_ANCHOR = tr_anchor_reg.shape[3]
    
    with tf.name_scope("Input"):
        image = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, IMAGE_SIZE[0], IMAGE_SIZE[1], BIRDVIEW_CHANNEL], name="input_image")
        gt_anchor_labels = tf.placeholder(tf.int64, shape=[FLAGS.batch_size * OUTPUT_IMAGE_SIZE[0] * OUTPUT_IMAGE_SIZE[1], NUM_OF_ANCHOR], name = "gt_anchor_labels")
        gt_anchor_regs = tf.placeholder(tf.float32, shape=[FLAGS.batch_size * OUTPUT_IMAGE_SIZE[0] * OUTPUT_IMAGE_SIZE[1], NUM_OF_ANCHOR, NUM_OF_REGRESSION_VALUE], name = "gt_anchor_regs")
        sample_anchor_cls_masks = tf.placeholder(tf.float32, shape=[FLAGS.batch_size * OUTPUT_IMAGE_SIZE[0] * OUTPUT_IMAGE_SIZE[1], NUM_OF_ANCHOR], name = "sample_anchor_masks")
        nonsample_anchor_cls_masks = tf.placeholder(tf.float32, shape=[FLAGS.batch_size * OUTPUT_IMAGE_SIZE[0] * OUTPUT_IMAGE_SIZE[1], NUM_OF_ANCHOR], name = "nonsample_anchor_cls_masks")
        anchor_reg_masks = tf.placeholder(tf.float32, shape=[FLAGS.batch_size * OUTPUT_IMAGE_SIZE[0] * OUTPUT_IMAGE_SIZE[1], NUM_OF_ANCHOR], name = "anchor_reg_masks")


    with tf.name_scope("3D-RPN"):
        net, classification_loss, regression_loss, Overall_loss, all_rpn_logits_softmax, all_rpn_regs = model.birdview_proposal_net(image, gt_anchor_labels, gt_anchor_regs, sample_anchor_cls_masks, anchor_reg_masks, weight, reg_weight, FLAGS.model_dir, FLAGS.batch_size, FLAGS.debug, keep_prob)
        sample_accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(all_rpn_logits_softmax, dimension=1), tf.reshape(gt_anchor_labels, [-1])), tf.float32) * tf.reshape(sample_anchor_cls_masks, [-1])) / tf.reduce_sum(tf.reshape(sample_anchor_cls_masks, [-1]))
        non_sample_accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(all_rpn_logits_softmax, dimension=1), tf.reshape(gt_anchor_labels, [-1])), tf.float32) * tf.reshape(nonsample_anchor_cls_masks, [-1])) / tf.reduce_sum(tf.reshape(nonsample_anchor_cls_masks, [-1]))
        
    with tf.name_scope("summary"):
        print("Setting up summary op...")
        tf.summary.image("bird_view_intensity", image[:, :, :, BIRDVIEW_CHANNEL - 2 : BIRDVIEW_CHANNEL - 1], max_outputs =2)
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
        # pdb.set_trace()
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

            # train in entire dataset
            for batch in range(0, BATCHES / FLAGS.batch_size):
                # for training 
                image_input_batch = tr_birdview[batch * FLAGS.batch_size : batch * FLAGS.batch_size + FLAGS.batch_size]
                anchor_label_batch = tr_anchor_label[batch * FLAGS.batch_size : batch * FLAGS.batch_size + FLAGS.batch_size]
                anchor_regs_batch = tr_anchor_reg[batch * FLAGS.batch_size : batch * FLAGS.batch_size + FLAGS.batch_size]
                before_sample_anchor_labelmask_batch = tr_anchor_label_mask[batch * FLAGS.batch_size : batch * FLAGS.batch_size + FLAGS.batch_size]
                anchor_regs_mask_batch = tr_anchor_reg_mask[batch * FLAGS.batch_size : batch * FLAGS.batch_size + FLAGS.batch_size]
                
                negative_index = np.where(np.logical_and(anchor_label_batch == 0, before_sample_anchor_labelmask_batch == 1))
                positive_index = np.where(anchor_label_batch == 1)


                # for validation
                val_image_input_batch = val_birdview[batch1 * FLAGS.batch_size : batch1 * FLAGS.batch_size + FLAGS.batch_size]
                val_anchor_label_batch = val_anchor_label[batch1 * FLAGS.batch_size : batch1 * FLAGS.batch_size + FLAGS.batch_size]
                val_anchor_regs_batch = val_anchor_reg[batch1 * FLAGS.batch_size : batch1 * FLAGS.batch_size + FLAGS.batch_size]
                val_before_sample_anchor_labelmask_batch = val_anchor_label_mask[batch1 * FLAGS.batch_size : batch1 * FLAGS.batch_size + FLAGS.batch_size]
                val_anchor_regs_mask_batch = val_anchor_reg_mask[batch1 * FLAGS.batch_size : batch1 * FLAGS.batch_size + FLAGS.batch_size]

                val_negative_index = np.where(np.logical_and(val_anchor_label_batch == 0, val_before_sample_anchor_labelmask_batch == 1))
                val_positive_index = np.where(val_anchor_label_batch == 1)

                batch1+=1
                if batch1 >= BATCHES1:
                    batch1 = 0

                if len(positive_index[0]) > 0:

                    # for training 
                    neg_sample_index = random.sample(range(len(negative_index[0])), len(positive_index[0]))
                    after_sample_anchor_labelmask_batch = np.zeros(before_sample_anchor_labelmask_batch.shape)
                    after_sample_anchor_labelmask_batch[negative_index[0][neg_sample_index], negative_index[1][neg_sample_index], negative_index[2][neg_sample_index], negative_index[3][neg_sample_index]] = 1
                    after_sample_anchor_labelmask_batch[positive_index[0], positive_index[1], positive_index[2], positive_index[3]] = 1
                    

                    # reshape data to fit input
                    anchor_label_batch = np.reshape(anchor_label_batch, [FLAGS.batch_size * OUTPUT_IMAGE_SIZE[0] * OUTPUT_IMAGE_SIZE[1], NUM_OF_ANCHOR])
                    anchor_regs_batch = np.reshape(anchor_regs_batch, [FLAGS.batch_size * OUTPUT_IMAGE_SIZE[0] * OUTPUT_IMAGE_SIZE[1], NUM_OF_ANCHOR, NUM_OF_REGRESSION_VALUE])
                    after_sample_anchor_labelmask_batch = np.reshape(after_sample_anchor_labelmask_batch, [FLAGS.batch_size * OUTPUT_IMAGE_SIZE[0] * OUTPUT_IMAGE_SIZE[1], NUM_OF_ANCHOR])
                    before_sample_anchor_labelmask_batch = np.reshape(before_sample_anchor_labelmask_batch, [FLAGS.batch_size * OUTPUT_IMAGE_SIZE[0] * OUTPUT_IMAGE_SIZE[1], NUM_OF_ANCHOR])
                    anchor_regs_mask_batch = np.reshape(anchor_regs_mask_batch, [FLAGS.batch_size * OUTPUT_IMAGE_SIZE[0] * OUTPUT_IMAGE_SIZE[1], NUM_OF_ANCHOR]) 

                    feed_dict = {image: image_input_batch, gt_anchor_labels: anchor_label_batch,
                                 gt_anchor_regs: anchor_regs_batch, sample_anchor_cls_masks:after_sample_anchor_labelmask_batch,
                                 nonsample_anchor_cls_masks:before_sample_anchor_labelmask_batch, anchor_reg_masks: anchor_regs_mask_batch}


                    # for validation
                    val_neg_sample_index = random.sample(range(len(val_negative_index[0])), len(val_positive_index[0]))
                    val_after_sample_anchor_labelmask_batch = np.zeros(val_before_sample_anchor_labelmask_batch.shape)
                    val_after_sample_anchor_labelmask_batch[val_negative_index[0][val_neg_sample_index], val_negative_index[1][val_neg_sample_index], val_negative_index[2][val_neg_sample_index], val_negative_index[3][val_neg_sample_index]] = 1
                    val_after_sample_anchor_labelmask_batch[val_positive_index[0], val_positive_index[1], val_positive_index[2], val_positive_index[3]] = 1
                    

                    # reshape data to fit input
                    val_anchor_label_batch = np.reshape(val_anchor_label_batch, [FLAGS.batch_size * OUTPUT_IMAGE_SIZE[0] * OUTPUT_IMAGE_SIZE[1], NUM_OF_ANCHOR])
                    val_anchor_regs_batch = np.reshape(val_anchor_regs_batch, [FLAGS.batch_size * OUTPUT_IMAGE_SIZE[0] * OUTPUT_IMAGE_SIZE[1], NUM_OF_ANCHOR, NUM_OF_REGRESSION_VALUE])
                    val_after_sample_anchor_labelmask_batch = np.reshape(val_after_sample_anchor_labelmask_batch, [FLAGS.batch_size * OUTPUT_IMAGE_SIZE[0] * OUTPUT_IMAGE_SIZE[1], NUM_OF_ANCHOR])
                    val_before_sample_anchor_labelmask_batch = np.reshape(val_before_sample_anchor_labelmask_batch, [FLAGS.batch_size * OUTPUT_IMAGE_SIZE[0] * OUTPUT_IMAGE_SIZE[1], NUM_OF_ANCHOR])
                    val_anchor_regs_mask_batch = np.reshape(val_anchor_regs_mask_batch, [FLAGS.batch_size * OUTPUT_IMAGE_SIZE[0] * OUTPUT_IMAGE_SIZE[1], NUM_OF_ANCHOR]) 

                    feed_dict1 = {image: val_image_input_batch, gt_anchor_labels: val_anchor_label_batch,
                                 gt_anchor_regs: val_anchor_regs_batch, sample_anchor_cls_masks:val_after_sample_anchor_labelmask_batch,
                                 nonsample_anchor_cls_masks:val_before_sample_anchor_labelmask_batch, anchor_reg_masks: val_anchor_regs_mask_batch}
                    #####################################################################################################################################################################
                    # pdb.set_trace()
                    # training
                    
                    # save
                    if (total_iter) % 10 == 0:
                        # pdb.set_trace()

                        train_Overall_loss, train_classification_loss, train_regression_loss, train_sample_accuracy, train_nonsample_accuracy, summary_str = sess.run([Overall_loss, classification_loss, regression_loss, sample_accuracy, 
                                                                                                                                                                       non_sample_accuracy, summary_op], feed_dict=feed_dict)

                        
                        print("Iter: %d, Num of postives: %d, Num of negatives: %d, Train_Overall_loss:%g, Train_classification_loss:%g, \
                               Train_regression_loss:%g, Sample_training_accuracy: %g, Nonsample_training_accuracy: %g" % (total_iter, \
                               len(positive_index[0]), len(negative_index[0]), train_Overall_loss, train_classification_loss, train_regression_loss, train_sample_accuracy, train_nonsample_accuracy))              
                        summary_writer.add_summary(summary_str, total_iter)
                        saver.save(sess, FLAGS.logs_dir + "model.ckpt", total_iter)


                    if (total_iter) % 50 == 0:
                        # pdb.set_trace()

                        val_Overall_loss, val_classification_loss, val_regression_loss, val_sample_accuracy, val_nonsample_accuracy = sess.run([Overall_loss, classification_loss, regression_loss, sample_accuracy, 
                                                                                                                                                                       non_sample_accuracy], feed_dict=feed_dict1)

                        
                        print("Iter: %d, Num of postives: %d, Num of negatives: %d, Validation_Overall_loss:%g, Validation_classification_loss:%g, \
                               Validation_regression_loss:%g, Sample_training_accuracy: %g, Nonsample_training_accuracy: %g" % (total_iter, \
                               len(val_positive_index[0]), len(val_negative_index[0]), val_Overall_loss, val_classification_loss, val_regression_loss, val_sample_accuracy, val_nonsample_accuracy))              


                    total_iter += 1

                    sess.run(train_op, feed_dict=feed_dict)
           

def prediction_network(data):
    # data preparation

    BATCHES = data.shape[0]
    BIRDVIEW_CHANNEL = data.shape[3]
    IMAGE_SIZE = [data.shape[1], data.shape[2]]
    OUTPUT_IMAGE_SIZE = [data.shape[1] / 4, data.shape[2] / 4]
    label = []
    reg = []
    
    with tf.name_scope("Input"):
        
        image = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, IMAGE_SIZE[0], IMAGE_SIZE[1], BIRDVIEW_CHANNEL], name="input_image")
        gt_anchor_labels = tf.placeholder(tf.int64, shape=[FLAGS.batch_size * OUTPUT_IMAGE_SIZE[0] * OUTPUT_IMAGE_SIZE[1], 4], name = "gt_anchor_labels")
        gt_anchor_regs = tf.placeholder(tf.float32, shape=[FLAGS.batch_size * OUTPUT_IMAGE_SIZE[0] * OUTPUT_IMAGE_SIZE[1], 4, 6], name = "gt_anchor_regs")
        sample_anchor_cls_masks = tf.placeholder(tf.float32, shape=[FLAGS.batch_size * OUTPUT_IMAGE_SIZE[0] * OUTPUT_IMAGE_SIZE[1], 4], name = "sample_anchor_masks")
        nonsample_anchor_cls_masks = tf.placeholder(tf.float32, shape=[FLAGS.batch_size * OUTPUT_IMAGE_SIZE[0] * OUTPUT_IMAGE_SIZE[1], 4], name = "nonsample_anchor_cls_masks")
        anchor_reg_masks = tf.placeholder(tf.float32, shape=[FLAGS.batch_size * OUTPUT_IMAGE_SIZE[0] * OUTPUT_IMAGE_SIZE[1], 4], name = "anchor_reg_masks")


    with tf.name_scope("3D-RPN"):
        net, classification_loss, regression_loss, Overall_loss, all_rpn_logits_softmax, all_rpn_regs = model.birdview_proposal_net(image, gt_anchor_labels, gt_anchor_regs, sample_anchor_cls_masks, anchor_reg_masks, 1, 0, FLAGS.model_dir, FLAGS.batch_size, FLAGS.debug)        

    with tf.name_scope("Prediction"):

        # uncomment BELOW TO RUNNING ON CPU
        # pdb.set_trace()
        # config = tf.ConfigProto(device_count = {'GPU': 0})
        # sess = tf.Session(config=config)    
        # uncomment to run on GPU
        sess = tf.Session()
        ###############################
        
        print("Setting up Saver...")
        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored...")


        for i in range (len(data)):
            birdview = data[i : i + 1]

            feed_dict = {image: birdview}

            _all_rpn_logits_softmax, _all_rpn_regs = sess.run([all_rpn_logits_softmax, all_rpn_regs], feed_dict=feed_dict)
            _all_rpn_logits_softmax = np.argmax(_all_rpn_logits_softmax, 1)
            label.append(_all_rpn_logits_softmax.reshape(data.shape[1] / 4, data.shape[2] / 4, 4, 2))
            reg.append(_all_rpn_regs.reshape(data.shape[1] / 4, data.shape[2] / 4, 4, 6))
            print("prediction %d"%i)

    return label, reg         
                     

