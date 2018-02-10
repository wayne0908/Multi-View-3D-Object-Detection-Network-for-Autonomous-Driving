# created by Weizhi
# birdview proposal net (3D RPN)
# 3D proposal network
# region fusion
import tensorflow as tf
import TensorflowUtils as utils
import numpy as np
import random
import pdb
# from roi_pooling.roi_pooling_ops import roi_pooling

# Vgg net modified for birdview input
def vgg_net_birdview(weights, image, debug, keep_prob):
    layers = (
        'birdview_conv1_1', 'birdview_relu1_1', 'birdview_conv1_2', 'birdview_relu1_2', 'birdview_pool1',

        'birdview_conv2_1', 'birdview_relu2_1', 'birdview_conv2_2', 'birdview_relu2_2', 'birdview_pool2',

        'birdview_conv3_1', 'birdview_relu3_1', 'birdview_conv3_2', 'birdview_relu3_2', 'birdview_conv3_3',
        'birdview_relu3_3', 'birdview_pool3',

        'birdview_conv4_1', 'birdview_relu4_1', 'birdview_conv4_2', 'birdview_relu4_2', 'birdview_conv4_3',
        'birdview_relu4_3'

        # 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        # 'relu5_3'
    )

    # output of retrained layer for vgg
    net = {}
    current = image
    channel = image.get_shape().as_list()[3]

    for i, name in enumerate(layers):
        kind = name[9:13]
        if kind == 'conv':
            if name == 'birdview_conv1_1':
                # Modify the first conv layer
                kernels, bias = weights[i][0][0][0][0]
                kernels = np.transpose(kernels, (1, 0, 2, 3))
                kernels = np.concatenate((np.repeat(kernels[:, :, 0 : 1], channel / 3, axis = 2), np.repeat(kernels[: , :, 1 : 2], channel / 3, axis = 2),
                                          np.repeat(kernels[:, :, 2 : 3], channel - 2 * (channel / 3), axis = 2)), axis = 2)
                kernels = utils.get_variable(kernels, name=name + "_w")
                bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
                # pdb.set_trace()
                current = utils.conv2d_basic(current, kernels, bias, keep_prob)
            elif name == 'birdview_conv4_1':
                # Modify the senventh conv layer
                # pdb.set_trace()
                kernels, bias = weights[i][0][0][0][0]
                kernels = np.transpose(kernels, (1, 0, 2, 3))
                sample_index = random.sample(range(512), 256)
                kernels = kernels[:, :, :, sample_index]
                bias = bias[:, sample_index]
                kernels = utils.get_variable(kernels, name=name + "_w")
                bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
                current = utils.conv2d_basic(current, kernels, bias, keep_prob)
            elif name == 'birdview_conv4_2':
                # Modify the eighth conv layer
                # pdb.set_trace()
                kernels, bias = weights[i][0][0][0][0]
                kernels = np.transpose(kernels, (1, 0, 2, 3))
                sample_index_1 = random.sample(range(512), 256)
                sample_index_2 = random.sample(range(512), 256)
                kernels = kernels[:, :, sample_index_1, :]
                kernels = kernels[:, :, :, sample_index_2]
                bias = bias[:, sample_index_2]
                kernels = utils.get_variable(kernels, name=name + "_w")
                bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
                current = utils.conv2d_basic(current, kernels, bias, keep_prob)
            elif name == 'birdview_conv4_3':
                # pdb.set_trace()
                # Modify the ninth conv layer
                kernels, bias = weights[i][0][0][0][0]
                kernels = np.transpose(kernels, (1, 0, 2, 3))
                sample_index_1 = random.sample(range(512), 256)
                sample_index_2 = random.sample(range(512), 256)
                kernels = kernels[:, :, sample_index_1, :]
                kernels = kernels[:, :, :, sample_index_2]
                bias = bias[:, sample_index_2]
                kernels = utils.get_variable(kernels, name=name + "_w")
                bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
                current = utils.conv2d_basic(current, kernels, bias, keep_prob)

            else:
                kernels, bias = weights[i][0][0][0][0]
                # matconvnet: weights are [width, height, in_channels, out_channels]
                # tensorflow: weights are [height, width, in_channels, out_channels]
                kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
                bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
                current = utils.conv2d_basic(current, kernels, bias, keep_prob)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            if debug: 
            	utils.add_activation_summary(current)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current
    # pdb.set_trace()

    return net


# Vgg net modified for frontview input
def vgg_net_frontview(weights, image, debug, keep_prob):
    layers = (
        'frontview_conv1_1', 'frontview_relu1_1', 'frontview_conv1_2', 'frontview_relu1_2', 'frontview_pool1',

        'frontview_conv2_1', 'frontview_relu2_1', 'frontview_conv2_2', 'frontview_relu2_2', 'frontview_pool2',

        'frontview_conv3_1', 'frontview_relu3_1', 'frontview_conv3_2', 'frontview_relu3_2', 'frontview_conv3_3',
        'frontview_relu3_3', 'frontview_pool3',

        'frontview_conv4_1', 'frontview_relu4_1', 'frontview_conv4_2', 'frontview_relu4_2', 'frontview_conv4_3',
        'frontview_relu4_3'

        # 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        # 'relu5_3'
    )

    # output of retrained layer for vgg
    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[10:14]
        if kind == 'conv':
            if name == 'frontview_conv4_1':
                # Modify the senventh conv layer
                # pdb.set_trace()
                kernels, bias = weights[i][0][0][0][0]
                kernels = np.transpose(kernels, (1, 0, 2, 3))
                sample_index = random.sample(range(512), 256)
                kernels = kernels[:, :, :, sample_index]
                bias = bias[:, sample_index]
                kernels = utils.get_variable(kernels, name=name + "_w")
                bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
                current = utils.conv2d_basic(current, kernels, bias, keep_prob)
            elif name == 'frontview_conv4_2':
                # Modify the eighth conv layer
                # pdb.set_trace()
                kernels, bias = weights[i][0][0][0][0]
                kernels = np.transpose(kernels, (1, 0, 2, 3))
                sample_index_1 = random.sample(range(512), 256)
                sample_index_2 = random.sample(range(512), 256)
                kernels = kernels[:, :, sample_index_1, :]
                kernels = kernels[:, :, :, sample_index_2]
                bias = bias[:, sample_index_2]
                kernels = utils.get_variable(kernels, name=name + "_w")
                bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
                current = utils.conv2d_basic(current, kernels, bias, keep_prob)
            elif name == 'frontview_conv4_3':
                # pdb.set_trace()
                # Modify the ninth conv layer
                kernels, bias = weights[i][0][0][0][0]
                kernels = np.transpose(kernels, (1, 0, 2, 3))
                sample_index_1 = random.sample(range(512), 256)
                sample_index_2 = random.sample(range(512), 256)
                kernels = kernels[:, :, sample_index_1, :]
                kernels = kernels[:, :, :, sample_index_2]
                bias = bias[:, sample_index_2]
                kernels = utils.get_variable(kernels, name=name + "_w")
                bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
                current = utils.conv2d_basic(current, kernels, bias, keep_prob)
            else:
                kernels, bias = weights[i][0][0][0][0]
                # matconvnet: weights are [width, height, in_channels, out_channels]
                # tensorflow: weights are [height, width, in_channels, out_channels]
                kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
                bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
                current = utils.conv2d_basic(current, kernels, bias, keep_prob)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            if debug: 
                utils.add_activation_summary(current)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current
    # pdb.set_trace()
    return net


# Vgg net modified for rgb input
def vgg_net_rgb(weights, image, debug, keep_prob):
    layers = (
        'rgb_conv1_1', 'rgb_relu1_1', 'rgb_conv1_2', 'rgb_relu1_2', 'rgb_pool1',

        'rgb_conv2_1', 'rgb_relu2_1', 'rgb_conv2_2', 'rgb_relu2_2', 'rgb_pool2',

        'rgb_conv3_1', 'rgb_relu3_1', 'rgb_conv3_2', 'rgb_relu3_2', 'rgb_conv3_3',
        'rgb_relu3_3', 'rgb_pool3',

        'rgb_conv4_1', 'rgb_relu4_1', 'rgb_conv4_2', 'rgb_relu4_2', 'rgb_conv4_3',
        'rgb_relu4_3'

        # 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        # 'relu5_3'
    )

    # output of retrained layer for vgg
    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[4:8]
        if kind == 'conv':
            if name == 'rgb_conv4_1':
                # Modify the senventh conv layer
                # pdb.set_trace()
                kernels, bias = weights[i][0][0][0][0]
                kernels = np.transpose(kernels, (1, 0, 2, 3))
                sample_index = random.sample(range(512), 256)
                kernels = kernels[:, :, :, sample_index]
                bias = bias[:, sample_index]
                kernels = utils.get_variable(kernels, name=name + "_w")
                bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
                current = utils.conv2d_basic(current, kernels, bias, keep_prob)
            elif name == 'rgb_conv4_2':
                # Modify the eighth conv layer
                # pdb.set_trace()
                kernels, bias = weights[i][0][0][0][0]
                kernels = np.transpose(kernels, (1, 0, 2, 3))
                sample_index_1 = random.sample(range(512), 256)
                sample_index_2 = random.sample(range(512), 256)
                kernels = kernels[:, :, sample_index_1, :]
                kernels = kernels[:, :, :, sample_index_2]
                bias = bias[:, sample_index_2]
                kernels = utils.get_variable(kernels, name=name + "_w")
                bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
                current = utils.conv2d_basic(current, kernels, bias, keep_prob)
            elif name == 'rgb_conv4_3':
                # pdb.set_trace()
                # Modify the ninth conv layer
                kernels, bias = weights[i][0][0][0][0]
                kernels = np.transpose(kernels, (1, 0, 2, 3))
                sample_index_1 = random.sample(range(512), 256)
                sample_index_2 = random.sample(range(512), 256)
                kernels = kernels[:, :, sample_index_1, :]
                kernels = kernels[:, :, :, sample_index_2]
                bias = bias[:, sample_index_2]
                kernels = utils.get_variable(kernels, name=name + "_w")
                bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
                current = utils.conv2d_basic(current, kernels, bias, keep_prob)
            else:
                kernels, bias = weights[i][0][0][0][0]
                # matconvnet: weights are [width, height, in_channels, out_channels]
                # tensorflow: weights are [height, width, in_channels, out_channels]
                kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
                bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
                current = utils.conv2d_basic(current, kernels, bias, keep_prob)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            if debug: 
                utils.add_activation_summary(current)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current
    # pdb.set_trace()

    return net

# 3D proposal network
def Proposal_net(birdview, frontview, rgbview, model_dir, MODEL_URL, debug, keep_prob):
    #""" 3D region proposal network """
    # input birdview, dropout probability, weight of vgg, ground-truth labels, ground-truth regression value, 
    # anchor classification mask and anchor regression mask
    # The birdview has more than three channel, thus we need to train first two conv layers in vgg-16
    MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-16.mat'
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(model_dir, MODEL_URL)
    # preprocessing
    # mean = model_data['normalization'][0][0][0]
    # mean_pixel = np.mean(mean, axis=(0, 1))
    # processed_image = utils.process_image(birdview, mean_pixel)
    weights = np.squeeze(model_data['layers'])     

    # vgg-birdview
    with tf.name_scope("birdview-Vgg-16"):
        
        birdview_net = vgg_net_birdview(weights, birdview, debug, keep_prob)
        current = birdview_net["birdview_relu4_3"]
    # upsample, output 256 channels
    with tf.name_scope("birdview_Upsample_layer"):
        kernels = utils.weight_variable([3, 3, 256, 256], name= "birdview_upsample_w")
        bias = utils.bias_variable([256], name="birdview_upsample_b")
        output_shape = current.get_shape().as_list()
        output_shape[1] *= 4
        output_shape[2] *= 4
        output_shape[3] = kernels.get_shape().as_list()[2]
        birdview_net['birdview_upsample'] = utils.conv2d_transpose_strided(current, kernels, bias, output_shape = output_shape, stride = 4, name = 'birdview_upsample', keep_prob = keep_prob)
        current = birdview_net['birdview_upsample']
        if debug: 
            utils.add_activation_summary(current)

        # vgg-birdview
    with tf.name_scope("frontview-Vgg-16"):
        frontview_net = vgg_net_frontview(weights, frontview, debug, keep_prob)
        current = frontview_net["frontview_relu4_3"]
        # pdb.set_trace()
    # upsample, output 256 channels
    with tf.name_scope("frontview_Upsample_layer"):
        kernels = utils.weight_variable([3, 3, 256, 256], name= "frontview_upsample_w")
        bias = utils.bias_variable([256], name="frontview_upsample_b")
        output_shape = current.get_shape().as_list()
        output_shape[1] *= 4
        output_shape[2] *= 4
        output_shape[3] = kernels.get_shape().as_list()[2]
        frontview_net['frontview_upsample'] = utils.conv2d_transpose_strided(current, kernels, bias, output_shape = output_shape, stride = 4, name = 'frontview_upsample', keep_prob = keep_prob)
        current = frontview_net['frontview_upsample']
        if debug: 
            utils.add_activation_summary(current)

        # vgg-birdview
    with tf.name_scope("rgb-Vgg-16"):
        rgbview_net = vgg_net_rgb(weights, rgbview, debug, keep_prob)
        current = rgbview_net["rgb_relu4_3"]
    # upsample, output 256 channels
    with tf.name_scope("rgb_Upsample_layer"):
        kernels = utils.weight_variable([3, 3, 256, 256], name= "rgb_upsample_w")
        bias = utils.bias_variable([256], name="rgb_upsample_b")
        output_shape = current.get_shape().as_list()
        output_shape[1] *= 2
        output_shape[2] *= 2
        output_shape[3] = kernels.get_shape().as_list()[2]
        rgbview_net['rgb_upsample'] = utils.conv2d_transpose_strided(current, kernels, bias, output_shape = output_shape, name = 'rgb_upsample', keep_prob = keep_prob)
        current = rgbview_net['rgb_upsample']
        if debug: 
            utils.add_activation_summary(current)
        
    return birdview_net, frontview_net, rgbview_net

# region pooling
# def region_pooling(birdview_feat, frontview_feat, rgbview_feat, birdview_proposals, frontview_proposals, rgbview_proposals, batch_size, ROI_H, ROI_W):

def region_pooling(birdview_feat, frontview_feat, rgbview_feat, birdview_rois, frontview_rois, rgbview_rois, birdview_rois_ind, frontview_rois_ind, rgbview_rois_ind, ROI_H, ROI_W, debug):

    # dynamic region pooling

    birdview_channel = birdview_feat.get_shape().as_list()[3]
    frontview_channel = frontview_feat.get_shape().as_list()[3]
    rgbview_channel = rgbview_feat.get_shape().as_list()[3]
    birdview_region_list = [] 
    frontview_region_list = []
    rgbview_region_list = []

    birdview_pooling_ROI = tf.image.crop_and_resize(birdview_feat, birdview_rois, birdview_rois_ind, [ROI_H, ROI_W], name = 'birdview_pooling_ROI')
    frontview_pooling_ROI = tf.image.crop_and_resize(frontview_feat, frontview_rois, frontview_rois_ind, [ROI_H, ROI_W], name = 'frontview_pooling_ROI')
    rgbview_pooling_ROI = tf.image.crop_and_resize(rgbview_feat, rgbview_rois, rgbview_rois_ind, [ROI_H, ROI_W], name = 'rgbview_pooling_ROI')
    
    if debug: 
        utils.add_activation_summary(birdview_pooling_ROI)
        utils.add_activation_summary(frontview_pooling_ROI)
        utils.add_activation_summary(rgbview_pooling_ROI)
    

    return birdview_pooling_ROI, frontview_pooling_ROI, rgbview_pooling_ROI

# fusion network
def region_fusion_net(birdview_region, frontview_region, rgbview_region, NUM_OF_REGRESSION_VALUE, ROI_H, ROI_W):
    # flat

    birdview_flatregion = tf.reshape(birdview_region, [-1, ROI_W * ROI_H * 256], name = 'birdview_flatregion')
    frontview_flatregion = tf.reshape(frontview_region, [-1, ROI_W * ROI_H * 256], name = 'frontview_flatregion')
    rgbview_flatregion = tf.reshape(rgbview_region, [-1, ROI_W * ROI_H * 256], name = 'rgbview_flatregion')

    with tf.name_scope("fusion-1"):
        # first fusion
        # feature transformation is implemented by fully connected netwok

        joint_1 = utils.join(birdview_flatregion, frontview_flatregion, rgbview_flatregion, name = 'joint_1')
        fusion_birdview_1 = utils.fully_connected(joint_1, 1024, name = 'fusion_birdview_1') 
        fusion_frontview_1 = utils.fully_connected(joint_1, 1024, name = 'fusion_frontview_1')
        fusion_rgbview_1 = utils.fully_connected(joint_1, 1024, name ='fusion_rgbview_1')

    with tf.name_scope("fusion-2"):
        # second fusion
        joint_2 = utils.join(fusion_birdview_1, fusion_frontview_1, fusion_rgbview_1, name = 'joint_2')
        fusion_birdview_2 = utils.fully_connected(joint_2, 1024, name ='fusion_birdview_2')
        fusion_frontview_2 = utils.fully_connected(joint_2, 1024,name = 'fusion_frontview_2')
        fusion_rgbview_2 = utils.fully_connected(joint_2, 1024, name = 'fusion_rgbview_2')

    with tf.name_scope("fusion-3"):
        # third fusion
        joint_3 = utils.join(fusion_birdview_2, fusion_frontview_2, fusion_rgbview_2, 'joint_3')
        fusion_birdview_3 = utils.fully_connected(joint_3, 1024, name ='fusion_birdview_3')
        fusion_frontview_3 = utils.fully_connected(joint_3, 1024,name = 'fusion_frontview_3')
        fusion_rgbview_3 = utils.fully_connected(joint_3, 1024,name = 'fusion_rgbview_3')
     

    with tf.name_scope("fusion-4"):
        joint_4 = utils.join(fusion_birdview_3, fusion_frontview_3, fusion_rgbview_3, name ='joint_4')
        #pdb.set_trace()
        #joint_4= utils.join(birdview_flatregion, frontview_flatregion, rgbview_flatregion, name = 'joint_1')
        logits_cls  = utils.fully_connected(joint_4, 2, name = 'fusion_cls_4', relu = False)
        logits_reg = utils.fully_connected(joint_4, NUM_OF_REGRESSION_VALUE, name = 'fusion_reg_4', relu = False)


    return logits_cls, logits_reg


# l2 loss for regression
def l2_loss(s, t):
    """ L2 loss function. """
    d = s - t
    x = d * d
    loss = tf.reduce_sum(x, 1)
    return loss

# MV3D network comprised of 3D proposal network and region fusion networkj
# def MV3D(birdview, frontview, rgbview, birdview_proposals, frontview_proposals, rgbview_proposals, proposals_mask, gt_ROI_labels, gt_ROI_regs, 
# 	     birdview_rois, frontview_rois, rgbview_rois, ROI_H, ROI_W, model_dir, debug):
def MV3D(birdview, frontview, rgbview, cls_mask, reg_mask, gt_ROI_labels, gt_ROI_regs, 
         birdview_rois, frontview_rois, rgbview_rois, birdview_box_ind, frontview_box_ind, 
         rgbview_box_ind, ROI_H, ROI_W, NUM_OF_REGRESSION_VALUE, model_dir, weight, reg_weight, debug, keep_prob = 1.0):
    
    MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-16.mat'
    
    
    with tf.name_scope("3D-Proposal-net"):
        birdview_net, frontview_net, rgbview_net = Proposal_net(birdview, frontview, rgbview, model_dir, MODEL_URL, debug, keep_prob)
        # accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(all_rpn_logits, dimension=1), tf.reshape(gt_anchor_labels, [-1])), tf.float32))

    with tf.name_scope("ROI-pooling"):
        # birdview_pooling_ROI, frontview_pooling_ROI, rgbview_pooling_ROI = region_pooling(birdview_net['birdview_relu4_3'], frontview_net['frontview_relu4_3'], rgbview_net['rgb_relu4_3'], 
        #                                                                                   birdview_proposals, frontview_proposals, rgbview_proposals, birdview_rois, frontview_rois, 
        #                                                                                   rgbview_rois, ROI_H, ROI_W, debug)
        birdview_pooling_ROI, frontview_pooling_ROI, rgbview_pooling_ROI = region_pooling(birdview_net['birdview_upsample'], frontview_net['frontview_upsample'], rgbview_net['rgb_upsample'], 
                                                                                          birdview_rois, frontview_rois, rgbview_rois, birdview_box_ind, frontview_box_ind, rgbview_box_ind, 
                                                                                          ROI_H, ROI_W, debug)
    # pdb.set_trace()
    with tf.name_scope("region-fusion-net"):
        logits_cls, logits_reg = region_fusion_net(birdview_pooling_ROI, frontview_pooling_ROI, rgbview_pooling_ROI, NUM_OF_REGRESSION_VALUE, ROI_H, ROI_W)
    # pdb.set_trace()
    with tf.name_scope("loss"):
        
        gt_ROI_regs = tf.reshape(gt_ROI_regs, [-1, NUM_OF_REGRESSION_VALUE])
        gt_ROI_labels = tf.reshape(gt_ROI_labels, [-1])
        regression_loss = tf.reduce_sum(l2_loss(logits_reg, gt_ROI_regs) * tf.cast(reg_mask, tf.float32)) / tf.cast(tf.reduce_sum(reg_mask), tf.float32)
        
        classification_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = gt_ROI_labels, logits = logits_cls) * tf.cast(cls_mask, tf.float32)) / tf.cast(tf.reduce_sum(cls_mask), tf.float32)
        # regularization
        trainable_var = tf.trainable_variables()
        weight_decay = 0
        for var in trainable_var:
            weight_decay = weight_decay + tf.nn.l2_loss(var)  

        loss = regression_loss  + weight * classification_loss + reg_weight * weight_decay

    return loss, classification_loss, regression_loss, logits_cls, logits_reg

# Birdview proposal network (3D RPN) 
def birdview_proposal_net(birdview, gt_anchor_labels, gt_anchor_regs, anchor_cls_masks, anchor_reg_masks, weight, reg_weight, model_dir, batch_size, debug, keep_prob = 1.0):
    #""" 3D region proposal network """
    # input birdview, dropout probability, weight of vgg, ground-truth labels, ground-truth regression value, 
    # anchor classification mask and anchor regression mask
    # The birdview has more than three channel, thus we need to train first two conv layers in vgg-16
    # Miscellaneous definition
    MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-16.mat'
    NUM_OF_REGRESSION_VALUE = 6
    NUM_OF_ANCHOR = 4
    FCN_KERNEL_SIZE = 3

    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(model_dir, MODEL_URL)
    weights = np.squeeze(model_data['layers'])    

    # store output from classfication layer and regression layer
    all_rpn_logits = []
    all_rpn_regs = []
    current = birdview


    # vgg
    with tf.name_scope("Vgg-16"):
        net = vgg_net_birdview(weights, birdview, debug, keep_prob)
        current = net["birdview_relu4_3"]
    # upsample, output 256 channels
    with tf.name_scope("Upsample_layer"):
        kernels = utils.weight_variable([3, 3, 256, 256], name= "upsample_w")
        bias = utils.bias_variable([256], name="upsample_b")
        net['upsample'] = utils.conv2d_transpose_strided(current, kernels, bias, name = 'upsample', keep_prob = keep_prob)
        current = net['upsample']
        if debug: 
            utils.add_activation_summary(current)
    with tf.name_scope("Fully_conv_layer"):
    # Fully convolution layer of 3D proposal network. Similar to the last layer of Region Prosal Network.
        
        for j in range(NUM_OF_ANCHOR):
            kernels_cls = utils.weight_variable([FCN_KERNEL_SIZE, FCN_KERNEL_SIZE, 256, 2], name= "FCN_cls_w" + str(j))
            kernels_reg = utils.weight_variable([FCN_KERNEL_SIZE, FCN_KERNEL_SIZE, 256, NUM_OF_REGRESSION_VALUE], name= "FCN_reg_w" + str(j))
            bias_cls = utils.bias_variable([2], name="FCN_cls_b" + str(j))
            bias_reg = utils.bias_variable([6], name="FCN_reg_b"+ str(j))
            rpn_logits = utils.conv2d_basic(current, kernels_cls, bias_cls)
            rpn_regs = utils.conv2d_basic(current, kernels_reg, bias_reg)
            net["FCN_cls_" + str(j)] = rpn_logits
            net["FCN_reg_" + str(j)] = rpn_regs
            if debug:
                utils.add_activation_summary(rpn_logits)

            rpn_logits = tf.reshape(rpn_logits, [batch_size, -1, 2])
            all_rpn_logits.append(rpn_logits)
            # Values required clip might be different
            # rpn_regs = tf.clip_by_value(rpn_regs, -0.2, 0.2)
            rpn_regs = tf.reshape(rpn_regs, [batch_size, -1, NUM_OF_REGRESSION_VALUE])
            all_rpn_regs.append(rpn_regs)
            
    with tf.name_scope("Cls_and_reg_loss"):
        
        all_rpn_logits = tf.concat(all_rpn_logits, 1)
        all_rpn_regs = tf.concat(all_rpn_regs, 1)
        # pdb.set_trace()
        all_rpn_logits = tf.reshape(all_rpn_logits, [-1, 2])
        all_rpn_logits_softmax = tf.nn.softmax(all_rpn_logits, dim = -1)
        all_rpn_regs = tf.reshape(all_rpn_regs, [-1, NUM_OF_REGRESSION_VALUE])
        
        # Compute the loss function
        gt_anchor_labels = tf.reshape(gt_anchor_labels, [-1])       
        gt_anchor_regs = tf.reshape(gt_anchor_regs, [-1, NUM_OF_REGRESSION_VALUE])
        anchor_cls_masks = tf.reshape(anchor_cls_masks, [-1])
        anchor_reg_masks = tf.reshape(anchor_reg_masks, [-1])

        # Classification loss
        classification_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = gt_anchor_labels, logits = all_rpn_logits) * anchor_cls_masks
        classification_loss = tf.reduce_sum(classification_loss) / tf.maximum(tf.reduce_sum(anchor_cls_masks), 1)

        #regression loss
        regression_loss = tf.reduce_sum(l2_loss(all_rpn_regs, gt_anchor_regs) * anchor_reg_masks) / tf.maximum(tf.reduce_sum(anchor_reg_masks), 1)

        # regularization
        trainable_var = tf.trainable_variables()
        weight_decay = 0
        for var in trainable_var:
            weight_decay = weight_decay + tf.nn.l2_loss(var)

        Overall_loss = weight * classification_loss + regression_loss + reg_weight * weight_decay

    
    return net, classification_loss, regression_loss, Overall_loss, all_rpn_logits_softmax, all_rpn_regs