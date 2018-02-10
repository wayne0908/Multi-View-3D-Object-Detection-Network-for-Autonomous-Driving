# Written by Weizhi Li
# 3D proposal network 
# You need to load the prepared data to train the network.

# training_data: a list containing training data
# training_data[0] = birdview
# training_data[1] = anchor_label
# training_data[2] = anchor_reg
# training_data[3] = anchor_label_mask
# training_data[4] = anchor_reg_mask

# validation_data has same structure of training_data

# The format of prepared data should be:

# birdview: Birdview generated from LIDAR point cloud. Format: .npy file. [number of images, row of a image, col of a image, channels]

# anchor_label: Anchor labels corresponding to birdview. Format: .npy file. [number of images, row of a image, col of a image, number of anchors]
# The values in anchor_label are either be one or zero. Being ones means the anchor are postive and otherwise are negative or 
# invalid.

# anchor_label_mask: Mask for valid anchors of label. Format: .npy file. [number of images, row of a image, col of a image, number of anchors]
# The values in anchor_label_mask are either be one or zero. Being one means the corresponding anchors of label are counted in training otherwise
# are not

# anchor_reg: Anchor regression value corresponding  to birdview. 
# Format: .npy file. [number of images, row of a image, col of a image, number of anchors, number of regression values]
# The values in anchor_reg are the regression values of each anchor.

# anchor_reg_mask: Mask for valid anchors of regression. Format: .npy file. [number of images, row of a image, col of a image, number of anchors]
# The values in anchor_reg_mask are either be one or zero. Being one means the corresponding anchors of rehression are counted in training otherwise
# are not

# MAX_EPOCH: training epoches. weight: the weight of classification loss. 
# reg_weight: the weight of l2 regularization. keep_prob: dropout probability.

import birdview_proposal_net
import numpy as np
import pdb
from os.path import expanduser


training_data = [] 
validation_data = []

loadroot = expanduser("~") + '/Desktop/MV3D/DEMO/TEST_DATA/Proposal-net/'

birdview = np.load(loadroot + 'birdview_set.npy')
anchor_label = np.load(loadroot + 'cls_label.npy')
anchor_label_mask = np.load(loadroot + 'anchors_cls_mask_set.npy')
anchor_reg = np.load(loadroot + 'reg_label.npy')
anchor_reg_mask = np.load(loadroot + 'anchors_reg_mask_set.npy')

training_data.extend((birdview, anchor_label, anchor_reg, anchor_label_mask, anchor_reg_mask))
validation_data.extend((birdview, anchor_label, anchor_reg, anchor_label_mask, anchor_reg_mask))

MAX_EPOCH = int(10e3)
weight = 1.0
reg_weight = 0.1
keep_prob = 0.7

# label, reg = birdview_proposal_net.prediction_network(birdview)

birdview_proposal_net.train_network(training_data, validation_data, MAX_EPOCH, weight, reg_weight, keep_prob)



