# Written by Weizhi Li
# region based fusion network 
# You need to load the prepared data to train the network.

# training_data: a list containing training data
# training_data[0] = data_birdview
# training_data[1] = data_frontview
# training_data[2] = data_rgbview
# training_data[3] = data_birdview_rois
# training_data[4] = data_frontview_rois
# training_data[5] = data_rgbview_rois
# training_data[6] = data_birdview_box_ind
# training_data[7] = data_frontview_box_ind
# training_data[8] = data_rgbview_box_ind
# training_data[9] = data_ROI_labels
# training_data[10] = data_ROI_regs
# training_data[11] = data_cls_mask
# training_data[12] = data_reg_mask

# validation_data has same structure of training_data

# The format of prepared data should be:

# data_birdview: Birdview generated from LIDAR point cloud. Format: .npy file. [number of images, row of a image, col of a image, channels]

# data_frontview: Frontview generated from LIDAR point cloud. Format: .npy file. [number of images, row of a image, col of a image, channels]

# data_rgbview: Rgbview generated from rgb image. Format: .npy file. [number of images, row of a image, col of a image, channels]

# data_birdview_rois: Region coordinate for birdview region pooling. Format: .npy file. [number of images * 200 regions, 4 (y1, x1, y2, x2)] 

# data_frontview_rois: Region coordinate for frontview region pooling. Format: .npy file. [number of images * 200 regions, 4 (y1, x1, y2, x2)] 

# data_rgbview_rois: Region coordinate for rgbview region pooling. Format: .npy file. [number of images * 200 regions, 4 (y1, x1, y2, x2)] 

# data_birdview_box_ind: The value of data_birdview_box_ind[i] specifies the birdview that the i-th box refers to. Format: .npy file. [number of images * 200 regions] 

# data_frontview_box_ind: The value of data_birdview_box_ind[i] specifies the frontview that the i-th box refers to. Format: .npy file. [number of images * 200 regions]  

# data_rgbview_box_ind: The value of data_birdview_box_ind[i] specifies the rgbview that the i-th box refers to. Format: .npy file. [number of images * 200 regions]   

# data_ROI_labels: Region labels. Format: .npy file. [number of images, 200 regions]
# The values in data_ROI_labels are either be one or zero. Being ones means the region are postive otherwise are negative or 
# invalid.

# data_ROI_regs: Region regression value. 
# Format: .npy file. [number of images, row of a image, col of a image, number of anchors, number of regression values]
# The values in data_ROI_regs are the regression values of each region.

# data_cls_mask: Mask for valid regions of label. Format: .npy file. [number of images * 200 regions, 1]
# The values in data_cls_mask are either be one or zero. Being one means the corresponding regions of label are counted in training otherwise
# are not

# data_reg_mask: Mask for valid regions of regression. Format: .npy file. [number of images * 200 regions, 1]
# The values in data_reg_mask are either be one or zero. Being one means the corresponding regions of regression are counted in training otherwise
# are not


# MAX_EPOCH: training epoches. weight: the weight of classification loss. 
# reg_weight: the weight of l2 regularization. keep_prob: dropout probability.

import fusion_net
import numpy as np
import pdb
from os.path import expanduser


train_data = [] 
val_data = []
MAX_EPOCH = int(10e3)
weight = 1.0
reg_weight = 0.1
keep_prob = 0.7

loadroot = expanduser("~") + '/Desktop/MV3D/DEMO/TEST_DATA/Fusion-net/'

data_birdview = np.load(loadroot + 'birdview_set.npy')
data_frontview = np.load(loadroot + 'frontview_set.npy')
data_rgbview = np.load(loadroot + 'rgbview_set.npy')

data_frontview_rois = np.load(loadroot + 'frontview_rois2.npy')
data_rgbview_rois = np.load(loadroot + 'rgbview_rois2.npy')
data_birdview_rois = np.load(loadroot + 'birdview_rois2.npy')

data_frontview_box_ind = np.zeros(len(data_frontview_rois))
data_rgbview_box_ind = np.zeros(len(data_frontview_rois))
data_birdview_box_ind = np.zeros(len(data_frontview_rois))

data_cls_mask = np.load(loadroot + 'cls_mask.npy')
data_reg_mask = np.load(loadroot + 'reg_mask.npy')
data_ROI_labels = np.load(loadroot + 'gt_ROI_labels.npy')
data_ROI_regs = np.load(loadroot + 'gt_ROI_regs.npy')

train_data.extend((data_birdview, data_frontview, data_rgbview, data_birdview_rois, data_frontview_rois, 
					  data_rgbview_rois, data_birdview_box_ind, data_frontview_box_ind, data_rgbview_box_ind,
					  data_ROI_labels, data_ROI_regs, data_cls_mask, data_reg_mask))

val_data.extend((data_birdview, data_frontview, data_rgbview, data_birdview_rois, data_frontview_rois, 
					  data_rgbview_rois, data_birdview_box_ind, data_frontview_box_ind, data_rgbview_box_ind,
					  data_ROI_labels, data_ROI_regs, data_cls_mask, data_reg_mask))

# label, reg = MV3D.prediction_network(train_data)
# pdb.set_trace()
fusion_net.train_network(train_data, val_data, MAX_EPOCH, weight, reg_weight, keep_prob)