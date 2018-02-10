# General description 
Reproduce a network described by the [1].

According to [1], training are divided into two stages:

1. Run train_proposal.py to train the proposal network.
The network is to generate proposal candidates. Aftering training,
run prediction_proposal.py to generate proposal candidates.

2. Run train_fusion.py to train the fusion network.
The network is to generate final results. Aftering training,
run prediction_fusion.py to generate final results.

# Training data for proposal network (train_proposal.py):

training_data: a list of  data

training_data[0] = birdview

training_data[1] = anchor_label

training_data[2] = anchor_reg

training_data[3] = anchor_label_mask

training_data[4] = anchor_reg_mask

validation_data has same structure as training_data.

The format of prepared data should be:

birdview: Birdview generated from LIDAR point cloud. Format: .npy file. [number of images, row of a image, col of a image, channels]

anchor_label: Anchor labels corresponding to birdview. Format: .npy file. [number of images, row of a image, col of a image, number of anchors]. The values in anchor_label are either be one or zero. Being ones means the anchor are postive and otherwise are negative or 
invalid.

anchor_label_mask: Mask for valid anchors of label. Format: .npy file. [number of images, row of a image, col of a image, number of anchors]. The values in anchor_label_mask are either be one or zero. Being one means the corresponding anchors of label are counted in training otherwise are not.

anchor_reg: Anchor regression value corresponding to birdview. Format: .npy file. [number of images, row of a image, col of a image, number of anchors, number of regression values]. The values in anchor_reg are the regression values of each anchor.

anchor_reg_mask: Mask for valid anchors of regression. Format: .npy file. [number of images, row of a image, col of a image, number of anchors]. The values in anchor_reg_mask are either be one or zero. Being one means the corresponding anchors of rehression are counted in training otherwise are not

# Training data for fusion network (train_fusion.py):

training_data: a list containing training data

training_data[0] = data_birdview

training_data[1] = data_frontview

training_data[2] = data_rgbview

training_data[3] = data_birdview_rois

training_data[4] = data_frontview_rois

training_data[5] = data_rgbview_rois

training_data[6] = data_birdview_box_ind

training_data[7] = data_frontview_box_ind

training_data[8] = data_rgbview_box_ind

training_data[9] = data_ROI_labels

training_data[10] = data_ROI_regs

training_data[11] = data_cls_mask

training_data[12] = data_reg_mask

Validation_data has same structure as training_data.

The format of prepared data should be:

data_birdview: Birdview generated from LIDAR point cloud. Format: .npy file. [number of images, row of a image, col of a image, channels]

data_frontview: Frontview generated from LIDAR point cloud. Format: .npy file. [number of images, row of a image, col of a image, channels]

data_rgbview: Rgbview generated from rgb image. Format: .npy file. [number of images, row of a image, col of a image, channels]

data_birdview_rois: Region coordinate for birdview region pooling. Format: .npy file. [number of images * 200 regions, 4 (y1, x1, y2, x2)] 

data_frontview_rois: Region coordinate for frontview region pooling. Format: .npy file. [number of images * 200 regions, 4 (y1, x1, y2, x2)] 

data_rgbview_rois: Region coordinate for rgbview region pooling. Format: .npy file. [number of images * 200 regions, 4 (y1, x1, y2, x2)] 

data_birdview_box_ind: The value of data_birdview_box_ind[i] specifies the birdview that the i-th box refers to. Format: .npy file. [number of images * 200 regions] 

data_frontview_box_ind: The value of data_birdview_box_ind[i] specifies the frontview that the i-th box refers to. Format: .npy file. [number of images * 200 regions]  

data_rgbview_box_ind: The value of data_birdview_box_ind[i] specifies the rgbview that the i-th box refers to. Format: .npy file. [number of images * 200 regions]   

data_ROI_labels: Region labels. Format: .npy file. [number of images, 200 regions]. The values in data_ROI_labels are either be one or zero. Being ones means the region are postive otherwise are negative or 
invalid.

# Reference

[1] Chen, Xiaozhi, et al. "Multi-view 3d object detection network for autonomous driving." IEEE CVPR. Vol. 1. No. 2. 2017.
