# Written by Weizhi Li
# prediction of region based fusion network 

# data: a list of data
# data[0] = data_birdview
# data[1] = data_frontview
# data[2] = data_rgbview
# data[3] = data_birdview_rois
# data[4] = data_frontview_rois
# data[5] = data_rgbview_rois
# data[6] = data_birdview_box_ind
# data[7] = data_frontview_box_ind
# data[8] = data_rgbview_box_ind


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

import fusion_net
import numpy as np
import pdb
from os.path import expanduser


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

data.extend((data_birdview, data_frontview, data_rgbview, data_birdview_rois, data_frontview_rois, 
		     data_rgbview_rois, data_birdview_box_ind, data_frontview_box_ind, data_rgbview_box_ind))

label, reg = fusion_net.prediction_network(data)
