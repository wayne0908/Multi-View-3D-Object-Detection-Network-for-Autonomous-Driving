# Written by Weizhi Li
# Prediction of 3D proposal network 
# birdview: Birdview generated from LIDAR point cloud. Format: .npy file. [number of images, row of a image, col of a image, channels]


import birdview_proposal_net
import numpy as np
import pdb
from os.path import expanduser


training_data = [] 
validation_data = []

loadroot = expanduser("~") + '/Desktop/MV3D/DEMO/TEST_DATA/Proposal-net/'

birdview = np.load(loadroot + 'birdview_set.npy')

label, reg = birdview_proposal_net.prediction_network(birdview)





