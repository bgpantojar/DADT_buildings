#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 10:46:06 2020

@author: pantoja
"""
import sys
sys.path.append("../src")
from openings_projector import op_projector, camera_matrices, load_intrinsics_poses, load_sfm_json
from opening_detector import main_op_detector
from crack_projector import crack_projector
import warnings
warnings.filterwarnings("ignore")


##################################USER INTERACTION#############################

#USER INPUT
data_folder = 'p4_DADT_00_La_capite'
bound_box = 'region' 
op_det_nn = 'unet' 
images_path = '../data/' + data_folder + '/images/'
cameras_path = '../data/' + data_folder + '/cameras/'
keypoints_path = '../data/' + data_folder + '/keypoints/'
polyfit_path = '../data/' + data_folder + '/polyfit/'
sfm_json_path = '../data/' + data_folder + '/sfm/'
how2get_kp = 0
im1 = ('DJI_0323',)

#USER CALLING FUNCTIONS
#Detector
if how2get_kp == 0 or how2get_kp == 4:
    opening_4points = opening_4points = main_op_detector(data_folder, images_path, bound_box, op_det_nn, cte_fil4=.1)
else:
    opening_4points = None

#Projector
intrinsic, poses, structure = load_sfm_json(sfm_json_path)
DADT = op_projector(data_folder, images_path, keypoints_path, polyfit_path, intrinsic, poses, structure, how2get_kp, im1, opening_4points, ctes=[.0, .2, .2, .3], ray_int_method="casting")
crack_projector(DADT, data_folder, images_path, keypoints_path, intrinsic, poses, structure, im1, opening_4points, ray_int_method="casting", compute_kinematics=True,cracks2textured=True, extra_features=["graffiti",])
print("Finished Opening Projection")