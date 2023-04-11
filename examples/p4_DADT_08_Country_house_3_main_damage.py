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
import time


##################################USER INTERACTION#############################
t0 = time.time()
###############USER INPUT
data_folder = 'p4_DT_LOD_08_Country_house_3'
bound_box = 'region' #region: using bounding box. poly: using polygoning
op_det_nn = 'unet' #ssd
images_path = '../data/' + data_folder + '/images/'
cameras_path = '../data/' + data_folder + '/cameras/'
keypoints_path = '../data/' + data_folder + '/keypoints/'
polyfit_path = '../data/' + data_folder + '/polyfit/'
sfm_json_path = '../data/' + data_folder + '/sfm/'
how2get_kp = 0 #it can be 0: using detectors, 1: loading npy arrrays, 2: by clicking in the image, 3:im1:numpy arrays, im2:homography, 4: LKT
im1 = ('frame000450','frame002400','frame003330','frame004650')

#############USER CALLING FUNCTIONS

#Detector
#if how2get_kp == 0 or how2get_kp == 4:
#    opening_4points = opening_4points = main_op_detector(data_folder, images_path, bound_box, op_det_nn, cte_fil4=.05)
#else:
#    opening_4points = None

#############USER CALLING FUNCTIONS
#Detector
if how2get_kp == 0 or how2get_kp == 4:
    opening_4points = main_op_detector(data_folder, images_path, bound_box, op_det_nn, cte_fil4=.1)
else:
    opening_4points = None
t1 = time.time() - t0
print("Time segmenting openings {}s".format(t1))
#Projector
intrinsic, poses, structure = load_sfm_json(sfm_json_path)
t2 = time.time() - t1 - t0
print("Time reading sfm information {}s".format(t2))
DADT = op_projector(data_folder, images_path, keypoints_path, polyfit_path, intrinsic, poses, structure, how2get_kp, im1, opening_4points, ctes=[.0, .3, .3, .3], ray_int_method="casting")
t3 = time.time() - t2 - t1 - t0
print("Time projecting openings to 3D {}s".format(t3))
crack_projector(DADT, data_folder, images_path, keypoints_path, intrinsic, poses, structure, im1, opening_4points, ray_int_method="casting", compute_kinematics=False, cracks2textured=True)
t4 = time.time() - t3 - t2 - t1 - t0
print("Time segmenting, characterizing and projecting cracks {}s".format(t4))
print("Finished DADT in {}s".format(time.time()-t0))

