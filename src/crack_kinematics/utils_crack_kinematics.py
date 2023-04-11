#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 15:53:13 2022

@author: pantoja
"""

from least_square_crack_kinematics import find_crack_kinematics, kinematics_full_edges
from tools_crack_kinematic import crop_patch, compute_kinematic_mean, get_tn_tt_from_results
import numpy as np
import matplotlib.pyplot as plt
import pylab
import cv2

def find_kinematics_patch_finite(data_path, mask_name, sk_pt=None, size=(256,256), mmpx=None, k_neighboors=50, m=1., l=4, k_n_normal_feature=10, omega=0., edges=False, normals = False, make_plots_local=True, make_plots_global = False, eta = 1., window_overlap=None):
    """
    Computes the kinematics for a image patch extracted from a full image of a crack pattern using finite edges approach.
    Prints also the value for the clicked point.

    Args:
        data_path (str): data folder path
        mask_name (str): image mask name of the crack pattern
        sk_pt (array, optional): x,y coordinate where the patch is centered. Defaults to None.
        size (tuple, optional): size of patch to be extracted. Defaults to (256,256).
        mmpx (float, optional): resolution as mm/px ratio. Defaults to None.
        k_neighboors (int, optional): value of k_neigh hyperparameter. Defaults to 50.
        m (_type_, optional): value of m hyperparameter. Defaults to 1..
        l (int, optional): value of l hyperparameter. Defaults to 4.
        k_n_normal_feature (int, optional): value of kn hyperparameter. Defaults to 10.
        omega (_type_, optional): value of omega hyperparameter. Defaults to 0..
        edges (bool, optional): True if edges are given as npy arrays in the data folder. Defaults to False.
        normals (bool, optional): True if parcel of loss function with normal features is taken into accout. Defaults to False.
        make_plots_local (bool, optional): True to plot results with local coordinates. Defaults to True.
        make_plots_global (bool, optional): True to plot results with global coordinates. Defaults to False.
        eta (_type_, optional): value of eta hyperparameter. Defaults to 1.
    """

    patch_name, sk_pt = crop_patch(data_path, mask_name, sk_pt = sk_pt, size=size, return_patch=False)
    crack_kinematic_dic  = find_crack_kinematics(data_path, patch_name, k_neighboors=k_neighboors, m=m, l=l, k_n_normal_feature=k_n_normal_feature, omega=omega, edges=edges, normals = normals, make_plots_local=make_plots_local, make_plots_global = make_plots_global, eta =eta, window_overlap=window_overlap)
    _ = compute_kinematic_mean(data_path, patch_name, [1.,1.,4,10,0.], crack_kinematic=None, full_edges=False, local_trans = True, abs_disp = True, use_eta = True)


    #Kinematics value for clicked point
    im_name = patch_name.split('.')[0]
    data_folder = data_path.split('/')[-2] + '/'
    hyper = [eta,m,l,k_n_normal_feature,omega]
    approach = "finite_edges"
    #Select sk_pt in patch as the middle points of the patch
    sk_pt_patch = np.array([int(size[1]/2), int(size[0]/2)]).reshape((1,2))
    tn_tt_read,_ = get_tn_tt_from_results(im_name, data_folder, sk_pt_patch, approach, hyper)
    if mmpx is not None:
        print("result in mm is ", np.array(tn_tt_read)*mmpx)


def find_kinematics_patch_full_edge(data_path, mask_name, sk_pt = None, size=(256,256), mmpx=None, k_n_normal_feature=10, omega = 0., edges=True, normals=False, make_plots_local=True, make_plots_global = True):
    """ 
        Computes the kinematics for a image patch extracted from a full image of a crack pattern using full edges approach..
        Prints also the value for the clicked point.

        Args:
            data_path (str): data folder path
            mask_name (str): image mask name of the crack pattern
            sk_pt (array, optional): x,y coordinate where the patch is centered. Defaults to None.
            size (tuple, optional): size of patch to be extracted. Defaults to (256,256).
            mmpx (float, optional): resolution as mm/px ratio. Defaults to None.
            k_n_normal_feature (int, optional): value of kn hyperparameter. Defaults to 10.
            omega (_type_, optional): value of omega hyperparameter. Defaults to 0..
            edges (bool, optional): True if edges are given as npy arrays in the data folder. Defaults to False.
            normals (bool, optional): True if parcel of loss function with normal features is taken into accout. Defaults to False.
            make_plots_local (bool, optional): True to plot results with local coordinates. Defaults to True.
            make_plots_global (bool, optional): True to plot results with global coordinates. Defaults to False.
    """
    
    patch_name, sk_pt = crop_patch(data_path, mask_name, sk_pt=sk_pt, size=size, return_patch=False)
    crack_kinematic_full_edge = kinematics_full_edges(data_path, patch_name, k_n_normal_feature=k_n_normal_feature, omega = omega, edges=edges, normals=normals, make_plots_global = make_plots_global)
    _ = compute_kinematic_mean(data_path, patch_name, [10,0.], crack_kinematic=None, full_edges=True, local_trans = False, abs_disp = False)

    #Kinematics value for clicked point
    im_name = patch_name.split('.')[0]
    data_folder = data_path.split('/')[-2] + '/'
    hyper = [k_n_normal_feature,omega]
    approach = "full_edges"
    #Select sk_pt
    sk_pt_patch = np.array([int(size[1]/2), int(size[0]/2)]).reshape((1,2))
    tn_tt_read,_ = get_tn_tt_from_results(im_name, data_folder, sk_pt_patch, approach, hyper)
    if mmpx is not None:
        print("result in mm is ", np.array(tn_tt_read)*mmpx)

def get_tn_tt(data_path, mask_name, approach, hyper, mmpx=None):
    """_summary_

    Args:
        data_path (str): data folder path
        mask_name (str): image mask name of the crack pattern
        approach (str): approached used to compute kinematics. :finite_edges or full_edges
        hyper (list): list of hyperparameters used. hyper = [k_n_normal_feature,omega] for full edges or yper = [eta,m,l,k_n_normal_feature,omega] for finite edges
        mmpx (float, optional): value of resolution ratio as mm/px. Defaults to None.
    """

    im_name = mask_name.split('.')[0]
    data_folder = data_path.split('/')[-2] + '/'
    #Select sk_pt
    img = cv2.imread(data_path+mask_name)
    plt.figure()
    plt.imshow(img, 'gray')
    print('Please click 1 points for skeleton')
    sk_pt = np.array(pylab.ginput(1,200))
    print('you clicked:', sk_pt)    
    plt.close()
    sk_pt = sk_pt[0]
    np.save('../results/'+data_folder+'sk_pt_'+mask_name+'.npy', sk_pt)

    tn_tt_read,_ = get_tn_tt_from_results(im_name, data_folder, sk_pt, approach, hyper)
    if mmpx is not None:
        print("result in mm is ", np.array(tn_tt_read)*mmpx)