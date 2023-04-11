#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 14:57:07 2020

This script contains the main function used to compute crack kinematics.
The first, kinematics_full_edges() computes the normal and tangential
crack displacements using entire oposite edges of the crack (used when there is
a single crack). The second, find_crack_kinematics() computes the displacements
using finite segments methodology.

The codes are based on "Determi"Determing crack kinematics from imaged crack patterns" 
by Pantoja-Rosero et., al.

@author: pantoja
"""

from __future__ import print_function
from traceback import print_last
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
from skimage import measure
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
from sklearn.neighbors import NearestNeighbors
from crack_kinematics.endpoint_branchpoint import find_end_points
from tqdm import tqdm
import os
import json
from crack_kinematics.tools_crack_kinematic import H_from_transformation, find_dir_nor, find_skeleton_intersections, plot_edges_kinematic,plot_kinematic,plot_two_dofs_kinematic,plot_n_t_kinematic, _getnodes
from crack_kinematics.tools_least_squares import run_crack_adjustment
from crack_kinematics.optimizer import kinematic_adjustment_lambda_based, kinematic_adjustment_pareto_based
from scipy.signal import argrelextrema
import time
#matplotlib.use('Agg')


def kinematics_full_edges(data_path, mask_name, k_n_normal_feature=10, omega = 2, edges=False, normals=False, make_plots_global = False):
    """
    It computes the crack kinematic when it is given a binary mask that represents the segmentation
    of a crack pattern. This function specifically finds oposite edges of a crack and registers them
    using euclidean transformation and least squares.

    Args:
        data_path (str): Data folder path. Where the binary mask for the crack pattern is located
        mask_name (str): Binary mask name_
        k_n_normal_feature (int, optional): Number of neighboors to compute normal direction of edges. Defaults to 10.
        omega (int, optional): Weight in loss function for parcel that considers normals of edges as features. Defaults to 2.
        edges (bool, optional): True if edges are given as npy files in the data folder. Defaults to False.
        normals (bool, optional): True to consider normal direction of edges as features in the loss function (original paper does not use it). Defaults to False.
        make_plots_global (bool, optional): True to plot solutions. Defaults to False.

    Returns:
        crack_kinematic (dict): Dictionary containing the kinematic values for 3dof, 2dof, and tn-tt for each skeleton pixel of the crack pattern
    """

    print("Running crack kinematics for detected crack: " + mask_name + "-------------")

    #Check if output directory exists, if not, create it
    batch_name = data_path.split('/')[-2]
    dir_save = '../results/' + batch_name + '/' + mask_name[:-4] + '/' + 'full_edges/knnor{}_omega{}/'.format(k_n_normal_feature,omega)
    check_dir = os.path.isdir(dir_save)
    if not check_dir:
        os.makedirs(dir_save)
    
    #Read mask
    mask1 = cv2.imread(data_path+mask_name)
    mask1 = mask1[:,:,0]
    
    #Finding regions in mask to analize them separetaly
    labels_mask = label(mask1)
    regions_mask = regionprops(labels_mask)
    
    #Dictionary with full output for each region detected
    crack_kinematic = {} 
        
    for reg in regions_mask:
        
        #Mask with the crack reagion
        crack = reg.filled_image*255
    
        #Change in coordinates due the crop of the image in the region
        centroid_global = reg.centroid
        centroid_local = reg.local_centroid
        dy,dx = np.array(centroid_local) - np.array(centroid_global)
        
        #Finding mask contours
        if edges:
            #If edges are given as npy files in data folder
            edge0 = np.flip(np.load(data_path+mask_name[:-4]+'_edge0.npy'),1) + np.array([dy,dx])
            edge1 = np.flip(np.load(data_path+mask_name[:-4]+'_edge1.npy'),1) + np.array([dy,dx])
            crack_contour = [edge0,edge1]
            
        else:
            #if edges not given
            #Getting contour points of masks through measure of skimage library
            crack_contour = measure.find_contours(crack,100)
            
            #Delete short contours
            while len(crack_contour)>2:
                len_cont = np.array([len(cc) for cc in crack_contour])
                crack_contour.pop(np.where(len_cont==np.min(len_cont))[0][0])

        
        #Plot in the original mask
        plt.figure()
        plt.imshow(mask1, 'gray')
        plt.plot(crack_contour[1][:,1]-dx, crack_contour[1][:,0]-dy, c=(1.,.2,.0), marker='o')
        plt.plot(crack_contour[0][:,1]-dx, crack_contour[0][:,0]-dy, c=(.0,.5,1.), marker='o')
        plt.savefig(dir_save+mask_name[:-4]+'_full_edges_mask_detected_edges.png', bbox_inches='tight', pad_inches=0)
        plt.savefig(dir_save+mask_name[:-4]+'_full_edges_mask_detected_edges.pdf', bbox_inches='tight', pad_inches=0)
        plt.close()
        
        #Selecting crack edges
        crack_edge_0 = crack_contour[0]
        crack_edge_0[:,[0,1]] = crack_edge_0[:,[1,0]]
        mean_crack_edge_0 = np.mean(crack_edge_0, axis=0)        
        crack_edge_1 = crack_contour[1]
        crack_edge_1[:,[0,1]] = crack_edge_1[:,[1,0]]
                
        #Crack edges in local coordinates
        crack_edge_0 = crack_edge_0 - mean_crack_edge_0
        crack_edge_1 = crack_edge_1 - mean_crack_edge_0    
        
        #Computing normal directions to crack_edge_0 and 1.
        crack_edge_0_normals = (np.pi*find_dir_nor(mask_name, crack_edge_0, k_n=k_n_normal_feature)[:,6]/180).reshape((len(crack_edge_0),-1))
        crack_edge_1_normals = (np.pi*find_dir_nor(mask_name, crack_edge_1, k_n=k_n_normal_feature)[:,6]/180).reshape((len(crack_edge_1),-1))

        #Adding normal directions to crack_edge_0 and 1.
        crack_edge_0 = np.concatenate((crack_edge_0, crack_edge_0_normals), axis=1)
        crack_edge_1 = np.concatenate((crack_edge_1, crack_edge_1_normals), axis=1)
        
        #RUNING LEAST SQUARES to register edges
        H_type = "euclidean"
        H_op, loss = run_crack_adjustment(crack_edge_0, crack_edge_1, omega, H_type=H_type, normals=normals)
        
        #Getting H as 3x3 matrix
        H = H_from_transformation(H_op, H_type)  
        
        #Transforming one edge to overlap over the other
        crack_edge_0_op = np.concatenate((np.copy(crack_edge_0[:,:2]), np.ones((len(crack_edge_0[:,:2]),1))), axis=1).T
        crack_edge_0_op = H @ crack_edge_0_op 
        crack_edge_0_op  /= crack_edge_0_op[2]
        crack_edge_0_op  = crack_edge_0_op[:2].T
        
        #Returning edges to global coordinates
        crack_edge_0 = crack_edge_0[:,:2] + mean_crack_edge_0
        crack_edge_1 = crack_edge_1[:,:2] + mean_crack_edge_0
        crack_edge_0_op = crack_edge_0_op + mean_crack_edge_0
        
        #Ploting: initial edges and transformed edge
        plt.figure()
        plt.imshow(mask1,'gray')
        plt.plot(crack_edge_0[:,0] - dx,crack_edge_0[:,1] - dy, c=(.0,.5,1.), marker='o')
        plt.plot(crack_edge_1[:,0] - dx, crack_edge_1[:,1] - dy, c=(1.,.2,.0), marker='o')
        plt.plot(crack_edge_0_op[:,0] - dx, crack_edge_0_op[:,1] - dy, c=(.1,.5,.1), marker='.')
        plt.savefig(dir_save+mask_name[:-4]+'_full_edges_mask_optimized.png', bbox_inches='tight', pad_inches=0)
        plt.savefig(dir_save+mask_name[:-4]+'_full_edges_mask_optimized.pdf', bbox_inches='tight', pad_inches=0)
        plt.close()
        
    #Append H_params for point x,y in skeleton in the region [(local coodinates), (global coordinates), H_params]
    #Finding skeleton to save in kinematics dictionary and then plot kinematics
    skl = skeletonize(mask1>0, method='lee')
    ind_skl = np.where(skl)        
    
    #Crack inclination to defing crack class (FOR FULL EDGES THIS IS KIND OF DIFFERENT)
    if crack_edge_0[:31][-1][0] > crack_edge_0[:31][0][0]: #to guarantee that vector is pointing to the right
            angle_edge0 = crack_edge_0[:31][-1] - crack_edge_0[:31][0]
    else:
        angle_edge0 = crack_edge_0[:31][0] - crack_edge_0[:31][-1]
    angle_edge0 = np.arctan(angle_edge0[1]/(angle_edge0[0]+1e-15)) #smooth\
    
    #Defining class
    if angle_edge0<0: #Acending, class 1
        cr_class = 1
    else:
        cr_class = 2
    crack_class = []
    
    #List with the H_params for each point in skeleton in the region
    H_params_crack_region = []
    
    #Saving kinematinc constant information 
    for x,y in zip(ind_skl[1],ind_skl[0]):
        H_params_crack_region.append([[float(x),float(y)], [float(x),float(y)], H_op.tolist()])
        crack_class.append([[float(x),float(y)], cr_class])
    
    #Decomposing kinematics in tangetial and normal displacements in relation with skeleton direction
    #Computing skeleton direction
    skeleton = np.array([ind_skl[1],ind_skl[0]]).T
    contour_dir_nor = find_dir_nor(mask_name, skeleton, dir_save=dir_save, mask=mask1, plot=True)
    contour_dir_nor = find_dir_nor(mask_name, skeleton, dir_save=dir_save, mask=None, plot=True)
    
    #Kinematics as horizontal and vertical displacements
    two_dofs_global = [[sk_i.tolist(),sk_i.tolist(), (np.mean(crack_edge_0_op - crack_edge_0[:,:2], axis=0)).tolist()] for sk_i in skeleton]
    
    #Computing kinematics normal tangential for global values
    kinematics_n_t = []#displacement array in normal and tangential directions to the crack plane
    for skl_pt, t_dofs_gl in zip(contour_dir_nor,two_dofs_global):
        #case II -- I and IV quartier
        beta = (skl_pt[6] if skl_pt[6]>0 else skl_pt[6]+180)*np.pi/180 #if the angle is negative, then add 180. This as n axis is pointing always in the I or II quartier.
        transf_matrix_skl_pt = np.array([[np.cos(beta), np.sin(beta)],[-np.sin(beta), np.cos(beta)]])
        t_global = np.array(t_dofs_gl[2]).reshape((2,-1))
        kin_n_t = (transf_matrix_skl_pt @ t_global).reshape(-1)
        kinematics_n_t.append([[skl_pt[0],skl_pt[1]],[skl_pt[0],skl_pt[1]], kin_n_t.tolist()])
    
    #Saving kinematics outputs in dictionary (substracting dx,dy to get global coordinates)
    crack_kinematic[0] =  {}
    crack_kinematic[0]["H_params_cracks"] = H_params_crack_region
    crack_kinematic[0]["two_dofs"] = two_dofs_global
    crack_kinematic[0]["kinematics_n_t"] = kinematics_n_t
    crack_kinematic[0]["dir_nor_skl"] = contour_dir_nor.tolist()
    crack_kinematic[0]["crack_class"] = crack_class
    
    if make_plots_global:
        #Plot kinematics
        plot_kinematic(data_path, mask_name, crack_kinematic, dir_save = dir_save, local_trans=False, full_edges=True)
        #Plot two_dofs kinematic
        plot_two_dofs_kinematic(data_path, mask_name, crack_kinematic, dir_save = dir_save, local_trans = False, full_edges=True)
        #Plot normal and tangential kinematic
        plot_n_t_kinematic(data_path, mask_name, crack_kinematic, dir_save = dir_save, local_trans=False, plot_trans_n=True, plot_trans_t=True, plot_trans_t_n=True, full_edges=True)

    #Save kinematic dictionary as json file if asked
    with open(dir_save+'crack_kinematic.json', 'w') as fp:
        json.dump(crack_kinematic, fp)
    
    return crack_kinematic


def find_crack_kinematics(data_folder, mask_name, k_neighboors=30, m=10, l=5, k_n_normal_feature=10, omega = 2, edges=False, normals=False, make_plots_local=False, make_plots_global = False, ignore_global = True, monte_carlo=False, pareto=False, eta = None, cmap_=None, resolution=None, window_overlap=None, skl=None):
    """ 
    It computes the crack kinematic when it is given a binary mask that represents the segmentation
    of a crack pattern. This function starts detecting crack contours and then using the junction/endpoints 
    of the skeleton to divide the contours into edges. Then for each crack skeleton point, two finite edges
    are found (edge0 and edge1) which are later register to find the kinematics of the crack at that
    skeleton point.     

    Args:
        data_path (str): Data folder path. Where the binary mask for the crack pattern is located
        mask_name (str): Binary mask name
        k_neighboors (int, optional): If eta is not given, this parameter defines the lenght of the edge0. Defaults to 30.
        m (int, optional): Parameter that defines lenght of edge1 proportional to k as m*k. Defaults to 10.
        l (int, optional): Parameter that defines the initial point of the subset of edge1 to register edge0. Defaults to 5.
        k_n_normal_feature (int, optional): Number of neighboors to compute normal direction of edges. Defaults to 10.
        omega (int, optional): Weight in loss function for parcel that considers normals of edges as features. Defaults to 2.
        edges (bool, optional): True if edges are given as npy files in the data folder. Defaults to False.
        normals (bool, optional): True to consider normal direction of edges as features in the loss function (original paper does not use it). Defaults to False.
        make_plots_local (bool, optional): True to plot solutions. Defaults to False.
        make_plots_global (bool, optional): True to plot solutions. Defaults to False.
        ignore_global (bool, optional): True to just compute kinematics based on local coordinates of finite edges. Defaults to True.
        monte_carlo (bool, optional): _True to use the function for montecarlo simulation sampling just one point from the skeleton. Defaults to False.
        pareto (bool, optional): True to use heuristics to find optimal subset of edge1 to which edge1 is register. This instead of using brute force varing l. Defaults to False.
        eta (float, optional): Parameter that defines the lenght of the edge 0 as k=len(contour_segment)/eta. Defaults to None.
        cmap_ (str, optional): Colormap to plot output figures. Defaults to None.
        resolution (float, optional): mm/px resolution ratio to compute output in mm. Defaults to None.
        window_overlap (int, optional): quantity of pixels that a new skeleton point needs to be away to get kinematics computed. If the distance is lower than last skeleton point, the kinematics is copied.

    Returns:
        crack_kinematic (dict): Dictionary containing the kinematic values for 3dof, 2dof, and tn-tt for each skeleton pixel of the crack pattern
    """

    data_path = '../results/' + data_folder + '/'
    
    print("Running crack kinematics for detected crack: " + mask_name + "-------------")
    
    #Check if output directory exists, if not, create it.
    #batch_name = data_path.split('/')[-2]
    if eta is None:
        dir_save = '../results/' + data_folder + '/kinematics/' 
        #dir_save = '../results/' + batch_name + '/' + mask_name[:-4] + '/' + 'finite_edges/kn{}_m{}_l{}_knnor{}_omega{}/'.format(k_neighboors, m, l,k_n_normal_feature,omega)
    else:
        dir_save = '../results/' + data_folder + '/kinematics/' 
        #dir_save = '../results/' + batch_name + '/' + mask_name[:-4] + '/' + 'finite_edges/eta{}_m{}_l{}_knnor{}_omega{}/'.format(eta, m, l,k_n_normal_feature,omega)
    check_dir = os.path.isdir(dir_save)
    if not check_dir:
        os.makedirs(dir_save)       
    
    #Timing process
    initial_time = time.time()
    
    #Read mask
    mask = cv2.imread(data_path+mask_name)
    print(data_path+mask_name)
    mask = (mask[:,:,0]>0)*255
    
    #Dictionary with full output for each region detected (H_params, edge0, edge1, edge0_transformed, skeleton)
    crack_kinematic = {} 
    
    k_neighboors = k_neighboors #selecting the k neighboors to the skeleton point from each edge
    
    #Binary crack pattern
    crack = mask
    
    #Define crack contours. If they are given, skip detection and spliting.
    if edges:
        #Read the edges if they are given in the data folder as npy files
        edge0_given = np.flip(np.load(data_path+mask_name[:-4]+'_edge0.npy'),1)
        edge1_given = np.flip(np.load(data_path+mask_name[:-4]+'_edge1.npy'),1)
        crack_contour = [edge0_given,edge1_given]  
        #plt.figure()
        #plt.imshow(crack, 'gray')        
        #for cr_ct in crack_contour:
        #    plt.scatter(cr_ct[:,1], cr_ct[:,0], c=np.random.rand(1,3))
        crack_edges = [np.flip(edge0_given,1),np.flip(edge1_given,1)] 
        #plt.savefig(dir_save+mask_name[:-4]+'_multiple_edges.png', bbox_inches='tight', pad_inches=0)
        #plt.savefig(dir_save+mask_name[:-4]+'_multiple_edges.pdf', bbox_inches='tight', pad_inches=0)
        #plt.close()
    else:
        #Finding mask/crack contours
        #Getting contour points of masks through measure of skimage library
        crack_contour = measure.find_contours(crack,100)
        
        #Get rid off small detected contours
        crack_contour = [cr_ct for cr_ct in crack_contour if len(cr_ct)>10]
        
        #Dividing cdetected contours in the skeleton endponitns
        if skl is None:
            skl = skeletonize(crack>0, method='lee')
        #Finding skeleton endpoints
        skl_end_points_mask = find_end_points(skl>0)
        skl_end_points = np.array(np.where(skl_end_points_mask>0)).T           
        
        #Finding junctions in the skeleton
        #skl_junct_points_mask = find_skeleton_intersections(skl)
        #skl_junct_points = np.array(np.where(skl_junct_points_mask>0)).T
        skl_junct_points = _getnodes(skl, extremes=False) #!more efficient

        
        #Divide contours in edges taking into account the endpoints and intersections.
        #Contours are divided in the closest point to either intersections or endpoints.
        #Junction points
        for ep in skl_junct_points:
            min_dist_c_ep = []
            dist_c_ep_array = []
            for cnt in crack_contour:
                dist_c_ep = np.linalg.norm(cnt-ep.reshape(-1,2), axis=1)
                dist_c_ep_array.append(dist_c_ep)
                min_dist_c_ep.append(np.min(dist_c_ep))
            dist_c_ep_array = np.array(dist_c_ep_array)
            
            #Contours that are closer than a thresold based on the min distance of all edges
            min_dist_c_ep = np.array(min_dist_c_ep)
            ind_clossest_cnt = np.where(min_dist_c_ep<3*np.min(min_dist_c_ep))[0]
            
            #Loop through the contours that met the criteria
            new_crack_contour = []
            for icc in ind_clossest_cnt:
            
                #max dist to point (to reorganize countour and divide it properly when endpoints is close to the start point)
                dist2ep = np.linalg.norm(crack_contour[icc]-ep.reshape(-1,2), axis=1)       
                max_dist2ep = np.max(dist2ep)
                ind_max_dist2ep = np.where(max_dist2ep==dist2ep)[0][0]        
                
                #Orginize points of crack contour to divide into edges properly
                cr_contour_org = np.copy(crack_contour[icc])
                cr_contour_org = np.concatenate((cr_contour_org[ind_max_dist2ep:], cr_contour_org[:ind_max_dist2ep]))
                crack_contour[icc] = np.copy(cr_contour_org)
                
                #Find the ind of the local minima
                ind_min_extrema = argrelextrema(dist_c_ep_array[icc], np.less)[0]
                #filter local minima. If the ind is to close to the next do not take into account
                ind_min_extrema_filter = []
                last_ime = 0
                for ime in ind_min_extrema:
                    if last_ime==0 or np.abs(last_ime-ime)>10: #at least 10 ind away
                        ind_min_extrema_filter.append(ime)
                        last_ime = ime
                ind_min_extrema_filter = np.array(ind_min_extrema_filter)
                        
                #Breaking the edg in the ime (ind min extreme) point
                last_ime = 0
                full_contour = np.copy(crack_contour[icc])            
                for ime in ind_min_extrema_filter:
                    crack_contour_div = full_contour[last_ime:ime]
                    last_ime = ime
                    new_crack_contour.append(crack_contour_div)
                crack_contour_div = full_contour[last_ime:]                  
                new_crack_contour.append(crack_contour_div)
              
            #Delete full contour from the list
            crack_contour = [crack_contour[i] for i in range(len(crack_contour)) if i not in ind_clossest_cnt]
            #Adding new contours to crack_countour initial list after deleting the divided contour
            crack_contour = crack_contour + new_crack_contour

        #Get rid off small detected contours
        crack_contour = [cr_ct for cr_ct in crack_contour if len(cr_ct)>10]

        #Intersection points
        for ep in skl_end_points:
            min_dist_c_ep = []
            for cnt in crack_contour:
                if len(cnt)==0:
                    continue
                dist_c_ep = np.linalg.norm(cnt-ep.reshape(-1,2), axis=1)
                min_dist_c_ep.append(np.min(dist_c_ep))
            #min dist count end point   
            min_dist_c_ep = np.array(min_dist_c_ep)
            ind_clossest_cnt = np.where(min_dist_c_ep==np.min(min_dist_c_ep))[0][0]
            
            #max dist to point (to reorganize countour and divide it properly when endpoints is close to the start point)
            dist2ep = np.linalg.norm(crack_contour[ind_clossest_cnt]-ep.reshape(-1,2), axis=1)       
            max_dist2ep = np.max(dist2ep)
            ind_max_dist2ep = np.where(max_dist2ep==dist2ep)[0][0]        
            
            #Orginize points of crack contour to divide into edges properly
            cr_contour_org = np.copy(crack_contour[ind_clossest_cnt])
            cr_contour_org = np.concatenate((cr_contour_org[ind_max_dist2ep:], cr_contour_org[:ind_max_dist2ep]))
            crack_contour[ind_clossest_cnt] = np.copy(cr_contour_org)
            
            #Minimum distance edge to end point to select point to divide contour
            #for the contour selected, what is the minimum distance to the ep
            dist2ep = np.linalg.norm(crack_contour[ind_clossest_cnt]-ep.reshape(-1,2), axis=1)       
            min_dist2ep = np.min(dist2ep)
            ind_min_dist2ep = np.where(min_dist2ep==dist2ep)[0][0]
            
            #Get the clossest full contour
            full_contour = np.copy(crack_contour[ind_clossest_cnt])
            #Delete full contour from the list
            crack_contour.pop(ind_clossest_cnt)
            #Dividing full contour
            crack_contour_div0 = full_contour[:ind_min_dist2ep]
            crack_contour_div1 = full_contour[ind_min_dist2ep:]
            #Adding new contours to crack_countour initial list after deleting the divided contour
            crack_contour.append(crack_contour_div0)
            crack_contour.append(crack_contour_div1)  

            #Get rid off small detected contours
            crack_contour = [cr_ct for cr_ct in crack_contour if len(cr_ct)>10]          
                
        
        #plt.figure()
        #plt.imshow(crack, 'gray')
        #for cr_ct in crack_contour:
        #    plt.scatter(cr_ct[:,1], cr_ct[:,0], c=np.random.rand(1,3))
        #plt.savefig(dir_save+mask_name[:-4]+'_multiple_edges.png', bbox_inches='tight', pad_inches=0)
        #plt.savefig(dir_save+mask_name[:-4]+'_multiple_edges.pdf', bbox_inches='tight', pad_inches=0)
        #plt.close()
        
        #Filter edges according their lenght. 
        if eta is None:        
            crack_edges = [np.flip(cr_ct,1) for cr_ct in crack_contour if len(cr_ct)>k_neighboors] # get rid of small edges
            #Warning as some edges will be deleted
            if len(crack_edges)<len(crack_contour):
                print("WARNING -- some contours were deleted as their quantity of points are less than the k_neighboors required")
                print("        -- consider reducing the k_neighbors")
        
        else: #get rid off really very edges
            crack_edges = [np.flip(cr_ct,1) for cr_ct in crack_contour if len(cr_ct)>10] 
                    
    
    #Adding normal information to the crack_edges
    crack_edges_normals = []
    if normals:
        k_n_normal_feature = k_n_normal_feature
    else:
        k_n_normal_feature = 3
    for ce in crack_edges:
        ce_n = (np.pi*find_dir_nor(None, ce, k_n=k_n_normal_feature)[:,6]/180).reshape((len(ce),-1))
        crack_edges_normals.append(np.concatenate((ce,ce_n),axis=1))
    
    #Adding k-adaptable based on eta
    #if eta is given, k-neighbor is computed as int(len(contour_segment)/eta)
    if eta is not None:
        k_crack_edges_normals = []
        for ce in crack_edges:
            k_crack_edges_normals.append(int(len(ce)/eta))
    
    #Finding skeleton
    if skl is None:
        skl = skeletonize(crack>0, method='lee')
    ind_skl = np.where(skl)
    
    #Array with transformed points of edge with optimal H
    crack_edge_transf = np.empty(shape=[0,2])
    crack_edge_transf_loc = np.empty(shape=[0,2]) #from local transf
    
    #Array with global coordinates of region skeleton
    crack_skl_global = np.empty(shape=[0,2])
    
    #List with the H_params for each point in skeleton in the region
    H_params_crack_region = []
    H_params_crack_region_loc = []
    H_params_crack_region_dict = {}
    
    #Kinematics for each point in the skeleton represented by two dofs, horizontal and vertical displacement.
    #The values will be the mean of the difference between final and initial coordinates of the k-neighboors
    #of the edge0 (global and local are the same)
    two_dofs_global = []
    two_dofs_local = []
    
    #To register time in each iteration
    time_iteration = []
    
    #List to define the class of crack. Class I ascending. Class II descending
    crack_class = []
    
    counter = 0
    pbar = tqdm(total=len(ind_skl[1]))
    #Looping through the skeleton points (x,y) and computing the kinematics for the related finite edge segments
    last_x, last_y = np.inf, np.inf
    for x,y in zip(ind_skl[1],ind_skl[0]):
        t0i = time.time() #initial time iteration i

        #If window overlaping is given, check if the pixel is further than the window_overlap. If so, compute kinematic. If not, copy last computed results
        if window_overlap is not None:
            #distance to the last sk pt where kinematic was computed
            dist_last_xy = ((last_x-x)**2 + (last_y-y)**2)**.5
            if dist_last_xy < window_overlap:
                
                if not ignore_global:
                    H_params_crack_region.append([[float(x),float(y)], [float(x),float(y)], H_params_crack_region_])
                    crack_edge_transf = np.concatenate((crack_edge_transf, crack_edge_transf_))
                    two_dofs_global.append([[float(x),float(y)],[float(x),float(y)], two_dofs_global_])                    

                crack_skl_global = np.concatenate((crack_skl_global, np.array([[x,y]])))
                H_params_crack_region_loc.append([[float(x),float(y)], [float(x),float(y)], H_params_crack_region_loc_])
                crack_edge_transf_loc = np.concatenate((crack_edge_transf_loc, crack_edge_transf_loc_))
                two_dofs_local.append([[float(x),float(y)],[float(x),float(y)], two_dofs_local_])
                crack_class.append([[float(x),float(y)], crack_class_])

                tfi = time.time()        
                time_iteration.append([[float(x),float(y)],[float(x),float(y)], tfi-t0i])
                pbar.update()
                continue
        
        #If monte_carlo is true, just sample randomly one point and break the loop.
        if monte_carlo:
            sample = np.random.randint(0, len(ind_skl[1]))
            x = ind_skl[1][sample]
            y = ind_skl[0][sample]        
        
        if counter%10==0: print("Progress thorugh skeletong is ", 100*(counter/len(ind_skl[1])), "%")
        
        counter+=1
        dist_edges = []
        set_crack_edges = []
        #Selecting the two oposite edges of the crack closest to the crack skeleton point (x,y)
        for i, cr_edg in enumerate(crack_edges_normals):
            
            #if eta is given, use the adaptable k
            if eta is not None:
                k_neighboors = k_crack_edges_normals[i]            
            
            #if neighboors are more than the edge lenght use lenght
            if k_neighboors+1>len(np.concatenate((np.array([[x,y]]), cr_edg[:,:2]))):
                k_neighboors = len(np.concatenate((np.array([[x,y]]), cr_edg[:,:2]))) - 1            
            
            #Computing distances of possible edges to the crack skeleton point (x,y)
            nbrs_crk_edge = NearestNeighbors(n_neighbors=k_neighboors+1, algorithm='ball_tree').fit(np.concatenate((np.array([[x,y]]), cr_edg[:,:2]))) #+1 as the first element is the skeleton point
            dist_edge, indic_edge = nbrs_crk_edge.kneighbors(np.concatenate((np.array([[x,y]]), cr_edg[:,:2])))
            ind_edge_xy = indic_edge[0, 1:]-1 #cols from 1: and -1 as the first element is the skeleton point
            dist_edge_xy = dist_edge[0, 1:]
            set_crack_edge = cr_edg[ind_edge_xy]
            
            #Lists with edge data to select points from 2 edges that are clossest to (x,y) point of the skeleton
            dist_edges.append(dist_edge_xy)
            set_crack_edges.append(set_crack_edge)
            
        dist_closest_point = np.array([np.min(dt_edg) for dt_edg in dist_edges])
        ids_sets_01 = np.argsort(dist_closest_point)[:2] #indices of set of points of the 2 closest edges
        
        
        #Defining edge0 and edge1 to compute registration among two edge possibilities.
        #There are two cases of crack segments. Case I, ascending. Case II, descending (left to right)
        #Edge0 always the edge at the top in ascending cracks and at the bottom for descending. (Check first draft paper)
        #For the selection of the edge0 and edge1 following that criteria, and approx crack angle is computed
        #with two points belonging to one of the edges.
        #For almost vertical cracks (45<|angle|), the sum of x coordinates are checked for both edge possibilities
        #For almost horizontal cracks (45>|angle|), the sum y coordinates are checked for both edge possibilities. 
        #According if crack is vertical or horizontal and the sum of x or sum of y the edges edge0 and edge1 are defined.
        if set_crack_edges[ids_sets_01[0]][:31][-1][0] > set_crack_edges[ids_sets_01[0]][:31][0][0]: #to guarantee that vector is pointing to the right #31 is defined by experiments --Quantity of points of edge to compute angle
            angle_edge0 = set_crack_edges[ids_sets_01[0]][:31][-1] - set_crack_edges[ids_sets_01[0]][:31][0]
        else:
            angle_edge0 = set_crack_edges[ids_sets_01[0]][:31][0] - set_crack_edges[ids_sets_01[0]][:31][-1]
        angle_edge0 = np.arctan(angle_edge0[1]/(angle_edge0[0]+1e-15))
        if np.abs(angle_edge0)<np.pi/4: 
            if angle_edge0<0: #Take into account that image coordinates are flipped that is why < instead of > --> #image ascending (in the normal axes would be descending)
                if np.sum(set_crack_edges[ids_sets_01[0]][:31][:,1]) < np.sum(set_crack_edges[ids_sets_01[1]][:31][:,1]):
                    set_crack_edge_0 = set_crack_edges[ids_sets_01[0]]
                    set_crack_edge_1 = crack_edges_normals[ids_sets_01[1]]
                    if eta is not None:
                        k_neighboors = k_crack_edges_normals[ids_sets_01[0]]  #The neighboors will depend on edge 0 
                else:
                    set_crack_edge_0 = set_crack_edges[ids_sets_01[1]]
                    set_crack_edge_1 = crack_edges_normals[ids_sets_01[0]]
                    if eta is not None:
                        k_neighboors = k_crack_edges_normals[ids_sets_01[1]]  #The neighboors will depend on edge 0 
            else:
                if np.sum(set_crack_edges[ids_sets_01[0]][:31][:,1]) > np.sum(set_crack_edges[ids_sets_01[1]][:31][:,1]):
                    set_crack_edge_0 = set_crack_edges[ids_sets_01[0]]
                    set_crack_edge_1 = crack_edges_normals[ids_sets_01[1]]
                    if eta is not None:
                        k_neighboors = k_crack_edges_normals[ids_sets_01[0]]  #The neighboors will depend on edge 0 
                else:
                    set_crack_edge_0 = set_crack_edges[ids_sets_01[1]]
                    set_crack_edge_1 = crack_edges_normals[ids_sets_01[0]]    
                    if eta is not None:
                        k_neighboors = k_crack_edges_normals[ids_sets_01[1]]  #The neighboors will depend on edge 0 
        else:        
            if np.sum(set_crack_edges[ids_sets_01[0]][:31][:,0]) < np.sum(set_crack_edges[ids_sets_01[1]][:31][:,0]):
                set_crack_edge_0 = set_crack_edges[ids_sets_01[0]]
                set_crack_edge_1 = crack_edges_normals[ids_sets_01[1]] 
                if eta is not None:
                    k_neighboors = k_crack_edges_normals[ids_sets_01[0]]  #The neighboors will depend on edge 0 
            else:
                set_crack_edge_0 = set_crack_edges[ids_sets_01[1]]
                set_crack_edge_1 = crack_edges_normals[ids_sets_01[0]] 
            if eta is not None:
                        k_neighboors = k_crack_edges_normals[ids_sets_01[1]]  #The neighboors will depend on edge 0 
                    
        #New signs convention (positive opening, positive shear displacement clockwise) -- To compute it is necessary to know crack class
        #Assending happens when angle_edge0 is negative (as the y axis is flipped in the images)
        #Known the class, the shear/tangential diplacements follow the next rule:
        #if it is class 1, change the sign. if it is class 2, keep the sign. This is because initial convention (check initial paper draft)        
        #Defining class
        if angle_edge0<0: #Acending, class 1. Descending, class 2.
            crack_class_ = 1
            crack_class.append([[float(x),float(y)], crack_class_])
            #crack_class.append([[float(x),float(y)], 1])
        else:
            crack_class_ = 2
            crack_class.append([[float(x),float(y)], crack_class_])
            #crack_class.append([[float(x),float(y)], 2])
        
        #Finding the m*k-neighboors for set_crack_edge_1
        mk_neighboors = int(m*k_neighboors) if m*k_neighboors+1 < len(set_crack_edge_1) else len(set_crack_edge_1)
        nbrs_crk_edge1 = NearestNeighbors(n_neighbors=mk_neighboors+1, algorithm='ball_tree').fit(np.concatenate((np.array([[x,y]]), set_crack_edge_1[:,:2]))) #+1 as the first element is the skeleton point
        dist_edge1, indic_edge1 = nbrs_crk_edge1.kneighbors(np.concatenate((np.array([[x,y]]), set_crack_edge_1[:,:2])))
        ind_edge1_xy = indic_edge1[0, 1:]-1 #cols from 1: and -1 as the first element is the skeleton point
        set_crack_edge_1_mk = set_crack_edge_1[ind_edge1_xy] 
        
        #Ordering set_crack_edge_1_mk from starting from endpoint --as it is it grows from the clossest point to the skeleton.
        #The last point of the set_crack_edge_1_mk is the furthest to the x,y point from skeleton, then it should be ordered starting from it. 
        #For this it is used kneighboors algorithm using k the len of set_crack_edge_1_mk
        nbrs_crk_edge1_ord = NearestNeighbors(n_neighbors=len(set_crack_edge_1_mk), algorithm='ball_tree').fit(set_crack_edge_1_mk[:,:2])
        dist_edge1_ord, indic_edge1_ord = nbrs_crk_edge1_ord.kneighbors(set_crack_edge_1_mk[:,:2])
        ind_edge1_ord = indic_edge1_ord[-1] #cols from 1: and -1 as the first element is the skeleton point
        set_crack_edge_1_mk_ord = set_crack_edge_1_mk[ind_edge1_ord] 
        
        #Computation of matrix transformation H optimal        
        H_type = "euclidean"        
        if not pareto or m==1: #it does not make sense to use pareto if k==mk
            H_op, H_op_loc = kinematic_adjustment_lambda_based(mask, mask_name, x,y, k_neighboors, set_crack_edge_0, set_crack_edge_1_mk_ord, l, omega, H_type=H_type, normals=normals, ignore_global=ignore_global)
        else:
            H_op, H_op_loc = kinematic_adjustment_pareto_based(k_neighboors, set_crack_edge_0, set_crack_edge_1_mk_ord, l, omega, H_type=H_type, normals=normals, ignore_global=ignore_global)
            
        #Concatenate skeleton for crack region in global coordinates
        crack_skl_global = np.concatenate((crack_skl_global, np.array([[x,y]])))
        #crack_skl_global = np.concatenate((crack_skl_global, np.array([[x,y]])))

        #By default computations using global coordinates are ignored. If required, H also is computed using those coordinates.
        if not ignore_global:
            print("DID NOT IGNORE GLOBAL")
            #Testing transformed GLOBAL
            H = H_from_transformation(H_op, H_type)    
           
            #Transforming one edge to overlap over the other globally
            set_crack_edge_0_op = np.concatenate((np.copy(set_crack_edge_0[:,:2]), np.ones((len(set_crack_edge_0),1))), axis=1).T
            set_crack_edge_0_op = H @ set_crack_edge_0_op 
            set_crack_edge_0_op  /= set_crack_edge_0_op[2]
            set_crack_edge_0_op  = set_crack_edge_0_op[:2].T
            
            #Concatenate transformed edge 0 (select just the transformation of the closest point to the x,y)
            crack_edge_transf_ = set_crack_edge_0_op[0].reshape(-1,2)
            crack_edge_transf = np.concatenate((crack_edge_transf, crack_edge_transf_))
            #crack_edge_transf = np.concatenate((crack_edge_transf, set_crack_edge_0_op[0].reshape(-1,2)))
            
            #Append H_params for point x,y in skeleton in the region [(local coodinates), (global coordinates), H_params]
            H_params_crack_region_ = H_op.tolist()
            H_params_crack_region.append([[float(x),float(y)], [float(x),float(y)], H_params_crack_region_])
            #H_params_crack_region.append([[float(x),float(y)], [float(x),float(y)], H_op.tolist()])#np.abs(H_op)])
            
            #Computing 2dofs for the current pixel(point) of the skeleton
            two_dofs_xy_global = np.mean((set_crack_edge_0_op-set_crack_edge_0[:,:2]), axis=0)
            two_dofs_global_ = two_dofs_xy_global.tolist()
            two_dofs_global.append([[float(x),float(y)],[float(x),float(y)], two_dofs_global_])
            #two_dofs_global.append([[float(x),float(y)],[float(x),float(y)], two_dofs_xy_global.tolist()])
        
        #Transformation H matrix as 3x3 size. 
        H_loc = H_from_transformation(H_op_loc, H_type)   
        
        #Local edges coordinates
        mean_edge_0 = np.mean(set_crack_edge_0[:,:2], axis=0)
        set_crack_edge_0_loc = np.copy(set_crack_edge_0) 
        set_crack_edge_0_loc[:,:2] -=  mean_edge_0
        
        #Transforming one edge to overlap over the other globally
        set_crack_edge_0_op_loc = np.concatenate((np.copy(set_crack_edge_0_loc[:,:2]), np.ones((len(set_crack_edge_0_loc),1))), axis=1).T
        set_crack_edge_0_op_loc = H_loc @ set_crack_edge_0_op_loc 
        set_crack_edge_0_op_loc  /= set_crack_edge_0_op_loc[2]
        set_crack_edge_0_op_loc  = set_crack_edge_0_op_loc[:2].T
        
        #Concatenate transformed edge 0 (select just the transformation of the closest point to the x,y)
        crack_edge_transf_loc_ = (set_crack_edge_0_op_loc[0].reshape(-1,2))+mean_edge_0
        crack_edge_transf_loc = np.concatenate((crack_edge_transf_loc, crack_edge_transf_loc_))
        #crack_edge_transf_loc = np.concatenate((crack_edge_transf_loc, (set_crack_edge_0_op_loc[0].reshape(-1,2))+mean_edge_0))
        
        #Append H_params for point x,y in skeleton in the region [(local coodinates), (global coordinates), H_params]
        H_params_crack_region_loc_ = H_op_loc.tolist()
        H_params_crack_region_loc.append([[float(x),float(y)], [float(x),float(y)], H_params_crack_region_loc_])
        #H_params_crack_region_loc.append([[float(x),float(y)], [float(x),float(y)], H_op_loc.tolist()])#np.abs(H_op)])

        #Computing 2dofs for the current pixel(point) of the skeleton
        two_dofs_xy_local = np.mean((set_crack_edge_0_op_loc-set_crack_edge_0_loc[:,:2]), axis=0)
        two_dofs_local_ = two_dofs_xy_local.tolist()
        two_dofs_local.append([[float(x),float(y)],[float(x),float(y)], two_dofs_local_])
        #two_dofs_local.append([[float(x),float(y)],[float(x),float(y)], two_dofs_xy_local.tolist()])
        
        #This contains the transformations globally and locally
        H_params_crack_region_dict[(x,y)] = (H_op, H_op_loc)
        
        pbar.update()
        
        tfi = time.time()        
        time_iteration.append([[float(x),float(y)],[float(x),float(y)], tfi-t0i])
        
        #If monte_carlo is true, just sample randomly one point and break the loop.
        if monte_carlo:
            break

        #Update last sk pt where kin was computed. When overlaping is given
        if window_overlap is not None:
            last_x, last_y = x, y
        
    
    pbar.close()
        
    #Computing skeleton direction
    skeleton = np.concatenate((ind_skl[1].reshape(-1,1), ind_skl[0].reshape(-1,1)), axis=1)
    #contour_dir_nor = find_dir_nor(mask_name,skeleton, dir_save=dir_save, mask=mask, plot=False)
    contour_dir_nor = find_dir_nor(mask_name,skeleton, dir_save=dir_save, mask=None, plot=False)
    
    #Kinematics as tangetial and normal displacements of the crack 
    if monte_carlo: 
        #If it is used for montecarlo simulation where just one sampled point of the skeleton is analyzed, just use the sample info
        #Computing kinematics normal tangential for global values
        kinematics_n_t = []#displacement array in normal and tangential directions to the crack plane
        for skl_pt, t_dofs_gl in zip([contour_dir_nor[sample]],two_dofs_global):
            beta = (skl_pt[6] if skl_pt[6]>0 else skl_pt[6]+180)*np.pi/180 #if the angle is negative, then add 180. This as n axis is pointing always in the I or II quartier.
            transf_matrix_skl_pt = np.array([[np.cos(beta), np.sin(beta)],[-np.sin(beta), np.cos(beta)]])
            t_global = np.array(t_dofs_gl[2]).reshape((2,-1))
            kin_n_t = (transf_matrix_skl_pt @ t_global).reshape(-1)
            kinematics_n_t.append([[skl_pt[0],skl_pt[1]],[skl_pt[0],skl_pt[1]], kin_n_t.tolist()])
        #Computing kinematics normal tangential for local values
        kinematics_n_t_local = []#displacement array in normal and tangential directions to the crack plane
        for skl_pt, t_dofs_loc in zip([contour_dir_nor[sample]],two_dofs_local):
            beta = (skl_pt[6] if skl_pt[6]>0 else skl_pt[6]+180)*np.pi/180 #if the angle is negative, then add 180. This as n axis is pointing always in the I or II quartier.
            transf_matrix_skl_pt_loc = np.array([[np.cos(beta), np.sin(beta)],[-np.sin(beta), np.cos(beta)]])
            t_local = np.array(t_dofs_loc[2]).reshape((2,-1))
            kin_n_t_loc = (transf_matrix_skl_pt_loc @ t_local).reshape(-1)
            kinematics_n_t_local.append([[skl_pt[0],skl_pt[1]],[skl_pt[0],skl_pt[1]], kin_n_t_loc.tolist()])
    else:        
        #Computing kinematics normal tangential for global values
        kinematics_n_t = []#displacement array in normal and tangential directions to the crack plane
        for skl_pt, t_dofs_gl in zip(contour_dir_nor,two_dofs_global):
            beta = (skl_pt[6] if skl_pt[6]>0 else skl_pt[6]+180)*np.pi/180 #if the angle is negative, then add 180. This as n axis is pointing always in the I or II quartier.
            transf_matrix_skl_pt = np.array([[np.cos(beta), np.sin(beta)],[-np.sin(beta), np.cos(beta)]])
            t_global = np.array(t_dofs_gl[2]).reshape((2,-1))
            kin_n_t = (transf_matrix_skl_pt @ t_global).reshape(-1)
            kinematics_n_t.append([[skl_pt[0],skl_pt[1]],[skl_pt[0],skl_pt[1]], kin_n_t.tolist()])        
        #Computing kinematics normal tangential for local values
        kinematics_n_t_local = []#displacement array in normal and tangential directions to the crack plane
        for skl_pt, t_dofs_loc in zip(contour_dir_nor,two_dofs_local):
            beta = (skl_pt[6] if skl_pt[6]>0 else skl_pt[6]+180)*np.pi/180 #if the angle is negative, then add 180. This as n axis is pointing always in the I or II quartier.
            transf_matrix_skl_pt_loc = np.array([[np.cos(beta), np.sin(beta)],[-np.sin(beta), np.cos(beta)]])
            t_local = np.array(t_dofs_loc[2]).reshape((2,-1))
            kin_n_t_loc = (transf_matrix_skl_pt_loc @ t_local).reshape(-1)
            kinematics_n_t_local.append([[skl_pt[0],skl_pt[1]],[skl_pt[0],skl_pt[1]], kin_n_t_loc.tolist()])
    
    #Saving kinematics outputs in dictionary 
    crack_kinematic[0] =  {}
    if not ignore_global:
        crack_kinematic[0]["H_params_cracks"] = H_params_crack_region        
        crack_kinematic[0]["edge_transf"] = crack_edge_transf.tolist() 
    crack_kinematic[0]["edges"] = [crack_edge.tolist() for crack_edge in crack_edges]
    crack_kinematic[0]["skl"] = crack_skl_global.tolist()
    crack_kinematic[0]["H_params_cracks_loc"] = H_params_crack_region_loc
    crack_kinematic[0]["edge_transf_loc"] = crack_edge_transf_loc.tolist()
    crack_kinematic[0]["two_dofs"] = two_dofs_global
    crack_kinematic[0]["two_dofs_loc"] = two_dofs_local
    crack_kinematic[0]["kinematics_n_t"] = kinematics_n_t
    crack_kinematic[0]["kinematics_n_t_loc"] = kinematics_n_t_local
    crack_kinematic[0]["dir_nor_skl"] = contour_dir_nor.tolist()
    crack_kinematic[0]["crack_class"] = crack_class
    crack_kinematic[0]["time_iteration"] = time_iteration
    
    #Timing process
    final_time = time.time()
    print("Time processing the crack kinematics: ", np.round(-initial_time+final_time, 2), "seconds")
    print("---------------Process finished---------------")
    
    #Making plots if asked
    if make_plots_local:
        #Plot edges and skeleton
        #plot_edges_kinematic(data_path, mask_name, crack_kinematic, dir_save=dir_save, plt_skeleton = True, local_trans=True)
        #Plot kinematics        
        #plot_kinematic(data_path, mask_name, crack_kinematic, dir_save=dir_save, local_trans=True, abs_disp = True)
        #Plot two_dofs kinematic        
        #plot_two_dofs_kinematic(data_path, mask_name, crack_kinematic,  dir_save=dir_save, local_trans = True, abs_disp = True)
        #Plot normal and tangential kinematic
        plot_n_t_kinematic(data_path, mask_name, crack_kinematic,  dir_save=dir_save,plot_trans_n=True, plot_trans_t=True, plot_trans_t_n=True,local_trans = True, abs_disp = True, cmap_=cmap_, resolution=resolution)
    
    if make_plots_global:
        #Plot edges and skeleton
        plot_edges_kinematic(data_path, mask_name, crack_kinematic,  dir_save=dir_save,plt_skeleton = True, local_trans=False)
        #Plot kinematics
        plot_kinematic(data_path, mask_name, crack_kinematic,  dir_save=dir_save, local_trans=False)
        #Plot two_dofs kinematic
        plot_two_dofs_kinematic(data_path, mask_name, crack_kinematic,  dir_save=dir_save, local_trans = False)
        #Plot normal and tangential kinematic
        plot_n_t_kinematic(data_path, mask_name, crack_kinematic, dir_save=dir_save, plot_trans_n=True, plot_trans_t=True, plot_trans_t_n=True, local_trans = False, resolution=resolution, dot_size=1)
    
    #Save kinematic dictionary as json file if asked
    with open(dir_save+'{}_crack_kinematic.json'.format(mask_name[:-4]), 'w') as fp:
        json.dump(crack_kinematic, fp)

    return crack_kinematic
