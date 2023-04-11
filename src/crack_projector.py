#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:35:45 2020

This script uses camera data provided by meshroom
software located in the cameras.sfm file (StructureFromMotion folder)
See function added in CameraInit.py module (nodes) in meshroom.
It was implemented a script to save intrinsic and poses information
as dictionaries in json files.

This script contains the codes to generate LOD3 models.
The codes are based on "Generation LOD3 models from structure-from-motion and semantic segmentation" 
by Pantoja-Rosero et., al.
https://doi.org/10.1016/j.autcon.2022.104430

@author: pantoja
"""
from PIL import Image
import numpy as np
import pylab as pylab
import os
import matplotlib.pyplot as plt
#import camera
import cv2
import sfm
from projection_tools import *
from facade_segmentation import fac_segmentation
from opening_segmentation import segment_opening
from utils_geometry import fit_plane_on_X
import open3d as o3d
from crack_kinematics.least_square_crack_kinematics import find_crack_kinematics
#import pymeshlab
#import copy
from classes import *
from crack_segmentation import crack_segmentation
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from tqdm import tqdm
from matplotlib.colors import ListedColormap
from matplotlib import cm
from tools_damage_projection import simple_crack_projector, simple_polygon_projector

def plot_n_t_kinematic(data_folder, mask_name, glob_coordinates_skl, two_dofs_n, two_dofs_t, crack_class, plot_trans_n=True, plot_trans_t=True, plot_trans_t_n = True, dot_size=None, resolution=None, save_name=""):
    """
    Plots 2dof tt (tangential), tn (normal) kinematic results

    Args:
    """

    #Folder to save kinematics results
    dir_save='../results/' + data_folder + '/'
    
    #To decide if local or global is required
    lt = "_loc"
    
    #Creating colormap
    top = cm.get_cmap('Oranges_r', 256)
    bottom = cm.get_cmap('Blues', 256)
    newcolors = np.vstack((top(np.linspace(0, 1, 256)), bottom(np.linspace(0, 1, 256))))
    newcmp = ListedColormap(newcolors, name='OrangeBlue')    
    
    #Reading mask        
    mask = cv2.imread('../results/' + data_folder + "/" + mask_name)
    mask = (mask==0)*255

    glob_coordinates_skl = np.array(glob_coordinates_skl, 'int')
    two_dofs_n = np.array(two_dofs_n)
    two_dofs_t = np.array(two_dofs_t)
    crack_class = np.array(crack_class)
    
    #Modifing signs according new sign sistem. Opening is possitive (always). Shear sliding, clockwise pair positive.
    #If class 1 (crack ascending), it is necessary to change sings with respect old convention
    #If class 2 (crack descending), it is necessary to keep sings with respect old convention
    two_dofs_n = np.abs(two_dofs_n)
    ind_class1 = np.where(crack_class==1)
    two_dofs_t[ind_class1[0]] *= -1

    if plot_trans_n and len(glob_coordinates_skl)>0:
        trans_n_img = np.zeros_like(mask, 'float')
        trans_n_img = trans_n_img[:,:,0]
        trans_n_img[(glob_coordinates_skl[:,1], glob_coordinates_skl[:,0])] = two_dofs_n.reshape(-1)
        
        if resolution is None:
            a = 1
        else:
            a = resolution ##mm/px
        
        #Ploting 
        plt.rcParams['font.size'] = '32'
        fig, ax = plt.subplots(1, figsize=(24,18))
            
        if dot_size is not None:
            psm = ax.scatter(glob_coordinates_skl[:,0], glob_coordinates_skl[:,1], c=a*two_dofs_n, cmap=bottom, vmin=0 , vmax=+20, s=dot_size)
        else:
            psm = ax.scatter(glob_coordinates_skl[:,0], glob_coordinates_skl[:,1], c=a*two_dofs_n, cmap=bottom, vmin=0 , vmax=+20, marker='.')
        ax.imshow(mask, alpha=0.3)
        clrbr = fig.colorbar(psm, ax=ax, ticks=[0,5,10,15,20])
        
        if resolution is None:
            clrbr.ax.set_title(r"$t_n [px]$")
        else:
            clrbr.ax.set_title(r"$t_n [mm]$")
        
        if resolution is not None: lt+="_mm"
        plt.tight_layout()
        fig.savefig(dir_save+save_name+mask_name[:-4]+'_n_t_kin_tn'+lt+'.png', bbox_inches='tight', pad_inches=0)
        fig.savefig(dir_save+save_name+mask_name[:-4]+'_n_t_kin_tn'+lt+'.pdf', bbox_inches='tight', pad_inches=0)
        plt.close()
        
    if plot_trans_t and len(glob_coordinates_skl)>0:
        trans_t_img = np.zeros_like(mask, 'float')
        trans_t_img = trans_t_img[:,:,0]
        trans_t_img[(glob_coordinates_skl[:,1], glob_coordinates_skl[:,0])] = two_dofs_t.reshape(-1)
        
        if resolution is None:
            a = 1
        else:
            a = resolution ##mm/px
        
        #Ploting 
        plt.rcParams['font.size'] = '32'
        fig, ax = plt.subplots(1, figsize=(24,18))       
        c_ = two_dofs_t
        if dot_size is not None:
            psm = ax.scatter(glob_coordinates_skl[:,0], glob_coordinates_skl[:,1], c=a*c_, cmap=newcmp, vmin=-20 , vmax=+20, marker='.', s=dot_size)
        else:
            psm = ax.scatter(glob_coordinates_skl[:,0], glob_coordinates_skl[:,1], c=a*c_, cmap=newcmp, vmin=-20 , vmax=+20, marker='.')
        ax.imshow(mask, alpha=0.3)
        clrbr = fig.colorbar(psm, ax=ax, ticks=[-20,-10,0,10,20])
        if resolution is None:
            clrbr.ax.set_title(r"$t_t [px]$")
        else:
            clrbr.ax.set_title(r"$t_t [mm]$")
        
        if resolution is not None: lt+="_mm"
        plt.tight_layout()
        fig.savefig(dir_save+save_name+mask_name[:-4]+'_n_t_kin_tt'+lt+'.png', bbox_inches='tight', pad_inches=0)
        fig.savefig(dir_save+save_name+mask_name[:-4]+'_n_t_kin_tt'+lt+'.pdf', bbox_inches='tight', pad_inches=0)
        plt.close()
        
    if plot_trans_t_n and len(glob_coordinates_skl)>0:
        trans_t_n_img = np.zeros_like(mask, 'float')
        trans_t_n_img = trans_t_n_img[:,:,0]
        trans_t_n_img[(glob_coordinates_skl[:,1], glob_coordinates_skl[:,0])] = (two_dofs_t/two_dofs_n).reshape(-1)
        
        #Ploting 
        plt.rcParams['font.size'] = '32'
        fig, ax = plt.subplots(1, figsize=(24,18))   
                    
        c_ = two_dofs_t/two_dofs_n

        if dot_size is not None:
            psm = ax.scatter(glob_coordinates_skl[:,0], glob_coordinates_skl[:,1], c=c_, cmap=newcmp, vmin=-2 , vmax=+2, marker='.', s=dot_size)
        else:
            psm = ax.scatter(glob_coordinates_skl[:,0], glob_coordinates_skl[:,1], c=c_, cmap=newcmp, vmin=-2 , vmax=+2, marker='.')
        ax.imshow(mask, alpha=0.3)
        clrbr = fig.colorbar(psm, ax=ax, ticks=[-2,-1,0,1,2])
        clrbr.ax.set_title(r"$t_t/t_n$")
        plt.tight_layout()
        fig.savefig(dir_save+save_name+mask_name[:-4]+'_n_t_kin_tt_tn'+lt+'.png', bbox_inches='tight', pad_inches=0)
        fig.savefig(dir_save+save_name+mask_name[:-4]+'_n_t_kin_tt_tn'+lt+'.pdf', bbox_inches='tight', pad_inches=0)
        plt.close()


def load_sfm_json(sfm_json_path):
    '''
    Given the path where the json file with the sfm information is (from Meshroom)
    extract information about instrinsic parameters, camera poses and structure.
    Retun them.
    Parameters
    ----------
    sfm_json_path : str
        Path to the json file of the sfm information.
    Returns
    -------
    intrinsic : dict
        Dictionary with the intrinsic information of the cameras used during SfM.
    poses : dict
        Dictionary with the poses information of the cameras used during SfM.
    structure : dict
        Dictionary with the structure information after SfM.
    '''

    # Load sfm.json file from meshroom output and return ordered
    # intrinsic, poses and structure for further manipulations
    with open(sfm_json_path + 'sfm.json', 'r') as fp:
        sfm = json.load(fp)

    v = sfm['views']
    i = sfm['intrinsics']
    p = sfm['poses']
    s = sfm['structure']
    # If there are more views than poses, delete extra views
    iii = 0
    while iii < len(p):
        if p[iii]['poseId'] == v[iii]['poseId']:
            iii += 1
        else:
            v.remove(v[iii])
            print(
                "More views than poses -- extra views deleted as images were not registered")

    while len(p) < len(v):  
        v.remove(v[iii])
        print("More views than poses -- extra views deleted as images were not registered")

    # Intrinsics
    k_intrinsic = {'pxInitialFocalLength', 'pxFocalLength', 'principalPoint',
                   'distortionParams'}
    intrinsic = {}
    for ii in i:
        key = ii['intrinsicId']
        intrinsic[key] = {}
        for l in k_intrinsic:
            intrinsic[key][l] = ii[l]

    # Poses
    k_poses = {'poseId', 'intrinsicId', 'path', 'rotation', 'center'}
    poses = {}
    for l, view in enumerate(v):
        key = view['path'].split('/')[-1]
        key = key.split('.')[0]
        poses[key] = {}
        for m in k_poses:
            if v[l]['poseId'] == p[l]['poseId']:
                if m in v[l]:
                    poses[key][m] = v[l][m]
                else:
                    poses[key][m] = p[l]['pose']['transform'][m]
            else:
                print("Error: views and poses are not correspondences")

    # Structure
    structure = {}
    for st in s:
        key = st['landmarkId']
        structure[key] = {}
        structure[key]['X_ID'] = st['landmarkId']
        structure[key]['X'] = st['X']
        structure[key]['descType'] = st['descType']
        structure[key]['obs'] = []
        for ob in st['observations']:
            structure[key]['obs'].append(
                {'poseId': ob['observationId'], 'x_id': ob['featureId'], 'x': ob['x']})

    return intrinsic, poses, structure

def crack_triangulation_1view_raycasting(im1, x1, P, R, t, K, k_dist, P_type, poses, i, plane_params=None, dom=None):
    """

    This function give the triangulation of 2D points related with the cracks skeleton (or points needed to be projected).
    It first find the points to be projected to 3D in the 2D image (view1) using CNN to segment the cracks and skeletoning the output.
    Then the 3D coordinates are found as the intersection of the ray passing through the camera center and the point in the image plane
    with:
        
    a) the facade plane in 3D. The facade plane in 3D can either be infered from the LOD2 model (clustering the model faces) or using the
    2D-3D correspondences given by the sfm.json output file from meshroom that lay inside the segmented facade

    b) directly the mesh triangles. The intersection is found using raycasting provided by open3D

    Args:
        im1 (list): list of images of view1 for each facade
        P (dict): camera matrices for all views registered in SfM
        i (int): id of current facade
        images_path (str): path to input images
        keypoints_path (str): path to opening corner points if given
        how2get_kp (int): value that defines the method to find 2D correspondences between 2 views
        opening_4points (dict): openings corner information comming from detection
        data_folder (_type_): name of the input data folder
        pp (list): list with the numer of points to be triangulated by facade. Needed if the how2get_kp is 2

    Returns:
        _type_: _description_
    """

    #Pose id
    P1_inid = P[im1[i]]['intrinsicId']

    #Make 2D coordinates, homogeneus
    x1 = (np.concatenate((x1, np.ones((len(x1),1))), axis=1)).T

    #Check if there are points to triangulate
    if len(x1.T)>0:


        #Correcting cordinates by lens distortion
        if P_type[P1_inid]=='radial3':
            x1_u = undistort_points(x1.T,K[P1_inid],k_dist[P1_inid]).T
            #undistort_img(imm1, K[P1_inid],k_dist[P1_inid])
        elif P_type[P1_inid]=='fisheye4':
            x1_u = undistort_points_fish(x1.T, K[P1_inid], k_dist[P1_inid]).T    
            #undistort_img_fish(imm1, K[P1_inid],k_dist[P1_inid])
        #print(x1, "before")
        x1 = x1_u
        #print(x1, "after")


        print("Cracks projection in pic " + im1[i])
        
        ## triangulate opening corners as intersection of plane and ray
        R_view = R[im1[i]]
        t_view = t[im1[i]]
        #Transformation matrix to transform image plane at pose place in 3D
        T_x3D = sfm.compute_T_for_x_to_3D(R_view, t_view)
        #Finding camera center and image plane coordinates of corners and points in 3D space
        c_h = np.array([0,0,0,1]).reshape((-1,1)) #camera_center
        c_3D = T_x3D @ c_h
        c_3D /= c_3D[3]
        X = []
        rays_list = []
        for v in x1.T:
            v_h = v.reshape((-1,1))
            v_h_norm = np.linalg.inv(K[poses[im1[i]]['intrinsicId']]) @ v_h        
            v_3D = T_x3D @ np.concatenate((v_h_norm, np.ones((1,1))), axis=0)
            v_3D /= v_3D[3]        
            if plane_params is not None:
                #Find itersection of plane with ray that passes by camera center and vertice
                V = (sfm.intersec_ray_plane(c_3D[:3], v_3D[:3], plane_params)).reshape(-1)
                X.append(V)
            elif dom is not None:
                ray_dir = v_3D[:3] - c_3D[:3]
                ray_dir = ray_dir/np.linalg.norm(ray_dir)
                rays_list.append([c_3D[0][0], c_3D[1][0], c_3D[2][0], ray_dir[0][0], ray_dir[1][0], ray_dir[2][0]])
                
        if plane_params is not None: 
            X = np.array(X)
        elif dom is not None:

            if len(x1.T)>0:
                LOD2_mesh_ = o3d.t.geometry.TriangleMesh.from_legacy(dom.LOD2.mesh)
                scene = o3d.t.geometry.RaycastingScene()
                scene.add_triangles(LOD2_mesh_)
                rays = o3d.core.Tensor(rays_list, dtype=o3d.core.Dtype.Float32)
                ans = scene.cast_rays(rays)
                dist2int = (ans['t_hit'].numpy()).reshape((-1,1))
                triangles_hit = ans['primitive_ids'].numpy()
                cr_dist = dist2int.reshape(-1)
                cr_hit = triangles_hit
                plane_id_hit = dom.LOD2.plane_clusters[0][cr_hit[np.where(cr_dist!=np.inf)]] #planes that are hit by the rays
                #Check if there are rays that do not hit the object for each point in skeleton
                if np.sum(cr_dist==np.inf)>0:
                    print("Some rays did not hit the obj")
                        
                dist2int = np.concatenate((dist2int, dist2int, dist2int), axis = 1)
                rays = rays.numpy()
                X = rays[:,:3] + dist2int*rays[:,3:]        
                id_cr_intersect = np.where(cr_dist!=np.inf)
                X = X[id_cr_intersect]
                X = X.reshape((-1,3))

                #Create crack objects and assign to the domain the cracs attribute
                cr = Crack()
                cr.view = im1[i]
                cr.plane = plane_id_hit
                cr.coord2d = (x1[:2].T)[id_cr_intersect]
                cr.coord = X
                cr.id_cr_intersect = id_cr_intersect
                dom.cracks.append(cr)
            
            else:
                X = np.array(X)
                #Create crack objects and assign to the domain the cracs attribute
                cr = Crack()
                cr.view = im1[i]
                cr.plane = np.array([])
                cr.coord2d = np.array(x1)
                cr.coord = X
                cr.id_cr_intersect = np.array([])
                dom.cracks.append(cr)
    else:
        X = np.empty(shape=[0,3])
        X = np.array(X)
        #Create crack objects and assign to the domain the cracs attribute
        cr = Crack()
        cr.view = im1[i]
        cr.plane = np.array([])
        cr.coord2d = np.array(x1)
        cr.coord = X
        cr.id_cr_intersect = np.array([])
        dom.cracks.append(cr)

    return X

def cr_detector(data_folder, images_path, opening_4points, fac_prediction, fac_opening_prediction):

    crack_prediction_bin = crack_segmentation(data_folder, images_path, out_return=True)

    #Getting original images
    list_images = os.listdir(images_path)
    list_images.sort()
    
    images = {}
    for img in list_images:
        images[img[:-4]] = cv2.imread(images_path + img)
    
    pred_bin_resized ={}
    for key in images:
        if opening_4points is not None:
            mask_facade = opening_4points[key][1]>0
            mask_openings = opening_4points[key][2]==0
        elif fac_prediction is not None:
            mask_facade = fac_prediction[key]>0
            mask_openings = fac_opening_prediction[key]==0
        pr_bn_rsz = crack_prediction_bin[key]/255
        pred_bin_resized[key] = pr_bn_rsz * mask_facade * mask_openings
    
        #Save filtered mask
        cv2.imwrite("../results/"+data_folder+"/crack_filtered_"+key+"_mask.png", (pred_bin_resized[key]>0)*255)

        #Overlay filtered mask
        mask_rgb = np.zeros((pred_bin_resized[key].shape[0], pred_bin_resized[key].shape[1], 3), dtype='uint8')
        mask_rgb[:,:,2] = (pred_bin_resized[key]>0)*255
        overlayed_mask = cv2.addWeighted(images[key], 0.5, mask_rgb, 1.0, 0)
        cv2.imwrite("../results/"+data_folder+"/crack_filtered_"+key+"_overlay.jpg", overlayed_mask)
        
    pred_bin_resized_skeleton ={}
    
    for key in pred_bin_resized:
        pred_bin_resized_skeleton[key] = skeletonize(pred_bin_resized[key], method='lee')
  
    #Extracting information of points contained in cracks using skeleton
    cracks_information = {}
    for key in images:
        cracks_information[key] ={}
        ind_skl = np.where(pred_bin_resized_skeleton[key])
        x_coord = ind_skl[1]
        x_coord = x_coord.reshape(x_coord.shape[0],1)
        y_coord = ind_skl[0]
        y_coord = y_coord.reshape(y_coord.shape[0],1)
        cracks_information[key]['points'] = np.hstack((x_coord, y_coord))
   
    return cracks_information, pred_bin_resized_skeleton


def crack_projector(DADT, data_folder, images_path, keypoints_path, intrinsic, poses, structure, im1, opening_4points, ray_int_method="casting", compute_kinematics=False, cracks2textured=False, extra_features = None):
    """

        Creates 3D objects for the openings segmented with deep learning using the camera matrices
        correspondend to 2 view points of a group of building's facades

    Args:
        data_folder (str): path to input data folder
        images_path (str): path to input images
        keypoints_path (str): path to keypoints
        polyfit_path (str): path to LOD2 model produced by polyfit
        how2get_kp (int): method to get opening corners to triangulate 0:from CNN detector+homography, 1: loading npy files, 2:clicking on two views, 3: v1 from npy+v2 with homography, 4:from CNN detector+homography
        im1 (list): list of input images view 1 for each facade
        im2 (list): list of input images view 2 for each facade
        pp (int, optional): list with number of corners to be triangulated for each facade - used if how2get_kp==2
        P (dict): camera matrices for each camera pose
        K (dict): intrinsic matrices for each camera
        k_dist (dict): distorsion parameters for each camera
        opening_4points (dict): opening corners for each facade
        dense (bool, optional): if true, it loads the polyfit model from dense point cloud. Defaults to False.
        ctes (list): constants that define thresholds when refining and aligning openings -- the lower the less influence
    """

    fac_prediction = None
    fac_opening_prediction = None
    #Detecting facades if it was not done during opening detection
    if opening_4points is None:
        fac_prediction = fac_segmentation(data_folder, images_path+'im1/', out_return=True)
        fac_opening_prediction = segment_opening(data_folder, images_path, 'im1/')

    #Getting crack information 
    crack_information, pred_bin_resized_skeleton = cr_detector(data_folder, images_path+'im1/', opening_4points, fac_prediction, fac_opening_prediction)

    #Reading camera poses information
    P, K, R, t, k_dist, P_type = camera_matrices(intrinsic, poses, return_Rt=True)

    #Loop through each par of pictures given by im1 and im2
    cc_vv = 1 #vertices counter. Helper to identify vertices in generated faces (all)
    cc_vv_c = 1 #vertices counter. Helper to identify vertices in generated cracks (all)
    for i in range(len(im1)): 
        #i=2
        #2D points to be projected
        x1 = crack_information[im1[i]]['points']
        if ray_int_method == "plane":
            #Finding 2D-3D correspondences between view and structure
            X_x_view, type_X_x_view = sfm.find_X_x_correspondences(im1[i], structure, poses, plot_X=False)
            for X_x in X_x_view:
                x_sfm.append(X_x[3])
                X_sfm.append(X_x[1])
            x_sfm = np.array(x_sfm).astype('float')
            X_sfm = np.array(X).astype('float')
            x_facade_sfm = np.empty(shape=[0,2])
            X_facade_sfm = np.empty(shape=[0,3])
            if opening_4points is None:
                mask_facade = fac_prediction[im1[i]]
            else:
                mask_facade = opening_4points[im1[i]][1]
            for X_x in X_x_view:
                inside_poly = mask_facade[int(float(X_x[3][1])), int(float(X_x[3][0]))]
                if inside_poly > 0:
                    x_facade_sfm = np.concatenate((x_facade_sfm, np.array(X_x[3]).astype('float').reshape((1,2))))
                    X_facade_sfm = np.concatenate((X_facade_sfm, np.array(X_x[1]).astype('float').reshape((1,3))))

            #Finding plane parameters
            plane_normal, plane_params = fit_plane_on_X(X_facade_sfm, fit_ransac=True)
            #Finding 3D initial coordinates of opening coordinates
            X = crack_triangulation_1view_raycasting(im1, x1, P, R, t, K, k_dist, P_type, poses, i, plane_params=plane_params)
        elif ray_int_method == "casting":
            X = crack_triangulation_1view_raycasting(im1, x1, P, R, t, K, k_dist, P_type, poses, i, dom=DADT)#LOD2_triangle_clusters, LOD2_cluster_plane_params, LOD2_mesh=LOD2_mesh) 
        
                
    #DADT.assign_cracks2planes()
    #Save cracks    
    DADT.save_cracks(data_folder)
    print("END CRACK PROJECTION")

    #Computing kinematics for detected cracks
    if compute_kinematics:
        crack_kinematics = {}
        for i in range(len(im1)):
            mask_name = "crack_filtered_" + im1[i] + "_mask.png"
            if os.path.isfile('../results/'+data_folder+'/kinematics/'+mask_name[:-4]+'_crack_kinematic.json'):
                # Opening JSON file
                with open('../results/'+data_folder+'/kinematics/'+mask_name[:-4]+'_crack_kinematic.json') as json_file:
                    crack_kinematics[im1[i]] = json.load(json_file)
                
            else:
                crack_kinematics[im1[i]]  = find_crack_kinematics(data_folder, mask_name, k_neighboors=50, m=1, l=1, k_n_normal_feature=10, omega=0, make_plots_local=True, window_overlap=3000, skl = pred_bin_resized_skeleton[im1[i]])
            
            #Reading kinematics for view and assigning to the crack object of DADT domain
            for key in crack_kinematics[im1[i]]:
                glob_coordinates_skl = [coord[0] for coord in crack_kinematics[im1[i]][key]["kinematics_n_t_loc"]]
                two_dofs_n = [t_dofs_n[2][0] for t_dofs_n in crack_kinematics[im1[i]][key]["kinematics_n_t_loc"]]
                two_dofs_t = [t_dofs_t[2][1] for t_dofs_t in crack_kinematics[im1[i]][key]["kinematics_n_t_loc"]]
                crack_class = [cr_cl[1] for cr_cl in crack_kinematics[im1[i]][key]["crack_class"]]            
            glob_coordinates_skl = np.array(glob_coordinates_skl, 'int')
            two_dofs_n = np.array(two_dofs_n).reshape((-1,1))
            two_dofs_t = np.array(two_dofs_t).reshape((-1,1))
            crack_class = np.array(crack_class).reshape((-1,1))
            
            if len(glob_coordinates_skl)>0:
                #Just select the information of projected cracks
                glob_coordinates_skl1 = glob_coordinates_skl[DADT.cracks[i].id_cr_intersect[0]]
                two_dofs_n1 = two_dofs_n[DADT.cracks[i].id_cr_intersect[0]]
                two_dofs_t1 = two_dofs_t[DADT.cracks[i].id_cr_intersect[0]]
                crack_class1 = crack_class[DADT.cracks[i].id_cr_intersect[0]]
            else:
                glob_coordinates_skl1 = glob_coordinates_skl
                two_dofs_n1 = two_dofs_n
                two_dofs_t1 = two_dofs_t
                crack_class1 = crack_class

            #Ploting kinematics on 2D (cracks projected)
            plot_n_t_kinematic(data_folder, mask_name, glob_coordinates_skl1, two_dofs_n1, two_dofs_t1, crack_class1, plot_trans_n=True, plot_trans_t=True, plot_trans_t_n = True, dot_size=1, save_name="cracks_projected_")

            #Defining signs according sign convention adopted on Pantoja-Rosero etal (2022)
            two_dofs_n1 = np.abs(two_dofs_n1)
            ind_class1 = np.where(crack_class1==1)
            two_dofs_t1[ind_class1[0]] *= -1
           
            #Assigning kinematic to domain
            if len(glob_coordinates_skl)>0:
                DADT.cracks[i].kinematics = np.concatenate((two_dofs_n1, two_dofs_t1, glob_coordinates_skl1), axis=1)
            else:
                DADT.cracks[i].kinematics = np.empty(shape=(0,4))

        #Saving pointcloud with kinematics color encoded
        DADT.save_cracks(data_folder, kin_n=True)
        DADT.save_cracks(data_folder, kin_t=True)
        DADT.save_cracks(data_folder, kin_tn=True)


    if cracks2textured:
        DADT.save_cracks2d(data_folder)
        images_path = "../data/"+data_folder+"/images/im1/"
        obj_path = "../data/"+data_folder+"/textured/texturedMesh.obj"
        if compute_kinematics:
            kinematics=True
        else:
            kinematics=False
        simple_crack_projector(data_folder, images_path, obj_path, intrinsic, poses, im1, kinematics=kinematics, load_cracks2d=True)

    if extra_features is not None:
        images_path = "../data/"+data_folder+"/images/im1/"
        for label_polygon in extra_features:
            obj_path = "../data/"+data_folder+"/polyfit/polyfit.obj"
            simple_polygon_projector(data_folder, images_path, obj_path, intrinsic, poses, im1, label_polygon = label_polygon)
            if cracks2textured:
                obj_path = "../data/"+data_folder+"/textured/texturedMesh.obj"
                simple_polygon_projector(data_folder, images_path, obj_path, intrinsic, poses, im1, label_polygon = label_polygon)       