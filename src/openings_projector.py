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
import homography
import cv2
import sfm
from projection_tools import *
from facade_segmentation import fac_segmentation
from utils_geometry import fit_plane_on_X, proj_op2plane, open2local, op_aligning1, op_aligning2, op_aligning3
from opening_detector import main_op_detector
import open3d as o3d
#import pymeshlab
#import copy
from classes import *
import time

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
    # If there are mor views than poses, delete extra views(I suppose are those images not taken in SfM by meshroom)
    iii = 0
    while iii < len(p):
        if p[iii]['poseId'] == v[iii]['poseId']:
            iii += 1
        else:
            v.remove(v[iii])
            print(
                "More views than poses -- extra views deleted as images were not registered")

    while len(p) < len(v):  # CHECK IF PERFORMS WELL
        v.remove(v[iii])
        print("More views than poses -- extra views deleted as images were not registered")

    # Intrinsics
    k_intrinsic = {'pxInitialFocalLength', 'pxFocalLength', 'principalPoint',
                   'distortionParams', 'type'}
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

def track_kps(data_folder, imm1, imm2, im_n1, im_n2, kps_imm1=None):
    """

    #LKT to find other points
    #Lukas Kanade Tracker. Used to finde detected opening corners in im1
    over im2 using sequential frames

    Args:
        data_folder (str): input data folder name
        imm1 (array): image view1
        imm2 (array): image view2
        im_n1 (str): image name view1
        im_n2 (str): image name view2
        kps_imm1 (array, optional): points to be mapped from view1 to view2. Defaults to None.

    Returns:
        good_new (array): points mapped from view1 to view2 using LKT
    """
    
    # Create params for lucas kanade optical flow
    lk_params = dict( winSize  = (60,60), maxLevel = 10, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    #Finding kps to be traked (if not given)
    gray1 = cv2.cvtColor(imm1,cv2.COLOR_RGB2GRAY)
    shp1 = gray1.shape[:2]
    gray2 = cv2.cvtColor(imm2,cv2.COLOR_RGB2GRAY)
    shp2 = gray2.shape[:2]
    
    #if shapes are different, resize 2nd image to perform LKT. Later is needed to rescale points.
    if shp1!=shp2:
        gray2 = cv2.resize(gray2, (shp1[1],shp1[0]))
    
    if kps_imm1 is None:
        # Create params for ShiTomasi corner detection
        feature_params = dict( maxCorners = 100,
                              qualityLevel = 0.3,
                              minDistance = 7,
                              blockSize = 7 )
        p0 = cv2.goodFeaturesToTrack(gray1, mask = None, **feature_params)
    else:
        p0 = np.round(kps_imm1.reshape((len(kps_imm1),1,2))).astype('float32')
    
    # Create some random colors
    color = np.random.randint(0,255,(10000,3))
    # Calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)
    # Select good points (if there is flow)
    good_new = p1[st==1]
    good_old = p0[st==1]
    
    #rescale if shapes are different
    if shp1!=shp2:
        scalex = shp2[1]/shp1[1]
        scaley = shp2[0]/shp1[0]
        good_new[:,0] = scalex * good_new[:,0]
        good_new[:,1] = scaley * good_new[:,1]    
    
    # Draw the tracks
    # Create a mask image for drawing purposes
    frame1 = np.copy(imm1)
    frame2 = np.copy(imm2)
    for i, (new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        print(frame1,(c,d),20,color[i].tolist(),-1)
        frame1 = cv2.circle(frame1,(int(c),int(d)),20,color[i].tolist(),-1)
        frame2 = cv2.circle(frame2,(int(a),int(b)),20,color[i].tolist(),-1)
    # Display the image with the flow lines
    #Save images with tracks
    cv2.imwrite('../results/'+data_folder+'/KLT_frame1_' + im_n1+ '.png', cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
    cv2.imwrite('../results/'+data_folder+'/KLT_frame2_' + im_n2+ '.png', cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
    
    plt.figure()
    plt.imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
    plt.figure()
    plt.imshow(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
    
    return good_new




def imgs_corners(im1, imm1, i, keypoints_path, how2get_kp, opening_4points, data_folder, im2=None, imm2=None, pp=None, save_op_corners=True):
    """

        Geting opening corners coordinates on images im1 (and im2)

    Args:
        im1 (list): list of input images view 1 for each facade
        im2 (list): list of input images view 2 for each facade
        imm1 (array): image for view one of facade
        imm2 (array): image for view two of facade
        i (int): facade id given by loop
        keypoints_path (str): path to keypoints
        how2get_kp (int): method to get opening corners to triangulate 0:from CNN detector+homography, 1: loading npy files, 2:clicking on two views, 3: v1 from npy+v2 with homography, 4:from CNN detector+homography
        opening_4points (dict): opening corners for each facade
        pp (int, optional): list with number of corners to be triangulated for each facade - used if how2get_kp==2
        data_folder (str): path to input data folder
        save_op_corners (bool, optional): _description_. Defaults to True.
    Returns:
        x1 (array): opening corners coordinates view1
        x2 (array): opening corners coordinates view2
        H (array): homography matrix to transform x1 to x2

    """
    v_H = 0
    #Depending of method selected on how2get_kp if findes the opening corner correspondences
    #for the 2 views
    if how2get_kp == 0:
        x1 = opening_4points[im1[i]][0]
        x1 = homography.make_homog(x1.T)
        np.save(keypoints_path + "x1_{}".format(i), x1)
        if im2 is not None:
            mask_facade = opening_4points[im1[i]][1]
            H = fast_homography(imm1,imm2,im1[i], im2[i], data_folder, mask_facade=mask_facade, save_im_kps=True)
            x2 = np.dot(H,x1)
            x2[0,:] = x2[0,:]/x2[2,:]
            x2[1,:] = x2[1,:]/x2[2,:]
            x2[2,:] = x2[2,:]/x2[2,:]
            np.save(keypoints_path + "x2_{}".format(i), x2)        
            v_H=1
        else:
            H=None
    elif how2get_kp == 1:
        #in case of have the corner points in a npy file
        x1 = np.load(keypoints_path + "x1_{}.npy".format(i))
        if im2 is not None:
            x2 = np.load(keypoints_path + "x2_{}.npy".format(i))
        H = None
    elif how2get_kp == 2:    
        #Selecting  correspondence points to be projected from image 1
        plt.figure()
        plt.imshow(imm1)
        print('Please click {} points'.format(pp[i]))
        x1 = np.array(pylab.ginput(pp[i],200))
        print('you clicked:',x1)
        plt.close()
        # make homogeneous and normalize with inv(K)
        x1 = homography.make_homog(x1.T)
        np.save(keypoints_path + "x1_{}".format(i), x1)

        if im2 is not None:
            #Selecting  points to be projected from image 2
            plt.figure()
            plt.imshow(imm2)
            print('Please click {} points'.format(pp[i]))
            x2 = np.array(pylab.ginput(pp[i],200))
            print('you clicked:',x2)
            plt.close()        
            # make homogeneous and normalize with inv(K)
            x2 = homography.make_homog(x2.T)
            np.save(keypoints_path + "x2_{}".format(i), x2)

        H = None
    
    elif how2get_kp ==3:
        x1 = np.load(keypoints_path + "x1_{}.npy".format(i))
        # make homogeneous and normalize with inv(K)
        x1 = homography.make_homog(x1.T)
        if im2 is not None:
            mask_facade = opening_4points[im1[i]][1]
            H = fast_homography(imm1,imm2,im1[i], im2[i], data_folder, mask_facade=mask_facade, save_im_kps=False)
            x2 = np.dot(H,x1)
            x2[0,:] = x2[0,:]/x2[2,:]
            x2[1,:] = x2[1,:]/x2[2,:]
            x2[2,:] = x2[2,:]/x2[2,:]
            v_H=1
        else:
            H = None
        
    elif how2get_kp ==4:
        #LKT to find other points
        x1 = opening_4points[im1[i]][0]        
        x1 = homography.make_homog(x1.T)
        np.save(keypoints_path + "x1_{}".format(i), x1)

        if im2 is not None:
            x2 = track_kps(data_folder, imm1,imm2,im1[i], im2[i], kps_imm1=x1)
            x2 = homography.make_homog(x2.T)           
            np.save(keypoints_path + "x2_{}".format(i), x2)

        v_H=0
        H = None       
    
    if save_op_corners:
        #view1 
        radius = int(imm1.shape[0]/100)
        im_c1 = np.copy(imm1)
        im_c1 = cv2.cvtColor(im_c1, cv2.COLOR_RGB2BGR)
        for c in x1.T:
            cv2.circle(im_c1, (int(c[0]),int(c[1])), radius=radius, color=(0,0,255), thickness=-1)
        cv2.imwrite('../results/' + data_folder + '/opening_corners_' + im1[i] + '.png', im_c1)
        #view2
        if im2 is not None:
            radius = int(imm2.shape[0]/100)
            im_c2 = np.copy(imm2)
            im_c2 = cv2.cvtColor(im_c2, cv2.COLOR_RGB2BGR)
            for c in x2.T:
                cv2.circle(im_c2, (int(c[0]),int(c[1])), radius=radius, color=(0,0,255), thickness=-1)
            cv2.imwrite('../results/' + data_folder + '/opening_corners_' + im2[i] + '.png', im_c2)    
    
    if im2 is not None:
        return x1, x2, H, v_H
    else:
        return x1, H, v_H

def initial_triangulation_2views(im1, im2, P, i, images_path, keypoints_path, how2get_kp, opening_4points, data_folder, pp):
    """

    This function give the initial triangulation of 2D point correspondences observed from
    2 different views. It takes the camera projection matrix and the 2D poind correspondences
    and find their 3D coordinates X.

    Args:
        im1 (list): list of images of view1 for each facade
        im2 (list): list of images of view2 for each facade
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

    print("Projection of openings in pics " + im1[i] + " and "\
        + im2[i])
    P1 = P[im1[i]]['P']
    P1_inid = P[im1[i]]['intrinsicId']
    P2 = P[im2[i]]['P']
    P2_inid = P[im2[i]]['intrinsicId']
    
    if os.path.isfile(images_path + "im1/" + im1[i] + '.jpg'):
        imm1 = np.array(Image.open(images_path + "im1/" + im1[i] + '.jpg'))
    elif os.path.isfile(images_path + "im1/" + im1[i] + '.JPG'):
        imm1 = np.array(Image.open(images_path + "im1/" + im1[i] + '.JPG'))
    else:
        imm1 = np.array(Image.open(images_path + "im1/" + im1[i] + '.png'))
    
    if os.path.isfile(images_path + "im2/" + im2[i] + '.jpg'):
        imm2 = np.array(Image.open(images_path + "im2/" + im2[i] + '.jpg'))
    elif os.path.isfile(images_path + "im2/" + im2[i] + '.JPG'):
        imm2 = np.array(Image.open(images_path + "im2/" + im2[i] + '.JPG'))
    else:
        imm2 = np.array(Image.open(images_path + "im2/" + im2[i] + '.png'))

    #Geting opening corners coordinates on images im1 and im2
    x1, x2, _, _ = imgs_corners(im1, imm1, i, keypoints_path, how2get_kp, opening_4points, data_folder, im2=im2, imm2=imm2, pp=pp)
    
    # triangulate inliers and remove points not in front of both cameras
    X = sfm.triangulate(x1,x2,P1,P2)

    return X

def initial_triangulation_1view_raycasting(im1, P, R, t, K, k_dist, P_type, poses, i, images_path, keypoints_path, how2get_kp, opening_4points, data_folder, pp, plane_params=None, dom=None):
    """

    This function give the initial triangulation of 2D points related with the opening corners (or points needed to be projected).
    It first find the points to be projected to 3D in the 2D image (view1) according with the method how2get_kp (CNN or manualy selected).
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
    
    P1_inid = P[im1[i]]['intrinsicId']
    print("Projection of openings in pic " + im1[i])
    
    if os.path.isfile(images_path + "im1/" + im1[i] + '.jpg'):
        imm1 = np.array(Image.open(images_path + "im1/" + im1[i] + '.jpg'))
    elif os.path.isfile(images_path + "im1/" + im1[i] + '.JPG'):
        imm1 = np.array(Image.open(images_path + "im1/" + im1[i] + '.JPG'))
    else:
        imm1 = np.array(Image.open(images_path + "im1/" + im1[i] + '.png'))
    
    
    #Geting opening corners coordinates on images im1 and im2
    x1, _, _ = imgs_corners(im1,imm1,i, keypoints_path, how2get_kp, opening_4points, data_folder, pp=pp)

    #Check if there are points to triangulate
    if len(x1.T)>0:
    
        #Correcting cordinates by lens distortion
        if P_type[P1_inid]=='radial3':
            x1_u = undistort_points(x1.T,K[P1_inid],k_dist[P1_inid]).T
            #undistort_img(imm1, K[P1_inid],k_dist[P1_inid])
        elif P_type[P1_inid]=='fisheye4':
            x1_u = undistort_points_fish(x1.T, K[P1_inid], k_dist[P1_inid]).T    
        x1 = x1_u

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
            LOD2_mesh_ = o3d.t.geometry.TriangleMesh.from_legacy(dom.LOD2.mesh)
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(LOD2_mesh_)
            rays = o3d.core.Tensor(rays_list, dtype=o3d.core.Dtype.Float32)
            ans = scene.cast_rays(rays)
            #print("Hello")
            dist2int = (ans['t_hit'].numpy()).reshape((-1,1))
            triangles_hit = ans['primitive_ids'].numpy()
            #Assign label to each opening according the plane the rays hit. If some of the rays do no hit the object, project the rays to a plane 
            # that contains the rays that did hit
            opening_plane_id = []
            for jj in range(int(len(rays_list)/4)):
                op_dist = dist2int[4*jj:4*jj+4].reshape(-1)
                op_hit = triangles_hit[4*jj:4*jj+4]
                op_rays = np.array(rays_list[4*jj:4*jj+4])
                plane_id_hit = dom.LOD2.plane_clusters[0][op_hit[np.where(op_dist!=np.inf)]] #planes that are hit by the rays 
                #Check if there ire rays that do not hit the object for each opening
                if np.sum(op_dist==np.inf)>0:
                    print("Some rays did not hit the obj")
                    #Select the clustered face where the rays fit
                    #If none of the rays hit the object, then do not create the opening
                    if sum(op_dist!=np.inf)==0: 
                        continue
                    
                    cluster_face_hit_params = dom.LOD2.plane_clusters[3][plane_id_hit[0]]
                    #THe rays that do not hit the mesh are projected to the plane of the clustered face
                    for r in np.where(op_dist==np.inf)[0]:
                        c_3D = op_rays[r][:3]
                        v_3D = c_3D + op_rays[r][3:]
                        r_int_plane = (sfm.intersec_ray_plane(c_3D[:3], v_3D[:3], cluster_face_hit_params)).reshape(-1)
                        op_dist[r] = np.linalg.norm(r_int_plane-c_3D)
                #Assigning label to opening accoring to plane that hit the most
                unique, counts = np.unique(plane_id_hit, return_counts=True)
                most_hit = unique[np.where(counts==np.max(counts))] 
                if len(most_hit)>1:
                    check = False
                    for ii in most_hit:
                        if ii in opening_plane_id:
                            opening_plane_id.append(ii)
                            check = True
                            break
                    if not check:
                        opening_plane_id.append(most_hit[0])
                else:
                    opening_plane_id.append(most_hit[0])
                
                    
            dist2int = np.concatenate((dist2int, dist2int, dist2int), axis = 1)
            rays = rays.numpy()
            X = rays[:,:3] + dist2int*rays[:,3:]        
            #print("Bye")
            dist2int = dist2int[np.where(X!=np.inf)] 
            X = X[np.where(X!=np.inf)]
            X = X[np.where(X!=-np.inf)]
            X = X.reshape((-1,3))
            

            #Create opening objects and assign to the domain the openings attribute (appending to old list)
            for jj in range(int(len(X)/4)):
                #get opening_id based on last
                if len(dom.openings)==0:
                    opening_id = 0
                else:
                    opening_id=dom.openings[-1].id+1
                op = Opening()
                op.id = opening_id
                op.plane = opening_plane_id[jj]
                op.coord = X[4*jj:4*jj+4]
                op.d2int = dist2int[4*jj:4*jj+4]
                dom.openings.append(op)

    else:
        X = np.empty(shape=[0,3])

    return X


def op_projector(data_folder, images_path, keypoints_path, polyfit_path, intrinsic, poses, structure, how2get_kp,\
                 im1, opening_4points, im2=None, two_view_triang=False, pp=None, dense=False, ctes=[.05, .8, .1, .3], ray_int_method="casting"):
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
    #Initial time t0
    #t0 = time.time()
    #Crating domain Damage Augmented Digital Twin DADT
    DADT = domain()
    #Load LOD2 obj file to domain - Assigning LOD2 to DADT domain
    DADT.load_LOD2(dense, polyfit_path)    
    #Initial cluster LOD2 triangle elements that are in the same plane and finde plane parameters
    DADT.LOD2.get_plane_clusters()
    #Find LOD2 oriented bounding box
    print("Here")
    #Get LOD2 boundary as lineset (to define facades, orientation, and simplify mesh?)
    DADT.LOD2.get_boundary_line_set(plot=False)
    #Decimate LOD2 model to reduce triangles number
    DADT.LOD2.decimate()
    #Update cluster LOD2 triangle elements that are in the same plane and find plane parameters
    DADT.LOD2.get_plane_clusters()
    #Get boundary for LOD2 planes (to help defining orientation maybe)
    DADT.LOD2.get_planes_contour()
    

    #Reading camera poses information
    P, K, R, t, k_dist, P_type = camera_matrices(intrinsic, poses, return_Rt=True)

    if ray_int_method == "plane":
        #Detecting facades if it was not done during opening detection
        if opening_4points is None:
            fac_prediction = fac_segmentation(data_folder, images_path+'im1/', out_return=True)
    
    #Loop through each par of pictures given by im1 and im2
    cc_vv = 1 #vertices counter. Helper to identify vertices in generated faces (all)
    cc_vv_c = 1 #vertices counter. Helper to identify vertices in generated cracks (all)
    for i in range(len(im1)): 
        #i=2
        #Create opening keypoints folder if does not exist in data folder
        check = os.path.isdir(keypoints_path)
        if not check:
            os.makedirs(keypoints_path)
        #Find 3D coordinates of openings keypoints using either two view or one view methods    
        if two_view_triang:
            X = initial_triangulation_2views(im1, im2, P, i, images_path, keypoints_path, how2get_kp, opening_4points, data_folder, pp) 
        else:
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
                X = initial_triangulation_1view_raycasting(im1, P, R, t, K, k_dist, P_type, poses, i, images_path, keypoints_path, how2get_kp, opening_4points, data_folder, pp, plane_params=plane_params) 
            elif ray_int_method == "casting":
                X = initial_triangulation_1view_raycasting(im1, P, R, t, K, k_dist, P_type, poses, i, images_path, keypoints_path, how2get_kp, opening_4points, data_folder, pp, dom=DADT)#LOD2_triangle_clusters, LOD2_cluster_plane_params, LOD2_mesh=LOD2_mesh) 
                    
            X = np.concatenate((X, np.ones((len(X),1))), axis=1).T
                   
        
        #Find a plane paralel to facade with a normal similar to mean of normals
        #of all openings. Project corners to that plane. 
        if two_view_triang:
            X, _, faces_normals, normal_op_plane = proj_op2plane(polyfit_path, X, dense)
        
            #CLEANING 2 - SAME WIDTH AND HEIGHT TO WINDOWS DOORS
            #Taking corners X to a local plane. Xl
            Xl, T = open2local(X, faces_normals, normal_op_plane)
            #Aligning the width and height of the openings (Aligment 1 --> to linear regression model).
            Xl_al = op_aligning1(Xl, cte = ctes[0])
        
            #CLEANING 2.1: aligning  each opening
            #Aligning the width and height of the openings (Aligment 2 --> same width and height)
            Xl_al2 = op_aligning2(Xl_al, cte = ctes[1])
            
            #Equalizing areas
            Xl_al3 = op_aligning3(Xl_al2, cte1 = ctes[2], cte2 = ctes[3])
            
            #Taking to global coordinates again
            X_al = np.dot(np.linalg.inv(T),Xl_al3) 
       
            #Check if directory exists, if not, create it
            check_dir = os.path.isdir('../results/' + data_folder)
            if not check_dir:
                os.makedirs('../results/' + data_folder) 
            
            #Writing an .obj file with information of the openings for each pics pair
            f = open('../results/' + data_folder + "/openings{}.obj".format(i), "w")
            
            for l in range(len(X_al.T)):
                f.write("v {} {} {}\n".format(X_al.T[l][0],X_al.T[l][1],X_al.T[l][2]))
            c_v = 1 #vertices counter. Helper to identify vertices in generated faces
            for j in range(int(len(X_al.T)/4)):
                f.write("f {} {} {}\n".format(c_v,c_v+1,c_v+2))
                f.write("f {} {} {}\n".format(c_v+1,c_v+2,c_v+3))
                c_v += 4
            f.close()
        
            #Writing an .obj file with information of the openings for all of them
            f = open('../results/' + data_folder + "/openings.obj".format(i), "a")
            for l in range(len(X_al.T)):
                f.write("v {} {} {}\n".format(X_al.T[l][0],X_al.T[l][1],X_al.T[l][2]))
        
            for j in range(int(len(X_al.T)/4)):
                f.write("f {} {} {}\n".format(cc_vv,cc_vv+1,cc_vv+2))
                f.write("f {} {} {}\n".format(cc_vv+1,cc_vv+2,cc_vv+3))
                cc_vv += 4
            f.close()
    
    if not two_view_triang:
        #Assign openings to each LOD2 plane
        DADT.assign_openings2planes()
        #Detect redundat openings to ignore them during post-processing
        DADT.detect_redundant_openings(parameter="camera-size")
        #Regularize openings
        DADT.regularize_openings(ctes)
        DADT.save_openings(data_folder)
        print("END PROJECTION")

    #final time tf
    #tf = time.time() - t0
    #print("-----The time for projecting openings to the LOD2 is: {}s -----".format(tf))

    return DADT
