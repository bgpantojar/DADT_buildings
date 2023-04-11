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
#from PIL import Image
import numpy as np
import pylab as pylab
import os
import matplotlib.pyplot as plt
#import camera
import cv2
#from projection_tools import *
import open3d as o3d
#from classes import *
from skimage.morphology import skeletonize
#from tqdm import tqdm
import json
from utils_geometry import read_LOD


def compute_T_for_x_to_3D(R,t):
    """
    Given a Camera object, it returns the transformation matrix needed
    to apply to the camera points (camera center, image plane features...)
    to 3D
    ----------
    """
    Rs = R.T
    ts = -R.T @ t
    Rs = np.row_stack((Rs,np.zeros(3)))
    ts = np.concatenate((ts.reshape((3,1)), np.ones((1,1))))
    T = np.concatenate((Rs, ts), axis=1)
    
    return T

def camera_matrices(intrinsic, poses, return_Rt=False):
    """
    Calculate Camera Matrix using intrinsic and extrinsic parameters P = K[R|t]

    Args:
        intrinsic (dict) : camera intrinsic information
        poses (dict) : camera poses information
        return_Rt (bool, opt) : True if want to return rotation matrix and translation vector

    Returns:
        P (list): camera matrices' list
        K (list): intrinsic matrices' list
        k_dist (list): distorsion parameters' list
    """
    

    K = {}
    k_dist = {}
    for ii in intrinsic:
        #Intrinsic Parameters
        f = float(intrinsic[ii]["pxFocalLength"])
        px = float(intrinsic[ii]["principalPoint"][0])
        py = float(intrinsic[ii]["principalPoint"][1])
        k_dist[ii] = np.ndarray.astype(np.array((intrinsic[ii]["distortionParams"])),float)
        
        K[ii]= np.array([[f,0.,px],
                         [0.,f,py],
                         [0.,0.,1.]])
    
    #Extrinsic Parameters
    R = {}
    t = {}
    C = {}
    
    for key in poses:
        R[key] = (np.float_(poses[key]["rotation"])).reshape((3,3)).T
        C = np.float_(poses[key]["center"]).reshape((3,1))
        t[key] = np.dot(-R[key],C)
    
    #List with camera matrices p
    P = {}
    for key in poses:
        P[key] = {}
        P[key]['P'] = np.dot(K[poses[key]['intrinsicId']],np.concatenate((R[key],t[key]),axis=1))
        P[key]['intrinsicId'] = poses[key]['intrinsicId']
        
    if return_Rt:        
        return P, K, R, t, k_dist
    else:
        return P, K, k_dist

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

def simple_triangulation_1view_raycasting(data_folder, im1, x1, R, t, K, poses, i, obj_path):
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

    #Make 2D coordinates, homogeneus
    x1 = (np.concatenate((x1, np.ones((len(x1),1))), axis=1)).T

    print("Projection in pic " + im1[i])
    
    ## triangulate opening corners as intersection of plane and ray
    R_view = R[im1[i]]
    t_view = t[im1[i]]
    #Transformation matrix to transform image plane at pose place in 3D
    T_x3D = compute_T_for_x_to_3D(R_view, t_view)
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
       
        ray_dir = v_3D[:3] - c_3D[:3]
        ray_dir = ray_dir/np.linalg.norm(ray_dir)
        rays_list.append([c_3D[0][0], c_3D[1][0], c_3D[2][0], ray_dir[0][0], ray_dir[1][0], ray_dir[2][0]])
    
    if obj_path.split('/')[-2] != "polyfit":
        mesh_obj =  o3d.io.read_triangle_mesh(obj_path)
    else:
        LOD2_vertices, LOD2_triangles = read_LOD(obj_path) #!own reading function
        mesh_obj = o3d.geometry.TriangleMesh()
        mesh_obj.vertices = o3d.utility.Vector3dVector(LOD2_vertices) #!own reading function
        mesh_obj.triangles = o3d.utility.Vector3iVector(LOD2_triangles) #!own reading function
        mesh_obj.compute_vertex_normals()        


    obj_mesh_ = o3d.t.geometry.TriangleMesh.from_legacy(mesh_obj)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(obj_mesh_)
    rays = o3d.core.Tensor(rays_list, dtype=o3d.core.Dtype.Float32)
    ans = scene.cast_rays(rays)
    dist2int = (ans['t_hit'].numpy()).reshape((-1,1))
    
    cr_dist = dist2int.reshape(-1)
    #Check if there ire rays that do not hit the object for each opening
    if np.sum(cr_dist==np.inf)>0:
        print("Some rays did not hit the obj")
            
    dist2int = np.concatenate((dist2int, dist2int, dist2int), axis = 1)
    rays = rays.numpy()
    X = rays[:,:3] + dist2int*rays[:,3:]        
    id_cr_intersect = np.where(cr_dist!=np.inf)
    X = X[id_cr_intersect]
    X = X.reshape((-1,3))

    return X, id_cr_intersect

def simple_cr_detector(data_folder, images_path):

    list_masks = os.listdir("../results/"+data_folder+"/")
    list_masks = [m for m in list_masks if m.startswith("crack_filtered_")]
    list_masks = [m for m in list_masks if m.endswith("mask.png")]
    list_masks.sort()
    crack_prediction_bin = {}
    for m in list_masks:
        key = m[15:-9]
        crack_prediction_bin[key] = cv2.imread("../results/"+data_folder+"/crack_"+key+"_mask.png")

    #Getting original images
    list_images = os.listdir(images_path)
    list_images.sort()
    
    images = {}
    for img in list_images:
        images[img[:-4]] = cv2.imread(images_path + img)
    
    pred_bin_resized ={}
    for key in images:
        pr_bn_rsz = crack_prediction_bin[key]/255
        pred_bin_resized[key] = pr_bn_rsz 
        
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
        y_coord = y_coord.reshape(x_coord.shape[0],1)
        cracks_information[key]['points'] = np.hstack((x_coord, y_coord))
  
    return cracks_information, pred_bin_resized_skeleton


def simple_crack_projector(data_folder, images_path, obj_path, intrinsic, poses, im1, kinematics=False, load_cracks2d=True):

    #Check if directory exists, if not, create it
    check_dir = os.path.isdir('../results/' + data_folder)
    if not check_dir:
        os.makedirs('../results/' + data_folder) 
    cc_vv=1
    num_pts=0

    #Getting crack information
    if not load_cracks2d:
        crack_information, pred_bin_resized_skeleton = simple_cr_detector(data_folder, images_path)

    #Reading camera poses information
    P, K, R, t, k_dist = camera_matrices(intrinsic, poses, return_Rt=True)

    #Loop through each par of pictures given by im1 and im2
    if kinematics:
        colors_full = {}
        colors_full["kin_n"] = np.empty(shape=[0,3])
        colors_full["kin_t"] = np.empty(shape=[0,3])
        colors_full["kin_tn"] = np.empty(shape=[0,3])

    for i, key in enumerate(im1): 
        #i=2
        #2D points to be projected
        if load_cracks2d:
            x1 = np.load("../results/" + data_folder + "/cracks2d_{}.npy".format(key)).astype('int')
        else:
            x1 = crack_information[im1[i]]['points'] #THIS CAN BE SAVED IN THE DOMAIN INSTEAD! to save properly the kinematics for texuture points
        if len(x1.T)==0:
            continue
        X, id_cr_intersect = simple_triangulation_1view_raycasting(data_folder, im1, x1, R, t, K, poses, i, obj_path)
        
        #Save cracks    
        #Creating rgb colors for saving kinematics
        rgb = [(0,0,0) for pt in X]

        #name_file
        name_file = obj_path.split('/')[-2]
        #Writing an .obj file with information of the openings for each pics pair
        f = open('../results/' + data_folder + "/" + name_file + "_cracks_{}_{}.ply".format(i, key), "w") #the colormap scale for this need to be modified for n and t
        f.write("ply\n\
        format ascii 1.0\n\
        element vertex {}\n\
        property float x\n\
        property float y\n\
        property float z\n\
        property uchar red\n\
        property uchar green\n\
        property uchar blue\n\
        end_header\n".format(X.shape[0]))

        for ii, pt in enumerate(X):
            xx = np.around(pt[0],decimals=5)
            yy = np.around(pt[1],decimals=5)
            zz = np.around(pt[2],decimals=5)
            f.write("{} {} {} {} {} {}\n".format(xx,yy,zz,int(255*rgb[ii][0]),int(255*rgb[ii][1]),int(255*rgb[ii][2])))
        f.close()

        num_pts += len(X)
        f = open('../results/' + data_folder + "/" + name_file + "_cracks.ply", "a")
        for ii, pt in enumerate(X):
            xx = np.around(pt[0],decimals=5)
            yy = np.around(pt[1],decimals=5)
            zz = np.around(pt[2],decimals=5)
            f.write("{} {} {} {} {} {}\n".format(xx,yy,zz,int(255*rgb[ii][0]),int(255*rgb[ii][1]),int(255*rgb[ii][2])))
        f.close()

        if i==len(im1)-1:
            f = open('../results/' + data_folder + "/" + name_file + "_cracks.ply", "r")
            read_ply = f.readlines()
            read_ply.insert(0,
            "ply\n\
            format ascii 1.0\n\
            element vertex {}\n\
            property float x\n\
            property float y\n\
            property float z\n\
            property uchar red\n\
            property uchar green\n\
            property uchar blue\n\
            end_header\n".format(num_pts))
            f.close()
            
            f = open('../results/' + data_folder + "/" + name_file + "_cracks.ply", "w")
            f.writelines(read_ply)
            f.close()
            print(num_pts)
        
        if kinematics:
            kin_key = ["kin_n", "kin_t", "kin_tn"]
            for kk in kin_key:
                path_origin = '../results/' + data_folder + "/" + name_file + "_cracks_{}_{}.ply".format(i, key)
                path_destiny = '../results/' + data_folder + "/cracks_{}_{}_{}.ply".format(i, key, kk)
                pcd_o = o3d.io.read_point_cloud(path_origin)
                pcd_d = o3d.io.read_point_cloud(path_destiny)
                colors = np.array(pcd_d.colors)[id_cr_intersect]
                colors_full[kk] = np.concatenate((colors_full[kk], colors), axis=0)
                #pcd_o.colors = pcd_d.colors
                #pcd_d.points = pcd_o.points
                pcd_o.colors = o3d.utility.Vector3dVector(colors)
                path_save = '../results/' + data_folder + "/" + name_file + "_cracks_{}_{}_{}.ply".format(i, key, kk)
                o3d.io.write_point_cloud(path_save, pcd_o, write_ascii=True)
                #o3d.io.write_point_cloud(path_save, pcd_d, write_ascii=True)

    if kinematics:
        kin_key = ["kin_n", "kin_t", "kin_tn"]
        for kk in kin_key:
            path_origin = '../results/' + data_folder + "/" + name_file + "_cracks.ply"
            path_destiny = '../results/' + data_folder + "/cracks_{}.ply".format(kk)
            pcd_o = o3d.io.read_point_cloud(path_origin)
            pcd_d = o3d.io.read_point_cloud(path_destiny)
            pcd_o.colors= o3d.utility.Vector3dVector(colors_full[kk])
            path_save = '../results/' + data_folder + "/" + name_file + "_cracks_{}.ply".format(kk)
            o3d.io.write_point_cloud(path_save, pcd_o, write_ascii=True)


def create_material_library(material_path, type="oop"):

    f = open(material_path + type + ".mtl", "w")
    f.write("newmtl {}\n".format(type))
    f.write("Ka 0.0000 0.0000 0.0000\n")
    if type=="LOD":
        f.write("Kd 0.4000 0.4000 0.4000\n")
    elif type=="cracks":
        f.write("Kd 1.0000 0.2000 0.2000\n")
    elif type=="roof":
        f.write("Kd 0.2000 1.0000 0.2000\n")
    elif type=="oop":
        f.write("Kd 0.0000 0.2000 1.0000\n")
    elif type=="graffiti":
        f.write("Kd 0.2000 1.0000 1.0000\n")
    else:
        print("No material {} created as type does not exist".format(type))
    f.write("Ks 1.0000 1.0000 1.0000\n")
    f.write("Tf 0.0000 0.0000 0.0000\n")
    f.write("d 1.0000\n")
    f.write("Ns 0.0000")
    f.close()

def simple_polygon_projector(data_folder, images_path, obj_path, intrinsic, poses, im1, label_polygon = "oop", min_trans=True):

    #Function to project the manually segmented polygons to obj files using raycasting
    plt.close()
    #Check if directory exists, if not, create it
    check_dir = os.path.isdir('../results/' + data_folder)
    if not check_dir:
        os.makedirs('../results/' + data_folder) 
    cc_vv=1
    num_pts=0

    #Reading camera poses information
    P, K, R, t, k_dist = camera_matrices(intrinsic, poses, return_Rt=True)

    #List images
    img_list = os.listdir(images_path)

    for i, key in enumerate(im1):
        
        #read image
        img_name = [i_n for i_n in img_list if i_n.startswith(key)][0]
        img = cv2.imread(images_path + "/" + img_name) 

        #Check if polygons were already assignated
        path0 = "../results/"+data_folder+'/{}_{}_vertices.npy'.format(label_polygon, key)
        check0 = os.path.isfile(path0)
        path1 = "../results/"+data_folder+'/{}_{}_vertices_number.npy'.format(label_polygon, key)
        check1 = os.path.isfile(path1)

        if check1 and check0:
            poly_vert = np.load(path0)
            vert_numb_list = list(np.load(path1))
        else:
            print("Check the image to select the number of polygons you want to project of the class " + label_polygon)   
            manager = plt.get_current_fig_manager()
            manager.full_screen_toggle()
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.show()
            num_polygons = input("Number of polygons to be projected to 3D model of the class " + label_polygon)
            num_polygons = int(num_polygons)
            plt.close()#
            if num_polygons==0:
                continue
            #Selecting vertices for the quantity of polygons selected
            poly_vert = np.empty(shape=[0,2])
            vert_numb_list = []

            for ii in range(num_polygons):
                vert_numb = input("Write number of vertices of your polygon {} for damage {}".format(ii, label_polygon))
                vert_numb_list.append(int(vert_numb))
                plt.figure()
                manager = plt.get_current_fig_manager()
                manager.full_screen_toggle()
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                print('Please click {} points that represents the polygon vertices of your object {}'.format(vert_numb, label_polygon))
                v =  np.array(pylab.ginput(int(vert_numb),200)).astype('int')
                print('you clicked:', v)
                plt.close()
                poly_vert = np.concatenate((poly_vert, v), axis=0)

        x1 = poly_vert
        X, id_cr_intersect = simple_triangulation_1view_raycasting(data_folder, im1, x1, R, t, K, poses, i, obj_path)
        if min_trans:
            #Translate objects for visual purposes
            X0, X1, X2 = X[0], X[1], X[2]
            aa = X0 - X1
            aa = aa/np.linalg.norm(aa)
            bb = X2 - X1
            bb = bb/np.linalg.norm(bb)
            n_v = np.cross(bb,aa)
            n_v = n_v/np.linalg.norm(n_v)
            X = X+.004*n_v        #!  

        obj_name = obj_path.split('/')[-2]
        f = open("../results/" + data_folder + "/{}_{}_{}.obj".format(obj_name,label_polygon, key), "w")
        f.write("mtllib {}.mtl\n".format(label_polygon))
        f.write("usemtl {}\n".format(label_polygon))
        check = os.path.isfile("../results/" + data_folder + "/{}.mtl".format(label_polygon))
        if not check:
            create_material_library("../results/" + data_folder + "/", type=label_polygon)

        for X_ in X:
            f.write("v {} {} {}\n".format(X_[0],X_[1],X_[2]))           
    
        v_c = 1
        for jj in vert_numb_list:
            f.write('f '+ ' '.join(str(v_c+i) for i in range(jj))+'\n')
            v_c+=jj
        f.close()

        #Saving npy files with the poligons in each image
        np.save("../results/"+data_folder+'/{}_{}_vertices.npy'.format(label_polygon, key), poly_vert)
        np.save("../results/"+data_folder+'/{}_{}_vertices_number.npy'.format(label_polygon, key), np.array(vert_numb_list))

        #Saving overlayed image
        poly_mask = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype='uint8')
        v_c = 0
        for jj in vert_numb_list:            
            cv2.fillPoly(poly_mask, [x1[v_c:v_c+jj].astype('int')], 1)
            v_c+=jj
        img_ov = cv2.addWeighted(img, .8, poly_mask*255, 0.9, 0)
        cv2.imwrite("../results/"+data_folder+'/{}_{}_overlayed.jpg'.format(label_polygon, key), img_ov)

        