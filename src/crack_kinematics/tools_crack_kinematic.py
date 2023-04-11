#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 11:33:08 2021

@author: pantoja
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.neighbors import NearestNeighbors
from matplotlib import cm
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
import os
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
import json
import numpy as np
import pylab
import scipy.stats as stats
import scipy.ndimage as spim

def H_from_transformation(H_op, H_type):
    """
    According transformation it returns H transformation given the optimal
    H parameters

    Args:
        H_op (array): Optimal transformation parameters as array
        H_type (_type_): Transformation type: euclidean, similarity, affine, projective

    Returns:
        H (array): 3x3 transformation matrix_
    """
    
    if H_type == "euclidean":
            theta = H_op[0]
            tx = H_op[1]
            ty = H_op[2]
            H = np.array([[np.cos(theta), -np.sin(theta), tx],
                 [np.sin(theta), np.cos(theta), ty],
                 [0,0,1]])
    elif H_type == "similarity":
        theta = H_op[0]
        tx = H_op[1]
        ty = H_op[2]
        s = H_op[3]
        H = np.array([[s*np.cos(theta), -np.sin(theta), tx],
             [np.sin(theta), s*np.cos(theta), ty],
             [0,0,1]])
    elif H_type == "affine":
        theta = H_op[0]
        tx = H_op[1]
        ty = H_op[2]
        m = H_op[3]
        sx = H_op[4]
        sy = H_op[5]
        A = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]]) @ np.array([[1,m],[0,1]]) @ np.array([[sx, 0],[0,sy]])
        H = np.eye(3)
        H[:2,:2] = A
        H[0,2] = tx
        H[1,2] = ty
    elif H_type == "projective":
        H = np.concatenate((H_op, np.array([1]))).reshape((3,3))
        
    return H


def cropp_resize_images(data_path, list_masks):
    """
    Crops and resize images to 256x256 given in a list selecting bounding boxes.
    Used to create toy examples

    Args:
        data_path (str): Data folder path
        list_masks (list): List with list of masks names as str_
    """
    
    #Check if results directory exists, if not, create it
    check_dir = os.path.isdir(data_path + "c_r/")
    if not check_dir:
        os.makedirs(data_path + "c_r/")
    
    for mask_name in list_masks:
        crack = cv2.imread(data_path + "/original_mask/"+mask_name)

        #Interact with user to click four points to find bounding boxes    
        print('Please click {} points that represent bounding boxes'.format(2))
        
        plt.figure()
        plt.imshow(cv2.cvtColor(crack, cv2.COLOR_RGB2BGR))
        fc = np.array(pylab.ginput(2,200))        
        plt.close()
        
        #Create bounding boxes from four_corners information
        tl = np.array([np.min(fc[:,0]),np.min(fc[:,1])])
        br = np.array([np.max(fc[:,0]),np.max(fc[:,1])])
        bb = [tl,br]
    
        #Saving cropped image
        bb_image = crack[int(bb[0][0]):int(bb[1][0]), int(bb[0][1]):int(bb[1][1]), :]
        resized = cv2.resize(bb_image, (256,256), interpolation = cv2.INTER_AREA)
        resized = resized>0
        resized = resized*255
        
        cv2.imwrite(data_path + "c_r/" + mask_name , resized)
        
        print("saved as " + data_path + "c_r/" + mask_name)


def convolve(image, kernel):
    """
    Convolution operation to find intersections

    Args:
        image (array): Image array to be convolved
        kernel (array): Kernel to convolve image
    """
    
	#spatial dimensions of the image and kernel
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]
	# output image with "pad" the borders of the input image so the spatial
	# size (i.e., width and height) are not reduced
    pad = (kW - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    output = np.zeros((iH, iW), dtype="float32")
    # loop over the input image, "sliding" the kernel across each (x, y)-coordinate from left-to-right and top to bottom
    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
			# extract the ROI of the image by extracting the *center* region of the current (x, y)-coordinates dimensions
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
			# perform the actual convolution by taking the element-wise multiplicate between the ROI and the kernel, then summing the matrix
            k = (roi * kernel).sum()
			# store the convolved value in the output (x,y) coordinate of the output image
            output[y - pad, x - pad] = k
            
    # rescale the output image to be in the range [0, 255]
    output = (output * 255).astype("uint8")
    	
    return output


def find_dir_nor(mask_name,skeleton, dir_save='../results', k_n=10, mask=None, plot=False):
    """
    It computes the crack skeleton directions using PCA and nearest neighbors.

    Args:
        mask_name (str): Name of binary crack pattern image
        skeleton (array): array with x,y skeeleton image coordinates
        dir_save (str, optional): Directory where results are saved. Defaults to '../results'.
        k_n (int, optional): Number of neighbors to compute directions. Defaults to 10.
        mask (array, optional): Image with the binary mask of the crack pattern. Defaults to None.
        plot (bool, optional): True to plot the results. Defaults to False.

    Returns:
        contour_dir_nor (array): Array with direction information of skeleton
    """
    
    if len(skeleton)>0:
        #NearestNeighbors for given skeleton
        nbrs = NearestNeighbors(n_neighbors=k_n, algorithm='ball_tree').fit(skeleton)
        distances, indices = nbrs.kneighbors(skeleton)
        
        #Arrays to save directions
        contour_dir_nor = np.zeros((len(skeleton),8)) #point, dir, norm, norm_angle, dir_angle
        contour_dir_nor[:,:2] = np.copy(skeleton)

        #Compute skeleton direction for each of its points using PCA with the k-neighboors
        for i in range(len(indices)):
            X = np.copy(skeleton[indices[i]])
            pca = PCA(n_components=2)
            pca.fit(X)
            comp = pca.components_
            direction = comp[0]
            normal = comp[1]
            smooth = 1e-5 #avoid zero division
            #counterwise angle of normal vector
            norm_angle = np.round(np.arctan(normal[1]/(normal[0]+smooth))*180/np.pi, 2)
            dir_angle = np.round(np.arctan(direction[1]/(direction[0]+smooth))*180/np.pi,2)
            
            #Assingning values to array
            if normal[0]*normal[1]>=0:
                if normal[0]<0:
                    contour_dir_nor[i,2:4] = np.array([-np.abs(direction[0]),np.abs(direction[1])])
                    contour_dir_nor[i,4:6] = -normal #to follow right hand rule 
                else:
                    contour_dir_nor[i,2:4] = np.array([-np.abs(direction[0]),np.abs(direction[1])])
                    contour_dir_nor[i,4:6] = normal
            else:
                if normal[0]<0:
                    contour_dir_nor[i,2:4] = -np.abs(direction)
                    contour_dir_nor[i,4:6] = normal                
                else:
                    contour_dir_nor[i,2:4] = -np.abs(direction)
                    contour_dir_nor[i,4:6] = -normal 
            contour_dir_nor[i,6] = norm_angle
            contour_dir_nor[i,7] = dir_angle
            
        if plot:
            X = contour_dir_nor[:,0]
            Y = contour_dir_nor[:,1]
            U = contour_dir_nor[:,4]
            V = contour_dir_nor[:,5]
            Ud = contour_dir_nor[:,2]
            Vd = contour_dir_nor[:,3]
            fig, ax = plt.subplots()
            plt.plot(skeleton[:,0],skeleton[:,1], 'y.')
            if mask is not None:
                ax.imshow(mask, 'gray')
                ax.quiver(X, Y, -V, -U)  
                ax.quiver(X, Y, Vd, Ud)
            else:
                ax.quiver(X, Y, U, V)
                ax.quiver(X, Y, Ud, Vd)
            
            plt.axis('scaled')
            plt.show()
            
            if mask is not None:
                capt = "_mask"
            else:
                capt = ""
            fig.savefig(dir_save+mask_name[:-4]+'_skeleton_dir_nor'+capt+'.png', bbox_inches='tight', pad_inches=0)
            fig.savefig(dir_save+mask_name[:-4]+'_skeleton_dir_nor'+capt+'.pdf', bbox_inches='tight', pad_inches=0)
            plt.close()
    else:
        #Arrays to save directions
        contour_dir_nor = np.zeros((len(skeleton),8)) #point, dir, norm, norm_angle, dir_angle
    
    return contour_dir_nor

def find_skeleton_corners(skl):
    """
    Find the harris skeleton corners

    Args:
        skl (array): Image of binary crack pattern

    Returns:
        skl_coorner: skeleton corners
    """
    
    skl_corner = cv2.cornerHarris(skl,2,3,0.04)
    ret, skl_corner = cv2.threshold(skl_corner,0.8*skl_corner.max(),255,0)
    
    return skl_corner>0


def conv_dice(image, kernel):
    """
    Convolution operation to find intersections
    
    Args:
        image (array): Binary image array to be convolved
        kernel (array): Kernel to convolve image
    Returns:
        output (array): image with "pad" the borders of the input image so the spatial
    """
    
	#spatial dimensions of the image and kernel
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]
    pad = (kW - 1) // 2
    image = (cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0))>0
    output = np.zeros((iH, iW), dtype="float32")
    # loop over the input image, "sliding" the kernel across each (x, y)-coordinate from left-to-right and top to bottom
    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
			# extract the ROI of the image by extracting the *center* region of the current (x, y)-coordinates dimensions
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
			# perform the actual convolution by taking the element-wise multiplicate between the ROI and the kernel, then summing the matrix
            k = (roi * kernel).sum()
            u = (roi + kernel).sum()
            dice_roi = 2*k/u
			# store the convolved value in the output (x,y)- coordinate of the output image
            output[y - pad, x - pad] = dice_roi
            
    output = (output == 1).astype("uint8")
    	
    return output

def find_skeleton_intersections(skl):
    """
    Use kernels of possible intersections in skeleton and find a dice score conv operations
    the result are the pixels where the kernel and roi give a dice_score == 1.

    Args:
        skl (array): Binary image with the crack skeleton

    Returns:
        inters (array): Binary image with the skeleton intersections
    """     

    #Group of kernels
    #follows https://legacy.imagemagick.org/Usage/morphology/#thin_pruning
    k = []
    #T like
    k0 = np.array([[1,0,1],[0,1,0],[0,1,0]])
    k1 = np.array([[0,1,0],[0,1,1],[1,0,0]])
    k2 = np.array([[0,0,1],[1,1,0],[0,0,1]])
    k3 = np.array([[1,0,0],[0,1,1],[0,1,0]])
    k4 = np.array([[0,1,0],[0,1,0],[1,0,1]])
    k5 = np.array([[0,0,1],[1,1,0],[0,1,0]])
    k6 = np.array([[1,0,0],[0,1,1],[1,0,0]])
    k7 = np.array([[0,1,0],[1,1,0],[0,0,1]])
    k8 = np.array([[1,0,0],[0,1,0],[1,0,1]])
    k9 = np.array([[1,0,1],[0,1,0],[1,0,0]])
    k10 = np.array([[1,0,1],[0,1,0],[0,0,1]])
    k11 = np.array([[0,0,1],[0,1,0],[1,0,1]])
    #X like
    k12 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    k13 = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    k = [k0,k1,k2,k3,k4,k5,k6,k7,k8,k9,k10,k11,k12,k13]
    inters = np.zeros_like(skl)
    for ki in k:
        inters = inters + conv_dice(skl, ki)
    
    inters = inters>0
    
    return inters

def _getnodes(Xin, extremes=True):
    # hits
    structures = []
 
    if extremes:
        structures.append([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
        structures.append([[0, 1, 0], [0, 1, 0], [0, 0, 0]])
        structures.append([[0, 0, 1], [0, 1, 0], [0, 0, 0]])
        structures.append([[0, 0, 0], [0, 1, 1], [0, 0, 0]])
        structures.append([[0, 0, 0], [0, 1, 0], [0, 0, 1]])
        structures.append([[0, 0, 0], [0, 1, 0], [0, 1, 0]])
        structures.append([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
        structures.append([[0, 0, 0], [1, 1, 0], [0, 0, 0]])
 
    # structures.append([[1, 1, 1], [0, 1, 1], [1, 0, 0]])
    # structures.append([[1, 1, 1], [1, 1, 0], [0, 0, 1]])
    # structures.append([[1, 0, 0], [0, 1, 1], [1, 1, 1]])
    # structures.append([[0, 0, 1], [1, 1, 0], [1, 1, 1]])
 
    crossings = [[0, 1, 0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 1, 0, 0, 1], [1, 0, 0, 1, 0, 1, 0, 0],
                 [0, 1, 0, 0, 1, 0, 1, 0], [0, 0, 1, 0, 0, 1, 0, 1], [1, 0, 0, 1, 0, 0, 1, 0],
                 [0, 1, 0, 0, 1, 0, 0, 1], [1, 0, 1, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0, 1],
                 [0, 1, 0, 1, 0, 0, 0, 1], [0, 1, 0, 1, 0, 1, 0, 0], [0, 0, 0, 1, 0, 1, 0, 1],
                 [1, 0, 1, 0, 0, 0, 1, 0], [1, 0, 1, 0, 1, 0, 0, 0], [0, 0, 1, 0, 1, 0, 1, 0],
                 [1, 0, 0, 0, 1, 0, 1, 0], [1, 0, 0, 1, 1, 1, 0, 0], [0, 0, 1, 0, 0, 1, 1, 1],
                 [1, 1, 0, 0, 1, 0, 0, 1], [0, 1, 1, 1, 0, 0, 1, 0], [1, 0, 1, 1, 0, 0, 1, 0],
                 [1, 0, 1, 0, 0, 1, 1, 0], [1, 0, 1, 1, 0, 1, 1, 0], [0, 1, 1, 0, 1, 0, 1, 1],
                 [1, 1, 0, 1, 1, 0, 1, 0], [1, 1, 0, 0, 1, 0, 1, 0], [0, 1, 1, 0, 1, 0, 1, 0],
                 [0, 0, 1, 0, 1, 0, 1, 1], [1, 0, 0, 1, 1, 0, 1, 0], [1, 0, 1, 0, 1, 1, 0, 1],
                 [1, 0, 1, 0, 1, 1, 0, 0], [1, 0, 1, 0, 1, 0, 0, 1], [0, 1, 0, 0, 1, 0, 1, 1],
                 [0, 1, 1, 0, 1, 0, 0, 1], [1, 1, 0, 1, 0, 0, 1, 0], [0, 1, 0, 1, 1, 0, 1, 0],
                 [0, 0, 1, 0, 1, 1, 0, 1], [1, 0, 1, 0, 0, 1, 0, 1], [1, 0, 0, 1, 0, 1, 1, 0],
                 [1, 0, 1, 1, 0, 1, 0, 0], [0, 1, 1, 1, 1, 0, 0, 1], [1, 1, 0, 1, 0, 1, 1, 1],
                 [1, 1, 1, 1, 0, 1, 0, 0], [0, 1, 0, 0, 1, 1, 1, 1]];
 
    for i in range(len(crossings)):
        A = crossings[i]
        B = np.ones((3, 3))
 
        B[1, 0] = A[0]
        B[0, 0] = A[1]
        B[0, 1] = A[2]
        B[0, 2] = A[3]
        B[1, 2] = A[4]
        B[2, 2] = A[5]
        B[2, 1] = A[6]
        B[2, 0] = A[7]
 
        structures.append(B.tolist())
 
    nodes = []
    for i in range(len(structures)):
        structure1 = np.array(structures[i])
        X0 = spim.binary_hit_or_miss(Xin, structure1=structure1).astype(int)
        r0, c0 = np.nonzero(X0 == 1)
 
        for j in range(len(r0)):
            nodes.append([r0[j], c0[j]])
 
    nodes = np.array(nodes)
 
    return nodes

def plot_edges_kinematic(data_path, mask_name, crack_kinematic, dir_save='../results/', plt_skeleton = True, local_trans = True):
    """
    Plots edges and skeleton results after kinematics is found

    Args:
        data_path (str): Data folder path
        mask_name (str): Name of the image crack pattern
        crack_kinematic (dict): Dictionary with the kinematics results
        dir_save (str, optional): Output folder path. Defaults to '../results/'.
        plt_skeleton (bool, optional): True to plot skeleton. Defaults to True.
        local_trans (bool, optional): True to plot local values. Otherwise global are plot. Defaults to True.
    """
    
    #To decide if local or global is required
    if local_trans:
            lt = "_loc"
    else:
            lt = ""   
    
    #read and plot mask
    mask = mask = cv2.imread(data_path+mask_name)
    mask = (mask[:,:,0]>0)*255
    plt.figure()
    plt.imshow(mask, 'gray')
    
    #GLOBAL
    #iterate thorugh regions according their assigned lables
    for lab in crack_kinematic:
        #Ploting crack edges
        for crk_edg in crack_kinematic[lab]["edges"]:
            crk_edg = np.array(crk_edg)
            plt.plot(crk_edg[:,0],crk_edg[:,1], c = (.0,.5,1.), marker = 'o')
        #Ploting transformed edges
        crk_edg_transf = crack_kinematic[lab]["edge_transf"+lt]
        crk_edg_transf = np.array(crk_edg_transf)
        plt.plot(crk_edg_transf[:,0],crk_edg_transf[:,1], c = (.2,.5,.2), marker = '.')
        #Ploting skeleton if True
        if plt_skeleton:
            crk_skl = crack_kinematic[lab]["skl"]
            crk_skl = np.array(crk_skl)
            plt.plot(crk_skl[:,0], crk_skl[:,1], 'y.')
    
    plt.savefig(dir_save+mask_name[:-4]+'_edges'+lt+'.png', bbox_inches='tight', pad_inches=0)
    plt.savefig(dir_save+mask_name[:-4]+'_edges'+lt+'.pdf', bbox_inches='tight', pad_inches=0)
    plt.close()
            

def plot_kinematic(data_path, mask_name, crack_kinematic, dir_save='../results/', plot_rotation=True, plot_trans_x=True, plot_trans_y=True, local_trans = True, full_edges=False, abs_disp = False):
    """
    Plots 3dof kinematic results

    Args:
        data_path (str): Data folder path
        mask_name (str): Name of the image crack pattern
        crack_kinematic (dict): Dictionary with the kinematics results
        dir_save (str, optional): Output folder path. Defaults to '../results/'.
        plot_rotation (bool, optional): True to plot rotation variable. Defaults to True.
        plot_trans_x (bool, optional): True to plot x displacement variable. Defaults to True.
        plot_trans_y (bool, optional): True to plot y displacement variable. Defaults to True.
        local_trans (bool, optional): True to plot local values. Otherwise global are plot. Defaults to True.
        full_edges (bool, optional): True if crack_kinematic corresponds to full edges approach. Defaults to False.
        abs_disp (bool, optional): True to compute absolute value of mean results. Defaults to False.
    """
    
    #To decide if local or global is required
    if local_trans:
            lt = "_loc"
    else:
            lt = ""
    
    #Creating colormap
    top = cm.get_cmap('Oranges_r', 256)
    bottom = cm.get_cmap('Blues', 256)
    newcolors = np.vstack((top(np.linspace(0, 1, 256)), bottom(np.linspace(0, 1, 256))))
    newcmp = ListedColormap(newcolors, name='OrangeBlue')    

    #Reading mask 
    mask = cv2.imread(data_path+mask_name)
    mask = (mask==0)*255
    
    #Reading skeleton and rotation information
    glob_coordinates_skl = []
    rotation = []
    trans_x = []
    trans_y = []
    for lab in crack_kinematic:
        glob_coordinates_skl = glob_coordinates_skl + [H_p[1] for H_p in crack_kinematic[lab]["H_params_cracks"+lt]]
        rotation = rotation + [H_p[2][0] for H_p in crack_kinematic[lab]["H_params_cracks"+lt]]
        trans_x = trans_x + [H_p[2][1] for H_p in crack_kinematic[lab]["H_params_cracks"+lt]]
        trans_y = trans_y + [H_p[2][2] for H_p in crack_kinematic[lab]["H_params_cracks"+lt]]
    
    glob_coordinates_skl = np.array(glob_coordinates_skl, 'int')
    rotation = np.array(rotation)
    trans_x = np.array(trans_x)
    trans_y = np.array(trans_y)
    
    #Ploting values
    if plot_rotation:
        #Rotation img
        rot_img = np.zeros_like(mask, 'float')
        rot_img = rot_img[:,:,0]
        rot_img[(glob_coordinates_skl[:,1], glob_coordinates_skl[:,0])] = rotation
        
        #Ploting rotation
        fig, ax = plt.subplots(1)        
        psm = ax.scatter(glob_coordinates_skl[:,0], glob_coordinates_skl[:,1], c=rotation, cmap=newcmp, vmin=-20 , vmax=+20, marker='.')
        ax.imshow(mask, alpha=0.3)
        clrbr = fig.colorbar(psm, ax=ax)
        clrbr.ax.set_title(r'$\theta [rad]$')
        
        if full_edges: lt+="_full_edges"
        fig.savefig(dir_save+mask_name[:-4]+'_kin_rot'+lt+'.png', bbox_inches='tight', pad_inches=0)
        fig.savefig(dir_save+mask_name[:-4]+'_kin_rot'+lt+'.pdf', bbox_inches='tight', pad_inches=0)
        plt.close()
    
        #Ploting values
    if plot_trans_x:
        #Rotation img
        trans_x_img = np.zeros_like(mask, 'float')
        trans_x_img = trans_x_img[:,:,0]
        trans_x_img[(glob_coordinates_skl[:,1], glob_coordinates_skl[:,0])] = trans_x
        
        #Ploting rotation
        fig, ax = plt.subplots(1)        
        psm = ax.scatter(glob_coordinates_skl[:,0], glob_coordinates_skl[:,1], c=trans_x, cmap=newcmp, vmin=-20 , vmax=+20, marker='.')
        ax.imshow(mask, alpha=0.3)
        clrbr = fig.colorbar(psm, ax=ax)
        clrbr.ax.set_title(r"$t_x [px]$")
        
        if full_edges: lt+="_full_edges"
        fig.savefig(dir_save+mask_name[:-4]+'_kin_tx'+lt+'.png', bbox_inches='tight', pad_inches=0)
        fig.savefig(dir_save+mask_name[:-4]+'_kin_tx'+lt+'.pdf', bbox_inches='tight', pad_inches=0)
        plt.close()
        
    if plot_trans_y:
        #Rotation img
        trans_y_img = np.zeros_like(mask, 'float')
        trans_y_img = trans_y_img[:,:,0]
        trans_y_img[(glob_coordinates_skl[:,1], glob_coordinates_skl[:,0])] = trans_y
        
        #Ploting rotation
        fig, ax = plt.subplots(1)        
        psm = ax.scatter(glob_coordinates_skl[:,0], glob_coordinates_skl[:,1], c=trans_y, cmap=newcmp, vmin=-20 , vmax=+20, marker='.')
        ax.imshow(mask, alpha=0.3)
        clrbr = fig.colorbar(psm, ax=ax)
        clrbr.ax.set_title(r"$t_y [px]$")
        
        if full_edges: lt+="_full_edges"
        fig.savefig(dir_save+mask_name[:-4]+'_kin_ty'+lt+'.png', bbox_inches='tight', pad_inches=0)
        fig.savefig(dir_save+mask_name[:-4]+'_kin_ty'+lt+'.pdf', bbox_inches='tight', pad_inches=0)
        plt.close()
        
    #Finding mean values
    if abs_disp:
        mean_rot = np.mean(np.abs(rotation))
        mean_trans_x = np.mean(np.abs(trans_x))
        mean_trans_y = np.mean(np.abs(trans_y))
    else:
        mean_rot = np.mean(rotation)
        mean_trans_x = np.mean(trans_x)
        mean_trans_y = np.mean(trans_y)
    
    print("The mean value for rotation, translation_x and translation_y are: ", mean_rot, mean_trans_x, mean_trans_y)


def plot_two_dofs_kinematic(data_path, mask_name, crack_kinematic, dir_save='../results/', plot_trans_x=True, plot_trans_y=True, local_trans = True, abs_disp=False, full_edges=False):
    """
    Plots 2dof horizontal, vertical kinematic results

    Args:
        data_path (str): Data folder path
        mask_name (str): Name of the image crack pattern
        crack_kinematic (dict): Dictionary with the kinematics results
        dir_save (str, optional): Output folder path. Defaults to '../results/'.
        plot_trans_x (bool, optional): True to plot x displacement variable. Defaults to True.
        plot_trans_y (bool, optional): True to plot y displacement variable. Defaults to True.
        local_trans (bool, optional): True to plot local values. Otherwise global are plot. Defaults to True.
        full_edges (bool, optional): True if crack_kinematic corresponds to full edges approach. Defaults to False.
        abs_disp (bool, optional): True to compute absolute value of mean results. Defaults to False.
    """
    
    
    #To decide if local or global is required
    if local_trans:
            lt = "_loc"
    else:
            lt = ""
    
    #Creating colormap
    top = cm.get_cmap('Oranges_r', 256)
    bottom = cm.get_cmap('Blues', 256)
    newcolors = np.vstack((top(np.linspace(0, 1, 256)), bottom(np.linspace(0, 1, 256))))
    newcmp = ListedColormap(newcolors, name='OrangeBlue')    

    #Reading mask 
    mask = cv2.imread(data_path+mask_name)
    mask = (mask==0)*255
    
    #Reading skeleton and rotation information
    glob_coordinates_skl = []
    two_dofs_x = []
    two_dofs_y = []
    for lab in crack_kinematic:
        glob_coordinates_skl = glob_coordinates_skl + [H_p[1] for H_p in crack_kinematic[lab]["H_params_cracks"+lt]]
        two_dofs_x = two_dofs_x + [t_dofs_x[2][0] for t_dofs_x in crack_kinematic[lab]["two_dofs"+lt]]
        two_dofs_y = two_dofs_y + [t_dofs_y[2][1] for t_dofs_y in crack_kinematic[lab]["two_dofs"+lt]]
    
    glob_coordinates_skl = np.array(glob_coordinates_skl, 'int')
    two_dofs_x = np.array(two_dofs_x)
    two_dofs_y = np.array(two_dofs_y)

    if plot_trans_x:
        #Rotation img
        trans_x_img = np.zeros_like(mask, 'float')
        trans_x_img = trans_x_img[:,:,0]
        trans_x_img[(glob_coordinates_skl[:,1], glob_coordinates_skl[:,0])] = two_dofs_x
        
        #Ploting rotation
        fig, ax = plt.subplots(1)        
        psm = ax.scatter(glob_coordinates_skl[:,0], glob_coordinates_skl[:,1], c=two_dofs_x, cmap=newcmp, vmin=-20 , vmax=+20, marker='.')
        ax.imshow(mask, alpha=0.3)
        clrbr = fig.colorbar(psm, ax=ax)
        clrbr.ax.set_title(r"$t_x [px]$")
        
        if full_edges: lt+="_full_edges"
        fig.savefig(dir_save+mask_name[:-4]+'_kin_two_dofs_tx'+lt+'.png', bbox_inches='tight', pad_inches=0)
        fig.savefig(dir_save+mask_name[:-4]+'_kin_two_dofs_tx'+lt+'.pdf', bbox_inches='tight', pad_inches=0)
        plt.close()
        
    if plot_trans_y:
        #Rotation img
        trans_y_img = np.zeros_like(mask, 'float')
        trans_y_img = trans_y_img[:,:,0]
        trans_y_img[(glob_coordinates_skl[:,1], glob_coordinates_skl[:,0])] = two_dofs_y
        
        #Ploting rotation
        fig, ax = plt.subplots(1)        
        psm = ax.scatter(glob_coordinates_skl[:,0], glob_coordinates_skl[:,1], c=two_dofs_y, cmap=newcmp, vmin=-20 , vmax=+20, marker='.')
        ax.imshow(mask, alpha=0.3)
        clrbr = fig.colorbar(psm, ax=ax)
        clrbr.ax.set_title(r"$t_y [px]$")
        
        if full_edges: lt+="_full_edges"
        fig.savefig(dir_save+mask_name[:-4]+'_kin_two_dofs_ty'+lt+'.png', bbox_inches='tight', pad_inches=0)
        fig.savefig(dir_save+mask_name[:-4]+'_kin_two_dofs_ty'+lt+'.pdf', bbox_inches='tight', pad_inches=0)
        plt.close()
    
    #Finding mean values
    if abs_disp:
        mean_trans_x = np.mean(np.abs(two_dofs_x))
        mean_trans_y = np.mean(np.abs(two_dofs_y))
    else:
        mean_trans_x = np.mean(two_dofs_x)
        mean_trans_y = np.mean(two_dofs_y)
    
    print("The mean value 2 dofs for translation_x and translation_y are: ", mean_trans_x, mean_trans_y)


def plot_n_t_kinematic(data_path, mask_name, crack_kinematic,dir_save='../results/', plot_trans_n=True, plot_trans_t=True, plot_trans_t_n = True, local_trans = True, abs_disp=False, full_edges=False, sign_convention = "new", dot_size=None, cmap_=None, resolution=None):
    """
    Plots 2dof tt (tangential), tn (normal) kinematic results

    Args:
        data_path (str): Data folder path
        mask_name (str): Name of the image crack pattern
        crack_kinematic (dict): Dictionary with the kinematics results
        dir_save (str, optional): Output folder path. Defaults to '../results/'.
        plot_trans_n (bool, optional): True to plot n displacement variable. Defaults to True.
        plot_trans_t (bool, optional): True to plot t displacement variable. Defaults to True.
        plot_trans_t_n (bool, optional): True to plot relation t/n. Defaults to True.
        local_trans (bool, optional): True to plot local values. Otherwise global are plot. Defaults to True.
        full_edges (bool, optional): True if crack_kinematic corresponds to full edges approach. Defaults to False.
        abs_disp (bool, optional): True to compute absolute value of mean results. Defaults to False.
        sign_convention (str, optional): Defines the sign convention assumed. Defaults to "new" (final paper).
        dot_size (_type_, optional): Size of dots representing the edges in plots. Defaults to None.
        cmap_ (str, optional): Colormap to plot figures. Defaults to None.
        resolution (float, optional): mm/px resolution ratio to compute output in mm. Defaults to None.
    """
    
    
    #To decide if local or global is required
    if local_trans:
            lt = "_loc"
    else:
            lt = ""
    
    #Creating colormap
    if cmap_ is None:
        top = cm.get_cmap('Oranges_r', 256)
        bottom = cm.get_cmap('Blues', 256)
        newcolors = np.vstack((top(np.linspace(0, 1, 256)), bottom(np.linspace(0, 1, 256))))
        newcmp = ListedColormap(newcolors, name='OrangeBlue')    
    else:
        newcmp = cmap_

    #Reading mask        
    mask = cv2.imread(data_path+mask_name)
    mask = (mask==0)*255
    
    #Reading skeleton and rotation information
    glob_coordinates_skl = []
    two_dofs_n = []
    two_dofs_t = []
    crack_class = []
    for lab in crack_kinematic:
        glob_coordinates_skl = glob_coordinates_skl + [coord[0] for coord in crack_kinematic[lab]["kinematics_n_t"+lt]]
        two_dofs_n = two_dofs_n + [t_dofs_n[2][0] for t_dofs_n in crack_kinematic[lab]["kinematics_n_t"+lt]]
        two_dofs_t = two_dofs_t + [t_dofs_t[2][1] for t_dofs_t in crack_kinematic[lab]["kinematics_n_t"+lt]]
        crack_class = crack_class + [cr_cl for cr_cl in crack_kinematic[lab]["crack_class"]]
    
    glob_coordinates_skl = np.array(glob_coordinates_skl, 'int')
    two_dofs_n = np.array(two_dofs_n)
    two_dofs_t = np.array(two_dofs_t)
    crack_class = np.array(crack_class)
    
    #Modifing signs according new sign sistem. Opening is possitive (always). Shear sliding, clockwise pair positive.
    #If class 1 (crack ascending), it is necessary to change sings with respect old convention
    #If class 2 (crack descending), it is necessary to keep sings with respect old convention
    if sign_convention=="new": #check later -- it might influence in the errory displa
        two_dofs_n = np.abs(two_dofs_n)
        ind_class1 = np.where(crack_class==1)
        two_dofs_t[ind_class1[0]] *= -1

    if plot_trans_n and len(glob_coordinates_skl)>0:
        trans_n_img = np.zeros_like(mask, 'float')
        trans_n_img = trans_n_img[:,:,0]
        trans_n_img[(glob_coordinates_skl[:,1], glob_coordinates_skl[:,0])] = two_dofs_n
        
        if resolution is None:
            a = 1
        else:
            a = resolution ##mm/px
        
        #Ploting 
        fig, ax = plt.subplots(1, figsize=(16,12))        
            
        if dot_size is not None:
            psm = ax.scatter(glob_coordinates_skl[:,0], glob_coordinates_skl[:,1], c=a*two_dofs_n, cmap=bottom, vmin=0 , vmax=+15, marker='.', s=dot_size)
            #psm = ax.scatter(glob_coordinates_skl[:,0], glob_coordinates_skl[:,1], c=a*two_dofs_n, cmap=newcmp, vmin=0 , vmax=+15, marker='.', s=dot_size)
        else:
            psm = ax.scatter(glob_coordinates_skl[:,0], glob_coordinates_skl[:,1], c=a*two_dofs_n, cmap=bottom, vmin=0 , vmax=+15, marker='.')
            #psm = ax.scatter(glob_coordinates_skl[:,0], glob_coordinates_skl[:,1], c=a*two_dofs_n, cmap=newcmp, vmin=0 , vmax=+15, marker='.')
        ax.imshow(mask, alpha=0.3)
        clrbr = fig.colorbar(psm, ax=ax)
        
        if resolution is None:
            clrbr.ax.set_title(r"$t_n [px]$")
        else:
            clrbr.ax.set_title(r"$t_n [mm]$")
        
        if full_edges: lt+="_full_edges"
        if resolution is not None: lt+="_mm"
        plt.tight_layout()
        fig.savefig(dir_save+mask_name[:-4]+'_n_t_kin_tn'+lt+'.png', bbox_inches='tight', pad_inches=0)
        fig.savefig(dir_save+mask_name[:-4]+'_n_t_kin_tn'+lt+'.pdf', bbox_inches='tight', pad_inches=0)
        plt.close()
        
    if plot_trans_t and len(glob_coordinates_skl)>0:
        trans_t_img = np.zeros_like(mask, 'float')
        trans_t_img = trans_t_img[:,:,0]
        trans_t_img[(glob_coordinates_skl[:,1], glob_coordinates_skl[:,0])] = two_dofs_t
        
        if resolution is None:
            a = 1
        else:
            a = resolution ##mm/px
        
        #Ploting 
        fig, ax = plt.subplots(1, figsize=(16,12))        
        #TO MAKE IT COMPARABLE WITH DIC METHOD, THE SIGN NEED TO BE CHANGED
        if cmap_=='jet':
            c_ = -two_dofs_t
        else:
            c_ = two_dofs_t
        if dot_size is not None:
            psm = ax.scatter(glob_coordinates_skl[:,0], glob_coordinates_skl[:,1], c=a*c_, cmap=newcmp, vmin=-20 , vmax=+20, marker='.', s=dot_size)
        else:
            psm = ax.scatter(glob_coordinates_skl[:,0], glob_coordinates_skl[:,1], c=a*c_, cmap=newcmp, vmin=-20 , vmax=+20, marker='.')
        #ax.imshow(mask, cmap=newcmp, alpha=0.2)
        ax.imshow(mask, alpha=0.3)
        clrbr = fig.colorbar(psm, ax=ax)
        if resolution is None:
            clrbr.ax.set_title(r"$t_t [px]$")
        else:
            clrbr.ax.set_title(r"$t_t [mm]$")
        
        if full_edges: lt+="_full_edges"
        if resolution is not None: lt+="_mm"
        plt.tight_layout()
        fig.savefig(dir_save+mask_name[:-4]+'_n_t_kin_tt'+lt+'.png', bbox_inches='tight', pad_inches=0)
        fig.savefig(dir_save+mask_name[:-4]+'_n_t_kin_tt'+lt+'.pdf', bbox_inches='tight', pad_inches=0)
        plt.close()
        
    if plot_trans_t_n and len(glob_coordinates_skl)>0:
        # img
        trans_t_n_img = np.zeros_like(mask, 'float')
        trans_t_n_img = trans_t_n_img[:,:,0]
        trans_t_n_img[(glob_coordinates_skl[:,1], glob_coordinates_skl[:,0])] = two_dofs_t/two_dofs_n
        
        #Ploting 
        fig, ax = plt.subplots(1, figsize=(16,12))        
        
        #TO MAKE IT COMPARABLE WITH DIC METHOD, THE SIGN NEED TO BE CHANGED
        if cmap_=='jet':
            c_ = -two_dofs_t/two_dofs_n
        else:
            c_ = two_dofs_t/two_dofs_n

        if dot_size is not None:
            psm = ax.scatter(glob_coordinates_skl[:,0], glob_coordinates_skl[:,1], c=c_, cmap=newcmp, vmin=-1 , vmax=+1, marker='.', s=dot_size)
        else:
            psm = ax.scatter(glob_coordinates_skl[:,0], glob_coordinates_skl[:,1], c=c_, cmap=newcmp, vmin=-1 , vmax=+1, marker='.')
        ax.imshow(mask, alpha=0.3)
        clrbr = fig.colorbar(psm, ax=ax)
        clrbr.ax.set_title(r"$t_t/t_n$")
        
        if full_edges: lt+="_full_edges"
        plt.tight_layout()
        fig.savefig(dir_save+mask_name[:-4]+'_n_t_kin_tt_tn'+lt+'.png', bbox_inches='tight', pad_inches=0)
        fig.savefig(dir_save+mask_name[:-4]+'_n_t_kin_tt_tn'+lt+'.pdf', bbox_inches='tight', pad_inches=0)
        plt.close()
   
    #Finding mean values
    if abs_disp:
        mean_trans_n = np.mean(np.abs(two_dofs_n))
        mean_trans_t = np.mean(np.abs(two_dofs_t))
    else:
        mean_trans_n = np.mean(two_dofs_n)
        mean_trans_t = np.mean(two_dofs_t)
    
    print("The mean value for translation_n and translation_t are: ", mean_trans_n, mean_trans_t)
    

def get_time_registration(data_path, mask_name, hyper_params, crack_kinematic=None, full_edges=False, local_trans = True, use_eta = False):
    """
    Gets the mean time of registration for results from a batch of crack patterns

    Args:
        data_path (str): Data folder path
        mask_name (str): Name of the image crack pattern
        hyper_params (list): List with the hyper_parameter values of the methodology (k/eta, m, l, kn, omega)._
        crack_kinematic (dict): Dictionary with the kinematics results
        full_edges (bool, optional): True if crack_kinematic corresponds to full edges approach. Defaults to False.
        local_trans (bool, optional): True to plot local values. Otherwise global are plot. Defaults to True.
        use_eta (bool, optional): True if eta is used instead of k-neigh. Defaults to False.

    Returns:
        mean_time_iteration (float): Mean time of registration for batch of crack patterns
    """
    
    
    #if crack_kinematic is not given. Read from json file
    if crack_kinematic==None:
        batch_name = data_path.split('/')[-2]
        #Path where is saved json file
        if full_edges:
            #dir_save = '../results/' + batch_name + '/' + mask_name[:-4] + '/' + 'full_edges/knnor{}_omega{}/'.format(hyper_params[0],hyper_params[1])
            dir_save = '../results/' + batch_name + '/' + mask_name[:-4] + '/' + 'full_edges/knnor{}_omega{}/'.format(int(hyper_params[0]),hyper_params[1])
        else:
            if use_eta:
                #dir_save = '../results/' + batch_name + '/' + mask_name[:-4] + '/' + 'finite_edges/eta{}_m{}_l{}_knnor{}_omega{}/'.format(hyper_params[0], hyper_params[1], hyper_params[2],hyper_params[3],hyper_params[4])
                dir_save = '../results/' + batch_name + '/' + mask_name[:-4] + '/' + 'finite_edges/eta{}_m{}_l{}_knnor{}_omega{}/'.format(hyper_params[0], hyper_params[1], hyper_params[2], int(hyper_params[3]), hyper_params[4])
            else:
                dir_save = '../results/' + batch_name + '/' + mask_name[:-4] + '/' + 'finite_edges/kn{}_m{}_l{}_knnor{}_omega{}/'.format(hyper_params[0], hyper_params[1], hyper_params[2],hyper_params[3],hyper_params[4])
                #dir_save = '../results/' + batch_name + '/' + mask_name[:-4] + '/' + 'finite_edges/kn{}_m{}_l{}_knnor{}_omega{}/'.format(hyper_params[0], int(hyper_params[1]), hyper_params[2], int(hyper_params[3]),int(hyper_params[4]))
        with open(dir_save+'crack_kinematic.json', 'r') as fp:
            crack_kinematic = json.load(fp)
    
    #To decide if local or global is required
    if local_trans:
            lt = "_loc"
    else:
            lt = ""
    
    #Reading skeleton and rotation information
    time_iteration = []
    
    for lab in crack_kinematic:
        
        time_iteration = time_iteration + [t_i[2] for t_i in crack_kinematic[lab]["time_iteration"]]
                
    
    time_iteration = np.array(time_iteration)
       
    #Finding mean values
    mean_time_iteration = np.mean(time_iteration)
                    
    print("--------")
    print("Mask name: " + mask_name)
    print("Hyper-parameters: ", hyper_params)
    print("The mean value for time iteration is:", mean_time_iteration)
    print("--------")
    
    return mean_time_iteration

def compute_kinematic_mean(data_path, mask_name, hyper_params, crack_kinematic=None, full_edges=False, local_trans = True, abs_disp = False, use_eta = False):
    """
    Computes the mean of the displacement values, theta, tx, ty, tx', ty', tn, tt
    Args:
        data_path (str): Data folder path
        mask_name (str): Name of the image crack pattern
        hyper_params (list): List with the hyper_parameter values of the methodology (k/eta, m, l, kn, omega)._
        crack_kinematic (dict): Dictionary with the kinematics results
        full_edges (bool, optional): True if crack_kinematic corresponds to full edges approach. Defaults to False.
        local_trans (bool, optional): True to plot local values. Otherwise global are plot. Defaults to True.
        use_eta (bool, optional): True if eta is used instead of k-neigh. Defaults to False.
        abs_disp (bool, optional): True to compute absolute value of mean results. Defaults to False.
        use_eta (bool, optional): _description_. Defaults to False.

    Returns:
        _type_three_dofs, two_dofs_x_y, two_dofs_n_t (list): lists with the mean values if kinematics
    """
    
    #if crack_kinematic is not given. Read from json file
    if crack_kinematic==None:
        batch_name = data_path.split('/')[-2]
        #Path where is saved json file
        if full_edges:
            #dir_save = '../results/' + batch_name + '/' + mask_name[:-4] + '/' + 'full_edges/knnor{}_omega{}/'.format(hyper_params[0],hyper_params[1])
            dir_save = '../results/' + batch_name + '/' + mask_name[:-4] + '/' + 'full_edges/knnor{}_omega{}/'.format(int(hyper_params[0]),hyper_params[1])
        else:
            if use_eta:
                #dir_save = '../results/' + batch_name + '/' + mask_name[:-4] + '/' + 'finite_edges/eta{}_m{}_l{}_knnor{}_omega{}/'.format(hyper_params[0], hyper_params[1], hyper_params[2],hyper_params[3],hyper_params[4])
                dir_save = '../results/' + batch_name + '/' + mask_name[:-4] + '/' + 'finite_edges/eta{}_m{}_l{}_knnor{}_omega{}/'.format(hyper_params[0], hyper_params[1], hyper_params[2], int(hyper_params[3]), hyper_params[4])
            else:
                dir_save = '../results/' + batch_name + '/' + mask_name[:-4] + '/' + 'finite_edges/kn{}_m{}_l{}_knnor{}_omega{}/'.format(hyper_params[0], hyper_params[1], hyper_params[2],hyper_params[3],hyper_params[4])
                #dir_save = '../results/' + batch_name + '/' + mask_name[:-4] + '/' + 'finite_edges/kn{}_m{}_l{}_knnor{}_omega{}/'.format(hyper_params[0], int(hyper_params[1]), hyper_params[2], int(hyper_params[3]),int(hyper_params[4]))
        with open(dir_save+'crack_kinematic.json', 'r') as fp:
            crack_kinematic = json.load(fp)
    
    #To decide if local or global is required
    if local_trans:
            lt = "_loc"
    else:
            lt = ""
    
    #Reading skeleton and rotation information
    glob_coordinates_skl = []
    rotation = []
    trans_x = []
    trans_y = []
    two_dofs_x = []
    two_dofs_y = []
    two_dofs_n = []
    two_dofs_t = []
    
    for lab in crack_kinematic:
        
        glob_coordinates_skl = glob_coordinates_skl + [H_p[1] for H_p in crack_kinematic[lab]["H_params_cracks"+lt]]
        rotation = rotation + [H_p[2][0] for H_p in crack_kinematic[lab]["H_params_cracks"+lt]]
        trans_x = trans_x + [H_p[2][1] for H_p in crack_kinematic[lab]["H_params_cracks"+lt]]
        trans_y = trans_y + [H_p[2][2] for H_p in crack_kinematic[lab]["H_params_cracks"+lt]]
        two_dofs_x = two_dofs_x + [t_dofs_x[2][0] for t_dofs_x in crack_kinematic[lab]["two_dofs"+lt]]
        two_dofs_y = two_dofs_y + [t_dofs_y[2][1] for t_dofs_y in crack_kinematic[lab]["two_dofs"+lt]]
        two_dofs_n = two_dofs_n + [t_dofs_n[2][0] for t_dofs_n in crack_kinematic[lab]["kinematics_n_t"+lt]]
        two_dofs_t = two_dofs_t + [t_dofs_t[2][1] for t_dofs_t in crack_kinematic[lab]["kinematics_n_t"+lt]]
        
    
    glob_coordinates_skl = np.array(glob_coordinates_skl, 'int')
    rotation = np.array(rotation)
    two_dofs_x = np.array(two_dofs_x)
    two_dofs_y = np.array(two_dofs_y)
    trans_x = np.array(trans_x)
    trans_y = np.array(trans_y)
    two_dofs_n = np.array(two_dofs_n)
    two_dofs_t = np.array(two_dofs_t)
    
       
    #Finding mean values
    if abs_disp:
        mean_rot = np.mean(np.abs(rotation))
        mean_trans_x = np.mean(np.abs(trans_x))
        mean_trans_y = np.mean(np.abs(trans_y))
        mean_trans_two_dofs_x = np.mean(np.abs(two_dofs_x))
        mean_trans_two_dofs_y = np.mean(np.abs(two_dofs_y))
        mean_trans_n = np.mean(np.abs(two_dofs_n))
        mean_trans_t = np.mean(np.abs(two_dofs_t))
    else:
        mean_rot = np.mean(rotation)
        mean_trans_x = np.mean(trans_x)
        mean_trans_y = np.mean(trans_y)
        mean_trans_two_dofs_x = np.mean(two_dofs_x)
        mean_trans_two_dofs_y = np.mean(two_dofs_y)
        mean_trans_n = np.mean(two_dofs_n)
        mean_trans_t = np.mean(two_dofs_t)
                
    print("--------")
    print("Mask name: " + mask_name)
    print("Hyper-parameters: ", hyper_params)
    print("The mean value for rotation, translation_x and translation_y are:", mean_rot, mean_trans_x, mean_trans_y)
    print("The mean value 2 dofs for translation_x and translation_y are:", mean_trans_two_dofs_x, mean_trans_two_dofs_y)
    print("The mean value for translation_n and translation_t are:", mean_trans_n, mean_trans_t)
    print("--------")
    
    three_dofs = [mean_rot, mean_trans_x, mean_trans_y]
    two_dofs_x_y = [mean_trans_two_dofs_x, mean_trans_two_dofs_y]
    two_dofs_n_t = [mean_trans_n, mean_trans_t]
    
    return three_dofs, two_dofs_x_y, two_dofs_n_t

def load_GT_kinematic(data_path, mask_name):
    """
    Reads and return npy arrays with the ground truth values of kinematics

    Args:
        data_path (str): Data folder path
        mask_name (str): Name of the image crack pattern

    Returns:
        H, dx_dy (array): Arrays with kinematics ground truth
    """
    
    H = np.load(data_path+mask_name[:-4]+'_H_params.npy')
    
    x = np.load(data_path+mask_name[:-4]+'_edge1.npy') #
    y = np.load(data_path+mask_name[:-4]+'_edge0.npy')#
    if len(x)<len(y):
        y = y[:len(x)]
    else:
        x = x[:len(y)]
    Dx_Dy = x-y
    dx_dy = np.mean(Dx_Dy, axis=0)
    
    return H, dx_dy

def error_kinematic(H, dx_dy, three_dofs, two_dofs, smooth=1e-17):
    """
    Computes the error of kinematic for the different outputs. 3dof, 2dof -x,y, 
    2dof-t,n.

    Args:
        H (array): Transformation matrix ground truth
        dx_dy (array): Delta displacements ground truth
        three_dofs (array): Computed kinematic values for 3dof
        two_dofs (array): Computed kinematic values for 2dof
        smooth (float, optional): Small value to avoid inf nan results. Defaults to 1e-17.

    Returns:
        kinematic errors: error of kinematic for the different outputs. 3dof, 2dof -x,y, 2dof-t,n.
    """

    #Absolute error
    error_3dof = np.abs(np.abs(H) - np.abs(three_dofs))
    error_2dof = np.abs(np.abs(dx_dy) - np.abs(two_dofs))
    #Relative percent difference 
    C_3dof = np.abs(100*2*(np.abs(H) - np.abs(three_dofs))/(np.abs(H) + np.abs(three_dofs)))#100*2((np.abs(H) - np.abs(three_dofs))/(np.abs(H) + np.abs(three_dofs)))
    C_2dof = np.abs(100*2*(np.abs(dx_dy) - np.abs(two_dofs))/(np.abs(dx_dy) + np.abs(two_dofs)))
    
    return error_3dof, error_2dof, C_3dof, C_2dof

def error_kinematic_batch(data_path, list_k_neighboors=None, list_m=None, list_l=None, list_k_n_normal_feature=None, list_omega=None, full_edges=False, monte_carlo=False, read_hyper_from_results = False, list_eta = None):
    """
    Computes and save as dictionary the kinematic error for different outputs for a
    batch of crack patterns

    Args:
        data_path (str): Data folder path
        list_k_neighboors (list, optional): List with different values of k neighbors aplied to the batch. Defaults to None.
        list_m (list, optional): List with different values of m aplied to the batch. Defaults to None.
        list_l (list, optional): List with different values of l  aplied to the batch. Defaults to None.
        list_k_n_normal_feature (liest, optional): List with different values of kn neighbors aplied to the batch. Defaults to None.
        list_omega (list, optional): List with different values of omega neighbors aplied to the batch. Defaults to None.
        full_edges (bool, optional): True if full edges approach is used. Defaults to False.
        monte_carlo (bool, optional): True if the batch was used for montecarlo simulation. Defaults to False.
        read_hyper_from_results (bool, optional): If true, hyper parameters lists are read from results. Defaults to False.
        list_eta (list, optional): List of different values of eta applied to the batch (if given k_neigh is ignored). Defaults to None.
    """
    
    #List with the batch of images
    list_mask_name = [mask_name for mask_name in os.listdir(data_path) if mask_name.endswith(".png")]
    list_mask_name.sort()
    
    #Dictionary with the error results for the kinematics of the batch of crack patterns
    dict_error_kinematic_batch = {}
    dict_error_kinematic_batch['three_dofs'] = {}
    dict_error_kinematic_batch['two_dofs'] = {}
    dict_error_kinematic_batch['two_dofs_unique'] = {}
    dict_error_kinematic_batch['time_iteration'] = {}
    dict_error_kinematic_batch['C_three_dofs'] = {}
    dict_error_kinematic_batch['C_two_dofs'] = {}    
    
    batch_name = data_path.split('/')[-2]
    list_mask_name = []
    list_k_neighboors = []
    list_etas = []
    list_m = []
    list_l = []
    list_k_n_normal_feature = []
    list_omega = []
    
    #Reading the hyperparameters used to run the batch kinematics
    if read_hyper_from_results:
        list_masks_folders = os.listdir('../results/'+batch_name+'/')
        list_masks_folders = [lmf for lmf in list_masks_folders if not lmf.endswith('.json')]
        list_masks_folders = [lmf for lmf in list_masks_folders if not lmf.endswith('.png')]
        list_masks_folders = [lmf for lmf in list_masks_folders if not lmf.endswith('.pdf')]
        list_masks_folders.sort()
        for lmn in list_masks_folders:
            if full_edges:
                approach = 'full_edges/'
            else:
                approach = 'finite_edges/'
            for hyp in os.listdir('../results/'+batch_name+'/'+lmn+'/'+approach):
                list_mask_name.append(lmn+'.png')
                hyp_split = hyp.split('_')
                #list_omega.append(float(hyp_split[-1].replace("omega","")))
                list_omega.append(hyp_split[-1].replace("omega",""))
                #list_k_n_normal_feature.append(int(float(hyp_split[-2].replace("knnor",""))))
                list_k_n_normal_feature.append(float(hyp_split[-2].replace("knnor","")))
                if not full_edges:
                    list_l.append(int(hyp_split[-3].replace("l","")))
                    list_m.append(float(hyp_split[-4].replace("m","")))
                    #list_m.append(int(hyp_split[-4].replace("m","")))
                    if list_eta is None:
                        list_k_neighboors.append(int(hyp_split[-5].replace("kn","")))
                    else:
                        list_etas.append(float(hyp_split[-5].replace("eta","")))
    
    if monte_carlo:
        if full_edges:
            for mask_name, omega, k_n_normal_feature in zip(list_mask_name, list_omega, list_k_n_normal_feature):    
                hyper_params = hyper_params = [k_n_normal_feature, omega]
                three_dofs, two_dofs, _ = compute_kinematic_mean(data_path, mask_name, hyper_params, crack_kinematic=None, full_edges=True, local_trans = False, abs_disp = True)
                H, dx_dy = load_GT_kinematic(data_path, mask_name)
                e_3dof,e_2dof, C_3dof, C_2dof = error_kinematic(H, dx_dy, three_dofs, two_dofs) 
                dict_error_kinematic_batch['three_dofs']['knnor{}_omega{}_mask_n{}'.format(k_n_normal_feature,omega, mask_name)] = [hyper_params, list(e_3dof)]
                dict_error_kinematic_batch['two_dofs']['knnor{}_omega{}_mask_n{}'.format(k_n_normal_feature,omega, mask_name)] = [hyper_params, list(e_2dof)]
                dict_error_kinematic_batch['two_dofs_unique']['knnor{}_omega{}_mask_n{}'.format(k_n_normal_feature,omega, mask_name)] = [hyper_params, [np.mean(e_2dof)]]
                dict_error_kinematic_batch['C_three_dofs']['knnor{}_omega{}_mask_n{}'.format(k_n_normal_feature,omega, mask_name)] = [hyper_params, list(C_3dof)]
                dict_error_kinematic_batch['C_two_dofs']['knnor{}_omega{}_mask_n{}'.format(k_n_normal_feature,omega, mask_name)] = [hyper_params, list(C_2dof)]
        else:
            if list_eta is None:
                for mask_name, k_neighboors, m, l, omega, k_n_normal_feature in zip(list_mask_name, list_k_neighboors, list_m, list_l, list_omega, list_k_n_normal_feature):
                    #Only make sense if m>1 (edge1 bigger than edge0) and kn*m>l
                    if l >= k_neighboors*m:
                        print("HERE", l, k_neighboors, m)
                        continue
                    #hyper_params = hyper_params = [k_neighboors,m,l,k_n_normal_feature, omega]
                    hyper_params = [k_neighboors,m,l,k_n_normal_feature, omega]
                    three_dofs, two_dofs, _ = compute_kinematic_mean(data_path, mask_name, hyper_params, crack_kinematic=None, full_edges=False, local_trans = True, abs_disp = True)
                    H, dx_dy = load_GT_kinematic(data_path, mask_name)
                    e_3dof,e_2dof, C_3dof, C_2dof = error_kinematic(H, dx_dy, three_dofs, two_dofs) 
                    mean_time_registration = 0
                    dict_error_kinematic_batch['three_dofs']['kn{}_m{}_l{}_knnor{}_omega{}_mask_n{}/'.format(k_neighboors, m, l,k_n_normal_feature,omega, mask_name)] = [hyper_params, list(e_3dof)]
                    dict_error_kinematic_batch['two_dofs']['kn{}_m{}_l{}_knnor{}_omega{}_mask_n{}/'.format(k_neighboors, m, l,k_n_normal_feature,omega, mask_name)] = [hyper_params, list(e_2dof)]
                    dict_error_kinematic_batch['two_dofs_unique']['kn{}_m{}_l{}_knnor{}_omega{}_mask_n{}/'.format(k_neighboors, m, l,k_n_normal_feature,omega, mask_name)] = [hyper_params, [np.mean(e_2dof)]]
                    dict_error_kinematic_batch['time_iteration']['kn{}_m{}_l{}_knnor{}_omega{}_mask_n{}/'.format(k_neighboors, m, l,k_n_normal_feature,omega, mask_name)] = [hyper_params, [mean_time_registration]]
                    dict_error_kinematic_batch['C_three_dofs']['kn{}_m{}_l{}_knnor{}_omega{}_mask_n{}/'.format(k_neighboors, m, l,k_n_normal_feature,omega, mask_name)] = [hyper_params, list(C_3dof)]
                    dict_error_kinematic_batch['C_two_dofs']['kn{}_m{}_l{}_knnor{}_omega{}_mask_n{}/'.format(k_neighboors, m, l,k_n_normal_feature,omega, mask_name)] = [hyper_params, list(C_2dof)]
            else:
                for mask_name, eta, m, l, omega, k_n_normal_feature in zip(list_mask_name, list_etas, list_m, list_l, list_omega, list_k_n_normal_feature):
                    hyper_params = [eta,m,l,k_n_normal_feature, omega]
                    #hyper_params = [eta,m,l,float(k_n_normal_feature), omega]
                    three_dofs, two_dofs, _ = compute_kinematic_mean(data_path, mask_name, hyper_params, crack_kinematic=None, full_edges=False, local_trans = True, abs_disp = True, use_eta=True)
                    H, dx_dy = load_GT_kinematic(data_path, mask_name)
                    e_3dof,e_2dof, C_3dof, C_2dof = error_kinematic(H, dx_dy, three_dofs, two_dofs) 
                    mean_time_registration = 0
                    dict_error_kinematic_batch['three_dofs']['eta{}_m{}_l{}_knnor{}_omega{}_mask_n{}/'.format(eta, m, l,k_n_normal_feature,omega, mask_name)] = [hyper_params, list(e_3dof)]
                    dict_error_kinematic_batch['two_dofs']['eta{}_m{}_l{}_knnor{}_omega{}_mask_n{}/'.format(eta, m, l,k_n_normal_feature,omega, mask_name)] = [hyper_params, list(e_2dof)]
                    dict_error_kinematic_batch['two_dofs_unique']['eta{}_m{}_l{}_knnor{}_omega{}_mask_n{}/'.format(eta, m, l,k_n_normal_feature,omega, mask_name)] = [hyper_params, [np.mean(e_2dof)]]
                    dict_error_kinematic_batch['time_iteration']['eta{}_m{}_l{}_knnor{}_omega{}_mask_n{}/'.format(eta, m, l,k_n_normal_feature,omega, mask_name)] = [hyper_params, [mean_time_registration]]
                    dict_error_kinematic_batch['C_three_dofs']['eta{}_m{}_l{}_knnor{}_omega{}_mask_n{}/'.format(eta, m, l,k_n_normal_feature,omega, mask_name)] = [hyper_params, list(C_3dof)]
                    dict_error_kinematic_batch['C_two_dofs']['eta{}_m{}_l{}_knnor{}_omega{}_mask_n{}/'.format(eta, m, l,k_n_normal_feature,omega, mask_name)] = [hyper_params, list(C_2dof)]
    else:
        if full_edges:    
            for omega in list_omega:
                check_omega = 0        
                for k_n_normal_feature in list_k_n_normal_feature:
                    #if omega==0, it just need one iteration as normals would not matter
                    if check_omega==1:
                        continue            
                    errors_3dof = []
                    errors_2dof = []
                    hyper_params = hyper_params = [k_n_normal_feature, omega]
                    for mask_name in list_mask_name:  
                        three_dofs, two_dofs, _ = compute_kinematic_mean(data_path, mask_name, hyper_params, crack_kinematic=None, full_edges=True, local_trans = False, abs_disp = True)
                        H, dx_dy = load_GT_kinematic(data_path, mask_name)
                        e_3dof,e_2dof, C_3dof, C_2dof = error_kinematic(H, dx_dy, three_dofs, two_dofs) 
                        errors_3dof.append(e_3dof)
                        errors_2dof.append(e_2dof)
                    errors_3dof = np.array(errors_3dof)
                    errors_2dof = np.array(errors_2dof)
                    dict_error_kinematic_batch['three_dofs']['knnor{}_omega{}'.format(k_n_normal_feature,omega)] = [hyper_params, list((np.mean(errors_3dof,axis=0)))]
                    dict_error_kinematic_batch['two_dofs']['knnor{}_omega{}'.format(k_n_normal_feature,omega)] = [hyper_params, list((np.mean(errors_2dof,axis=0)))]
                    dict_error_kinematic_batch['two_dofs_unique']['knnor{}_omega{}'.format(k_n_normal_feature,omega)] = [hyper_params, [np.mean(np.mean(errors_2dof,axis=0))]]
                    if omega==0:
                        check_omega+=1
        else:
            for k_neighboors in list_k_neighboors:
                for m in list_m:
                    #Only make sense iterate in l if m>1. If m==1 the just compute once
                    check_m = 0
                    for l in list_l:
                        #Only make sense if m>1 (edge1 bigger than edge0) and kn*m>l
                        if check_m == len(list_omega)*len(list_k_n_normal_feature):
                                continue
                        if l >= k_neighboors*m:
                            continue
                        for omega in list_omega:
                            check_omega = 0        
                            for k_n_normal_feature in list_k_n_normal_feature:
                                #if omega==0, it just need one iteration as normals would not matter
                                if check_omega==1:
                                    continue   
                                errors_3dof = []
                                errors_2dof = []
                                hyper_params = hyper_params = [k_neighboors,m,l,k_n_normal_feature, omega]
                                for mask_name in list_mask_name:                    
                                    three_dofs, two_dofs, _ = compute_kinematic_mean(data_path, mask_name, hyper_params, crack_kinematic=None, full_edges=False, local_trans = True, abs_disp = True)
                                    H, dx_dy = load_GT_kinematic(data_path, mask_name)
                                    e_3dof,e_2dof, C_3dof, C_2dof = error_kinematic(H, dx_dy, three_dofs, two_dofs) 
                                    errors_3dof.append(e_3dof)
                                    errors_2dof.append(e_2dof)
                                    
                                errors_3dof = np.array(errors_3dof)
                                errors_2dof = np.array(errors_2dof)
                                
                                dict_error_kinematic_batch['three_dofs']['kn{}_m{}_l{}_knnor{}_omega{}/'.format(k_neighboors, m, l,k_n_normal_feature,omega)] = [hyper_params, list(np.mean(errors_3dof,axis=0))]
                                dict_error_kinematic_batch['two_dofs']['kn{}_m{}_l{}_knnor{}_omega{}/'.format(k_neighboors, m, l,k_n_normal_feature,omega)] = [hyper_params, list(np.mean(errors_2dof,axis=0))]
                                dict_error_kinematic_batch['two_dofs_unique']['kn{}_m{}_l{}_knnor{}_omega{}/'.format(k_neighboors, m, l,k_n_normal_feature,omega)] = [hyper_params, [np.mean(np.mean(errors_2dof,axis=0))]]
                                if omega==0:
                                    check_omega+=1
                                if m==1:
                                        check_m+=1
        
    batch_name = data_path.split('/')[-2] #data_path has to finish in '/'; e.g.'../data/toy_examples_re/'
    
    if full_edges:
        with open('../results/' + batch_name + '/full_edges_error_kinematic_batch.json', 'w') as fp:
            json.dump(dict_error_kinematic_batch, fp)
    else:
        with open('../results/' + batch_name + '/finite_edges_error_kinematic_batch.json', 'w') as fp:
            json.dump(dict_error_kinematic_batch, fp)

        
def plot_ablation(data_path, k_n_normal_feature=None, omega=None, full_edges=False):
    """
    Plots figures related with ablation study of kn and omega

    Args:
        data_path (str): data folder path
        k_n_normal_feature (int, optional): fixed kn value. Defaults to None.
        omega (float, optional): fixed omega value. Defaults to None.
        full_edges (bool, optional): True if full edges approach was used. Defaults to False.
    """
    
    batch_name = data_path.split('/')[-2] #data_path has to finish in '/'; e.g.'../data/toy_examples_re/'
    
    if full_edges:
        with open('../results/' + batch_name + '/full_edges_error_kinematic_batch.json', 'r') as fp:
            dict_error_kinematic_batch = json.load(fp)
        data = np.empty(shape=(0,3))
        for key in dict_error_kinematic_batch['two_dofs_unique']:
            knnor = dict_error_kinematic_batch['two_dofs_unique'][key][0][0]
            om = dict_error_kinematic_batch['two_dofs_unique'][key][0][1]
            error = dict_error_kinematic_batch['two_dofs_unique'][key][1][0]
            data = np.concatenate((data, np.array([[knnor,om,error]])))
        Xs = data[:,0]
        Ys = data[:,1]
        Zs = data[:,2]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        pnt3d = ax.scatter(Xs,Ys,Zs, c=Zs, cmap=cm.bone)
        cbar=plt.colorbar(pnt3d)
        cbar.set_label("$\|error\|$ $[px]$")
        ax.set_xlabel("$k_n$")
        ax.set_ylabel("$\omega$")
        ax.set_zlabel("$\|error\|$ $[px]$")
        fig.tight_layout()
        plt.show()
        fig.savefig('../results/' + batch_name + '/full_edges_error_kinematic_batch_scatter3D.pdf')
        fig.savefig('../results/' + batch_name + '/full_edges_error_kinematic_batch_scatter3D.png')
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        pnt2d = ax.scatter(Xs,Ys, c=Zs, cmap=cm.bone)
        cbar=plt.colorbar(pnt2d)
        cbar.set_label("$\|error\|$ $[px]$")
        ax.set_xlabel("$k_n$")
        ax.set_ylabel("$\omega$")
        fig.tight_layout()
        plt.show()
        
        fig.savefig('../results/' + batch_name + '/full_edges_error_kinematic_batch_scatter2D.pdf')
        fig.savefig('../results/' + batch_name + '/full_edges_error_kinematic_batch_scatter2D.png')

        #Was not able to change colormap related with 4th variable .. Surface
        if len(Xs)>2:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_trisurf(Xs, Ys, Zs, cmap=cm.bone, linewidth=0)
            fig.colorbar(surf)        
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(6))
            ax.zaxis.set_major_locator(MaxNLocator(5))        
            ax.set_xlabel("$k_n$")
            ax.set_ylabel("$\omega$")
            ax.set_zlabel("$\|error\|$ $[px]$")
            fig.tight_layout()
            plt.show()
            
            fig.savefig('../results/' + batch_name + '/full_edges_error_kinematic_batch_trisurf3D.pdf')
            fig.savefig('../results/' + batch_name + '/full_edges_error_kinematic_batch_trisurf3D.png')

        #Best parameters
        best_id = np.where((data[:,2]==np.min(data[:,2])))
        best_kn , best_omega , best_error = data[best_id[0][0]]
        
        print("The best combination of hyperparameters when the full edges are used is k_n_normal_feature: {} , omega: {}. It gives an error of {} [px]".format(best_kn , best_omega , best_error))
    
    else:
        with open('../results/' + batch_name + '/finite_edges_error_kinematic_batch.json', 'r') as fp:
            dict_error_kinematic_batch = json.load(fp)
        
        #Fixed k_n_normal_feature and omega according to full edges and plot the error in function of k,m,l
        data = np.empty(shape=(0,6))
        for key in dict_error_kinematic_batch['two_dofs_unique']:
            k = dict_error_kinematic_batch['two_dofs_unique'][key][0][0]
            m = dict_error_kinematic_batch['two_dofs_unique'][key][0][1]
            l = dict_error_kinematic_batch['two_dofs_unique'][key][0][2]
            knnor = dict_error_kinematic_batch['two_dofs_unique'][key][0][3]
            om = dict_error_kinematic_batch['two_dofs_unique'][key][0][4]
            error = dict_error_kinematic_batch['two_dofs_unique'][key][1][0]
            data = np.concatenate((data, np.array([[k,m,l,knnor,om,error]])))
        
        #Taken data where kn_normal_feature and omega are fixed
        data_filtered = data[np.where((data[:,3]==k_n_normal_feature))]
        data_filtered = data_filtered[np.where((data_filtered[:,4]==omega))]
        
        Xs = data_filtered[:,0]
        Ys = data_filtered[:,1]
        Zs = data_filtered[:,2]
        Error = data_filtered[:,-1]
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        pnt3d = ax.scatter(Xs,Ys,Zs, c=Error, cmap=cm.bone)
        cbar=plt.colorbar(pnt3d)
        cbar.set_label("$\|error\|$ $[px]$")
        ax.set_xlabel("$k$")
        ax.set_ylabel("$\mu$")
        ax.set_zlabel("$\lambda$")
        
        fig.tight_layout()
        plt.show()

        fig.savefig('../results/' + batch_name + '/finite_edges_error_kinematic_batch_knnormal{}_omega{}.pdf'.format(k_n_normal_feature,omega))
        fig.savefig('../results/' + batch_name + '/finite_edges_error_kinematic_batch_knnormal{}_omega{}.png'.format(k_n_normal_feature,omega))

        #Best parameters
        best_id = np.where((data_filtered[:,-1]==np.min(data_filtered[:,-1])))
        best_k , best_mu , best_lambda  = data_filtered[best_id[0][0]][:3]
        best_error = data_filtered[best_id[0][0]][-1]
        
        print("For k_n_normal_feature: {} , omega: {}, The best combination of hyperparameters when the full edges are used is k: {}, mu: {}, lambda: {}. It gives an error of {} [px]".format(k_n_normal_feature, omega , best_k, best_mu, best_lambda,  best_error))
  

def plot_pdf(data_path, full_edges=False, use_eta = False):
    """
    Plot probability distribution functions for montecarlo simulations

    Args:
        data_path (str): data folder path
        full_edges (bool, optional): True if full edges approach is applied. Defaults to False.
        use_eta (bool, optional): True if eta is used instead of k_neigh. Defaults to False.
    """
    
    batch_name = data_path.split('/')[-2] #data_path has to finish in '/'; e.g.'../data/toy_examples_re/'
    
    if full_edges:
        with open('../results/' + batch_name + '/full_edges_error_kinematic_batch.json', 'r') as fp:
            dict_error_kinematic_batch = json.load(fp)
        data = np.empty(shape=(0,3))
        for key in dict_error_kinematic_batch['two_dofs_unique']:
            knnor = dict_error_kinematic_batch['two_dofs_unique'][key][0][0]
            om = dict_error_kinematic_batch['two_dofs_unique'][key][0][1]
            error = dict_error_kinematic_batch['two_dofs_unique'][key][1][0]
            data = np.concatenate((data, np.array([[knnor,om,error]])))
            
        data = data.astype(float)
        x1 = data[:,2]
        print(x1)
        min_x1 = np.min(x1)
        max_x1 = np.max(x1)
        kde = stats.gaussian_kde(x1)
        xs = np.linspace(0, max_x1+2, num=500)
        kde.set_bandwidth(bw_method='silverman')
        y2 = kde(xs)
        fig, ax = plt.subplots()
        ax.plot(x1, np.full(x1.shape, 1 / (4. * x1.size)), 'c.',
        label='Data point')
        ax.plot(xs, y2, label='KDE', color = 'g')
        ax.set_xlabel("$\|error\|$ $[px]$")
        ax.set_ylabel("Probability density")
        ax.legend()
        plt.xticks(np.arange(0, int(max_x1+2), step=2))
        plt.xlim(0,int(max_x1+2))
        plt.show()
        plt.close()
        fig.savefig('../results/' + batch_name + '/full_edges_error_kinematic_batch_pdf.pdf')
        fig.savefig('../results/' + batch_name + '/full_edges_error_kinematic_batch_pdf.png')
        print("The error mean and standar deviation for the montecarlo sampling experiment is: mean: {} , std: {}. This with a total of {} combinations of hyperparameters".format(np.mean(x1) , np.std(x1), len(x1)))
    else:
        with open('../results/' + batch_name + '/finite_edges_error_kinematic_batch.json', 'r') as fp:
            dict_error_kinematic_batch = json.load(fp)
        
        #Fixed k_n_normal_feature and omega according to full edges and plot the error in function of k,m,l        
        data = np.empty(shape=(0,6))
        for key in dict_error_kinematic_batch['two_dofs_unique']:
            if use_eta:
                eta = dict_error_kinematic_batch['two_dofs_unique'][key][0][0]
            else:
                k = dict_error_kinematic_batch['two_dofs_unique'][key][0][0]
            m = dict_error_kinematic_batch['two_dofs_unique'][key][0][1]
            l = dict_error_kinematic_batch['two_dofs_unique'][key][0][2]
            knnor = dict_error_kinematic_batch['two_dofs_unique'][key][0][3]
            om = dict_error_kinematic_batch['two_dofs_unique'][key][0][4]
            error = dict_error_kinematic_batch['two_dofs_unique'][key][1][0]
            if use_eta:
                data = np.concatenate((data, np.array([[eta,m,l,knnor,om,error]])))
            else:
                data = np.concatenate((data, np.array([[k,m,l,knnor,om,error]])))
        
        data = data.astype(float)
        x1 = data[:,-1]
        min_x1 = np.min(x1)
        max_x1 = np.max(x1)
        kde = stats.gaussian_kde(x1)
        xs = np.linspace(0, max_x1+2, num=500)
        kde.set_bandwidth(bw_method='silverman')
        y2 = kde(xs)
        fig, ax = plt.subplots()
        ax.plot(x1, np.full(x1.shape, 1 / (4. * x1.size)), 'c.',
        label='Data point')
        ax.plot(xs, y2, label='KDE', color = 'g')
        ax.set_xlabel("$\|error\|$ $[px]$")
        ax.set_ylabel("Probability density")
        ax.legend()
        plt.xticks(np.arange(0, int(max_x1+2), step=5))
        plt.xlim(0,int(max_x1+2))
        plt.show()
        fig.savefig('../results/' + batch_name + '/finite_edges_error_kinematic_batch_pdf.pdf')
        fig.savefig('../results/' + batch_name + '/finite_edges_error_kinematic_batch_pdf.png')
       
        print("The error mean and standar deviation for the montecarlo sampling experiment is: mean: {} , std: {}. This with a total of {} combinations of hyperparameters".format(np.mean(x1) , np.std(x1), len(x1)))

def plot_pdf_full(data_paths, full_edges = False, use_eta_list = [False,], labels_list = ['',], colors_list = ['#1f77b4',], plot_name = ''):
    """
    Plotting probability distribution functions for all the batches 

    Args:
        data_paths (str): data folder path
        full_edges (bool, optional): True if full edges approach is applied. Defaults to False.
        use_eta (bool, optional): True if eta is used instead of k_neigh. Defaults to False.
        labels_list (list, optional): List with the plot labels. Defaults to ['',].
        colors_list (list, optional): List with the plot colors. Defaults to ['#1f77b4',].
        plot_name (str, optional): additional name to save plots. Defaults to ''.
    """
    
    batch_names = []
    for data_path in data_paths:
        batch_names.append(data_path.split('/')[-2])
    
    fig, ax = plt.subplots()
    ax.set_xlabel("$\|error\|$ $[px]$", fontsize = 10)
    ax.set_ylabel("Probability density", fontsize = 10)
        
    c=0
    for use_eta, labels, batch_name, col in zip(use_eta_list, labels_list, batch_names, colors_list):
        c+=1
        if full_edges:
            with open('../results/' + batch_name + '/full_edges_error_kinematic_batch.json', 'r') as fp:
                dict_error_kinematic_batch = json.load(fp)
            data = np.empty(shape=(0,3))
            for key in dict_error_kinematic_batch['two_dofs_unique']:
                knnor = dict_error_kinematic_batch['two_dofs_unique'][key][0][0]
                om = dict_error_kinematic_batch['two_dofs_unique'][key][0][1]
                error = dict_error_kinematic_batch['two_dofs_unique'][key][1][0]
                data = np.concatenate((data, np.array([[knnor,om,error]])))
    
            data = data.astype(float)
            x1 = data[:,2]
            print(x1)
            min_x1 = np.min(x1)
            max_x1 = np.max(x1)
            kde = stats.gaussian_kde(x1)
            xs = np.linspace(0, max_x1+2, num=500)
            kde.set_bandwidth(bw_method='silverman')
            y2 = kde(xs)
            ax.scatter(x1, np.full(x1.shape, 1 / (4. * x1.size))-c*.01, c=col, marker='.', label='Data point '+labels)
            ax.plot(xs, y2, label='KDE '+labels, color = col)
            
            print("The error mean and standar deviation for the montecarlo sampling experiment is: mean: {} , std: {}. This with a total of {} combinations of hyperparameters".format(np.mean(x1) , np.std(x1), len(x1)))
        
        else:
            with open('../results/' + batch_name + '/finite_edges_error_kinematic_batch.json', 'r') as fp:
                dict_error_kinematic_batch = json.load(fp)
            
            data = np.empty(shape=(0,6))
            for key in dict_error_kinematic_batch['two_dofs_unique']:
                if use_eta:
                    eta = dict_error_kinematic_batch['two_dofs_unique'][key][0][0]
                else:
                    k = dict_error_kinematic_batch['two_dofs_unique'][key][0][0]
                m = dict_error_kinematic_batch['two_dofs_unique'][key][0][1]
                l = dict_error_kinematic_batch['two_dofs_unique'][key][0][2]
                knnor = dict_error_kinematic_batch['two_dofs_unique'][key][0][3]
                om = dict_error_kinematic_batch['two_dofs_unique'][key][0][4]
                error = dict_error_kinematic_batch['two_dofs_unique'][key][1][0]
                if use_eta:
                    data = np.concatenate((data, np.array([[eta,m,l,knnor,om,error]])))
                else:
                    data = np.concatenate((data, np.array([[k,m,l,knnor,om,error]])))
            
            data = data.astype(float)
            x1 = data[:,-1]
            min_x1 = np.min(x1)
            max_x1 = np.max(x1)
            kde = stats.gaussian_kde(x1)
            xs = np.linspace(0, max_x1+2, num=500)
            kde.set_bandwidth(bw_method='silverman')
            y2 = kde(xs)
            ax.scatter(x1, np.full(x1.shape, 1 / (4. * x1.size))-c*.01, c=col, marker='.', label='Data point '+labels, s=2)
            ax.plot(xs, y2, label='KDE '+labels, color = col, linewidth=1)
            
            print("The error mean and standar deviation for the montecarlo sampling experiment is: mean: {} , std: {}. This with a total of {} combinations of hyperparameters".format(np.mean(x1) , np.std(x1), len(x1)))
            
    plt.xlim(left=0)
    plt.xticks(fontsize=10)
    ax.legend(prop={"size":8})
    plt.show()
    plt.close()
    if full_edges:
        fig.savefig('../results/' + batch_name + '/'+plot_name+'full_edges_error_kinematic_batch_pdf_full.pdf')
        fig.savefig('../results/' + batch_name + '/'+plot_name+'full_edges_error_kinematic_batch_pdf_full.png')        
    else:
        fig.savefig('../results/'+plot_name+'finite_edges_error_kinematic_batch_pdf_full.pdf')
        fig.savefig('../results/'+plot_name+'finite_edges_error_kinematic_batch_pdf_full.png')


def plot_pdf_time(data_path, full_edges=False, use_eta = False):
    """
    Plot probability distribution functions for montecarlo simulations for time

    Args:
        data_path (str): data folder path
        full_edges (bool, optional): True if full edges approach is applied. Defaults to False.
        use_eta (bool, optional): True if eta is used instead of k_neigh. Defaults to False.
    """

    batch_name = data_path.split('/')[-2] #data_path has to finish in '/'; e.g.'../data/toy_examples_re/'
    if full_edges:
        with open('../results/' + batch_name + '/full_edges_error_kinematic_batch.json', 'r') as fp:
            dict_error_kinematic_batch = json.load(fp)
        data = np.empty(shape=(0,3))
        for key in dict_error_kinematic_batch['two_dofs_unique']:
            knnor = dict_error_kinematic_batch['two_dofs_unique'][key][0][0]
            om = dict_error_kinematic_batch['two_dofs_unique'][key][0][1]
            time = dict_error_kinematic_batch['time_iteration'][key][1][0]
            data = np.concatenate((data, np.array([[knnor,om,time]])))

        data = data.astype(float)
        x1 = data[:,2]
        min_x1 = np.min(x1)
        max_x1 = np.max(x1)
        kde = stats.gaussian_kde(x1)
        xs = np.linspace(0, max_x1+2, num=500)
        kde.set_bandwidth(bw_method='silverman')
        y2 = kde(xs)
        fig, ax = plt.subplots()
        ax.plot(x1, np.full(x1.shape, 1 / (4. * x1.size)), 'c.',
        label='Data point')
        ax.plot(xs, y2, label='KDE', color = 'g')
        ax.set_xlabel("$registration$ $time$ $[s]$")
        ax.set_ylabel("Probability density")
        ax.legend()
        plt.xticks(np.arange(0, int(max_x1+2), step=2))
        plt.xlim(0,int(max_x1+2))
        plt.show()
        plt.close()
        fig.savefig('../results/' + batch_name + '/full_edges_time_kinematic_batch_pdf.pdf')
        fig.savefig('../results/' + batch_name + '/full_edges_time_kinematic_batch_pdf.png')
       
        print("The time mean and standar deviation for the montecarlo sampling experiment is: mean: {} , std: {}. This with a total of {} combinations of hyperparameters".format(np.mean(x1) , np.std(x1), len(x1)))
    
    else:
        with open('../results/' + batch_name + '/finite_edges_error_kinematic_batch.json', 'r') as fp:
            dict_error_kinematic_batch = json.load(fp)
        
        data = np.empty(shape=(0,6))
        for key in dict_error_kinematic_batch['two_dofs_unique']:
            if use_eta:
                eta = dict_error_kinematic_batch['two_dofs_unique'][key][0][0]
            else:
                k = dict_error_kinematic_batch['two_dofs_unique'][key][0][0]
            m = dict_error_kinematic_batch['two_dofs_unique'][key][0][1]
            l = dict_error_kinematic_batch['two_dofs_unique'][key][0][2]
            knnor = dict_error_kinematic_batch['two_dofs_unique'][key][0][3]
            om = dict_error_kinematic_batch['two_dofs_unique'][key][0][4]
            time = dict_error_kinematic_batch['time_iteration'][key][1][0]
            if use_eta:
                data = np.concatenate((data, np.array([[eta,m,l,knnor,om,time]])))
            else:
                data = np.concatenate((data, np.array([[k,m,l,knnor,om,time]])))
        
        data = data.astype(float)
        x1 = data[:,-1]
        min_x1 = np.min(x1)
        max_x1 = np.max(x1)
        kde = stats.gaussian_kde(x1)
        xs = np.linspace(0, max_x1+2, num=500)
        kde.set_bandwidth(bw_method='silverman')
        y2 = kde(xs)
        fig, ax = plt.subplots()
        ax.plot(x1, np.full(x1.shape, 1 / (4. * x1.size)), 'c.',
        label='Data point')
        ax.plot(xs, y2, label='KDE', color = 'g')
        ax.set_xlabel("$registration$ $time$ $[s]$")
        ax.set_ylabel("Probability density")
        ax.legend()
        plt.xticks(np.arange(0, int(max_x1+2), step=5))
        plt.xlim(0,int(max_x1+2))
        plt.show()
        
        fig.savefig('../results/' + batch_name + '/finite_edges_time_kinematic_batch_pdf.pdf')
        fig.savefig('../results/' + batch_name + '/finite_edges_time_kinematic_batch_pdf.png')
       
        print("The time mean and standar deviation for the montecarlo sampling experiment is: mean: {} , std: {}. This with a total of {} combinations of hyperparameters".format(np.mean(x1) , np.std(x1), len(x1)))

def plot_pdf_time_full(data_paths, full_edges = False, use_eta_list = [False,], labels_list = ['',], colors_list = ['#1f77b4',], plot_name=''):
    """
    Plotting probability distribution functions for all the batches for time

    Args:
        data_paths (str): data folder path
        full_edges (bool, optional): True if full edges approach is applied. Defaults to False.
        use_eta (bool, optional): True if eta is used instead of k_neigh. Defaults to False.
        labels_list (list, optional): List with the plot labels. Defaults to ['',].
        colors_list (list, optional): List with the plot colors. Defaults to ['#1f77b4',].
        plot_name (str, optional): additional name to save plots. Defaults to ''.
    """
    
    batch_names = []
    for data_path in data_paths:
        batch_names.append(data_path.split('/')[-2])
    fig, ax = plt.subplots()
    ax.set_xlabel("registration time $[s]$", fontsize = 10)
    ax.set_ylabel("Probability density", fontsize = 10)
        
    c=0
    for use_eta, labels, batch_name, col in zip(use_eta_list, labels_list, batch_names, colors_list):
        c+=1
        if full_edges:
            with open('../results/' + batch_name + '/full_edges_error_kinematic_batch.json', 'r') as fp:
                dict_error_kinematic_batch = json.load(fp)
            data = np.empty(shape=(0,3))
            for key in dict_error_kinematic_batch['two_dofs_unique']:
                knnor = dict_error_kinematic_batch['two_dofs_unique'][key][0][0]
                om = dict_error_kinematic_batch['two_dofs_unique'][key][0][1]
                time = dict_error_kinematic_batch['time_iteration'][key][1][0]
                data = np.concatenate((data, np.array([[knnor,om,time]])))
    
            data = data.astype(float)
            x1 = data[:,2]
            print(x1)
            min_x1 = np.min(x1)
            max_x1 = np.max(x1)
            kde = stats.gaussian_kde(x1)
            xs = np.linspace(0, max_x1+2, num=500)
            kde.set_bandwidth(bw_method='silverman')
            y2 = kde(xs)
            ax.scatter(x1, np.full(x1.shape, 1 / (4. * x1.size))-c*.01, c=col, marker='.', label='Data point '+labels)
            ax.plot(xs, y2, label='KDE '+labels, color = col)
            
            print("The time mean and standar deviation for the montecarlo sampling experiment is: mean: {} , std: {}. This with a total of {} combinations of hyperparameters".format(np.mean(x1) , np.std(x1), len(x1)))
            
        else:
            with open('../results/' + batch_name + '/finite_edges_error_kinematic_batch.json', 'r') as fp:
                dict_error_kinematic_batch = json.load(fp)
            
            data = np.empty(shape=(0,6))
            for key in dict_error_kinematic_batch['two_dofs_unique']:
                if use_eta:
                    eta = dict_error_kinematic_batch['two_dofs_unique'][key][0][0]
                else:
                    k = dict_error_kinematic_batch['two_dofs_unique'][key][0][0]
                m = dict_error_kinematic_batch['two_dofs_unique'][key][0][1]
                l = dict_error_kinematic_batch['two_dofs_unique'][key][0][2]
                knnor = dict_error_kinematic_batch['two_dofs_unique'][key][0][3]
                om = dict_error_kinematic_batch['two_dofs_unique'][key][0][4]
                time = dict_error_kinematic_batch['time_iteration'][key][1][0]
                if use_eta:
                    data = np.concatenate((data, np.array([[eta,m,l,knnor,om,time]])))
                else:
                    data = np.concatenate((data, np.array([[k,m,l,knnor,om,time]])))
            
            data = data.astype(float)
            x1 = data[:,-1]
            min_x1 = np.min(x1)
            max_x1 = np.max(x1)
            kde = stats.gaussian_kde(x1)
            xs = np.linspace(0, max_x1+2, num=500)
            kde.set_bandwidth(bw_method='silverman')
            y2 = kde(xs)
            ax.scatter(x1, np.full(x1.shape, 1 / (4. * x1.size))-c*.03, c=col, marker='.', label='Data point '+labels, s=2)
            ax.plot(xs, y2, label='KDE '+labels, color = col, linewidth=1)
            print("The time mean and standar deviation for the montecarlo sampling experiment is: mean: {} , std: {}. This with a total of {} combinations of hyperparameters".format(np.mean(x1) , np.std(x1), len(x1)))
            
    plt.xlim(left=0)
    plt.xticks(fontsize=10)
    ax.legend(prop={"size":8})
    plt.show()
    plt.close()
    if full_edges:
        fig.savefig('../results/' + batch_name + '/'+plot_name+'full_edges_time_kinematic_batch_pdf_full.pdf')
        fig.savefig('../results/' + batch_name + '/'+plot_name+'full_edges_time_kinematic_batch_pdf_full.png')        
        
    else:
        fig.savefig('../results/'+plot_name+'finite_edges_kinematic_time_pdf_full.pdf')
        fig.savefig('../results/'+plot_name+'finite_edges_kinematic_time_pdf_full.png')
        
def plot_hypers_error(data_path, full_edges=False, use_eta=False):
    """
    Plots error as function of hyperparameters

    Args:
        data_paths (str): data folder path
        full_edges (bool, optional): True if full edges approach is applied. Defaults to False.
        use_eta (bool, optional): True if eta is used instead of k_neigh. Defaults to False.
    """
    
    batch_name = data_path.split('/')[-2] #data_path has to finish in '/'; e.g.'../data/toy_examples_re/'
    
    if full_edges:
        with open('../results/' + batch_name + '/full_edges_error_kinematic_batch.json', 'r') as fp:
            dict_error_kinematic_batch = json.load(fp)
        data = np.empty(shape=(0,3))
        for key in dict_error_kinematic_batch['two_dofs_unique']:
            knnor = dict_error_kinematic_batch['two_dofs_unique'][key][0][0]
            om = dict_error_kinematic_batch['two_dofs_unique'][key][0][1]
            error = dict_error_kinematic_batch['two_dofs_unique'][key][1][0]
            data = np.concatenate((data, np.array([[knnor,om,error]])))
            
        Xs = data[:,0]
        Ys = data[:,1]
        Zs = data[:,2]
        
        #data for mean , std and cv of hyoer params erros
        Xs_unique = np.unique(Xs)        
        Xs_std = []
        Xs_mean = []
        for xs in Xs_unique:
            xs_for_std = Zs[np.where(Xs==xs)[0]]
            Xs_std.append(np.std(xs_for_std))
            Xs_mean.append(np.mean(xs_for_std))
        Xs_std = np.array(Xs_std)
        Xs_mean = np.array(Xs_mean)
        
        Ys_unique = np.unique(Ys)        
        Ys_std = []
        Ys_mean = []
        for ys in Ys_unique:
            ys_for_std = Zs[np.where(Ys==ys)[0]]
            Ys_std.append(np.std(ys_for_std))
            Ys_mean.append(np.mean(ys_for_std))
        Ys_std = np.array(Ys_std)
        Ys_mean = np.array(Ys_mean)
    
        #plot error vs kn
        plt.figure()
        plt.plot(Xs,Zs, 'b.')
        plt.xlabel("$k_n$")
        plt.ylabel("$\|error\|$ $[px]$")       
        plt.tight_layout()
        plt.show()        
        plt.savefig('../results/' + batch_name + '/full_edges_error_kinematic_batch_knn.pdf')
        plt.savefig('../results/' + batch_name + '/full_edges_error_kinematic_batch_knn.png')  
        plt.close()
    
        #plot error vs omega
        plt.figure()
        plt.plot(Ys,Zs, 'b.')
        plt.xlabel("$\omega$")
        plt.ylabel("$\|error\|$ $[px]$")       
        plt.tight_layout()
        plt.show()         
        plt.savefig('../results/' + batch_name + '/full_edges_error_kinematic_batch_omega.pdf')
        plt.savefig('../results/' + batch_name + '/full_edges_error_kinematic_batch_omega.png')    
        plt.close()
        
        #Plot mean, std and cv for hyperparameters error
        
        #k_n#########
        plt.figure()
        plt.plot(Xs_unique, Xs_mean, 'b.')
        plt.plot(Xs_unique, Xs_mean, 'b:')
        plt.xlabel("$k_n$")
        plt.ylabel("$mean$ $\|error\|$ $[px]$")       
        plt.tight_layout()
        plt.show()        
        plt.savefig('../results/' + batch_name + '/full_edges_error_kinematic_batch_kn_mean.pdf')
        plt.savefig('../results/' + batch_name + '/full_edges_error_kinematic_batch_kn_mean.png')
        
        plt.figure()
        plt.plot(Xs_unique, Xs_std, 'g.')
        plt.plot(Xs_unique, Xs_std, 'g:')
        plt.xlabel("$k_n$")
        plt.ylabel("$std$ $\|error\|$ $[px]$")       
        plt.tight_layout()
        plt.show()        
        plt.savefig('../results/' + batch_name + '/full_edges_error_kinematic_batch_kn_std.pdf')
        plt.savefig('../results/' + batch_name + '/full_edges_error_kinematic_batch_kn_std.png')
        
        plt.figure()
        plt.plot(Xs_unique, Xs_std/Xs_mean, 'r.')
        plt.plot(Xs_unique, Xs_std/Xs_mean, 'r:')
        plt.xlabel("$k_n$")
        plt.ylabel("$cv$ $\|error\|$")       
        plt.tight_layout()
        plt.show()        
        plt.savefig('../results/' + batch_name + '/full_edges_error_kinematic_batch_kn_cv.pdf')
        plt.savefig('../results/' + batch_name + '/full_edges_error_kinematic_batch_kn_cv.png') 
        
        
        #k_n#########
        plt.figure()
        plt.plot(Ys_unique, Ys_mean, 'b.')
        plt.xlabel("$\omega$")
        plt.ylabel("$mean$ $\|error\|$ $[px]$")       
        plt.tight_layout()
        plt.show()        
        plt.savefig('../results/' + batch_name + '/full_edges_error_kinematic_batch_omega_mean.pdf')
        plt.savefig('../results/' + batch_name + '/full_edges_error_kinematic_batch_omega_mean.png')
        
        plt.figure()
        plt.plot(Ys_unique, Ys_std, 'g.')
        plt.plot(Ys_unique, Ys_std, 'g:')
        plt.xlabel("$\omega$")
        plt.ylabel("$std$ $\|error\|$ $[px]$")       
        plt.tight_layout()
        plt.show()        
        plt.savefig('../results/' + batch_name + '/full_edges_error_kinematic_batch_omega_std.pdf')
        plt.savefig('../results/' + batch_name + '/full_edges_error_kinematic_batch_omega_std.png')
        
        plt.figure()
        plt.plot(Ys_unique, Ys_std/Ys_mean, 'r.')
        plt.plot(Ys_unique, Ys_std/Ys_mean, 'r:')
        plt.xlabel("$\omega$")
        plt.ylabel("$cv$ $\|error\|$")       
        plt.tight_layout()
        plt.show()        
        plt.savefig('../results/' + batch_name + '/full_edges_error_kinematic_batch_omega_cv.pdf')
        plt.savefig('../results/' + batch_name + '/full_edges_error_kinematic_batch_omega_cv.png') 
        
    else:
        with open('../results/' + batch_name + '/finite_edges_error_kinematic_batch.json', 'r') as fp:
            dict_error_kinematic_batch = json.load(fp)
        
        data = np.empty(shape=(0,6))
        for key in dict_error_kinematic_batch['two_dofs_unique']:
            if use_eta:
                eta = dict_error_kinematic_batch['two_dofs_unique'][key][0][0]
            else:
                k = dict_error_kinematic_batch['two_dofs_unique'][key][0][0]
            m = dict_error_kinematic_batch['two_dofs_unique'][key][0][1]
            l = dict_error_kinematic_batch['two_dofs_unique'][key][0][2]
            knnor = dict_error_kinematic_batch['two_dofs_unique'][key][0][3]
            om = dict_error_kinematic_batch['two_dofs_unique'][key][0][4]
            error = dict_error_kinematic_batch['two_dofs_unique'][key][1][0]
            if use_eta:
                data = np.concatenate((data, np.array([[eta,m,l,knnor,om,error]])))
            else:
                data = np.concatenate((data, np.array([[k,m,l,knnor,om,error]])))
        
        Us = data[:,0].astype(float)
        Vs = data[:,1].astype(float)
        Ws = data[:,2].astype(float)
        Xs = data[:,3].astype(float)
        Ys = data[:,4].astype(float)
        Zs = data[:,5].astype(float)
        
        Us_unique = np.unique(Us)        
        Us_std = []
        Us_mean = []
        for us in Us_unique:
            us_for_std = Zs[np.where(Us==us)[0]]
            Us_std.append(np.std(us_for_std))
            Us_mean.append(np.mean(us_for_std))
        Us_std = np.array(Us_std)
        Us_mean = np.array(Us_mean)
        
        Vs_unique = np.unique(Vs)        
        Vs_std = []
        Vs_mean = []
        for vs in Vs_unique:
            vs_for_std = Zs[np.where(Vs==vs)[0]]
            Vs_std.append(np.std(vs_for_std))
            Vs_mean.append(np.mean(vs_for_std))
        Vs_std = np.array(Vs_std)
        Vs_mean = np.array(Vs_mean)
        
        Ws_unique = np.unique(Ws)        
        Ws_std = []
        Ws_mean = []
        for ws in Ws_unique:
            ws_for_std = Zs[np.where(Ws==ws)[0]]
            Ws_std.append(np.std(ws_for_std))
            Ws_mean.append(np.mean(ws_for_std))
        Ws_std = np.array(Ws_std)
        Ws_mean = np.array(Ws_mean)
        
        Xs_unique = np.unique(Xs)        
        Xs_std = []
        Xs_mean = []
        for xs in Xs_unique:
            xs_for_std = Zs[np.where(Xs==xs)[0]]
            Xs_std.append(np.std(xs_for_std))
            Xs_mean.append(np.mean(xs_for_std))
        Xs_std = np.array(Xs_std)
        Xs_mean = np.array(Xs_mean)
        
        Ys_unique = np.unique(Ys)        
        Ys_std = []
        Ys_mean = []
        for ys in Ys_unique:
            ys_for_std = Zs[np.where(Ys==ys)[0]]
            Ys_std.append(np.std(ys_for_std))
            Ys_mean.append(np.mean(ys_for_std))
        Ys_std = np.array(Ys_std)
        Ys_mean = np.array(Ys_mean)
                
        #plot error vs kn or eta
        plt.figure()
        plt.plot(Us,Zs, 'b.')
        if use_eta:
            plt.xlabel("$\eta$")
        else:
            plt.xlabel("$k$")
        plt.ylabel("$\|error\|$ $[px]$")       
        plt.tight_layout()
        plt.show()
        if use_eta:        
            plt.savefig('../results/' + batch_name + '/finite_edges_error_kinematic_batch_eta.pdf')
            plt.savefig('../results/' + batch_name + '/finite_edges_error_kinematic_batch_eta.png') 
        else:
            plt.savefig('../results/' + batch_name + '/finite_edges_error_kinematic_batch_k.pdf')
            plt.savefig('../results/' + batch_name + '/finite_edges_error_kinematic_batch_k.png') 

        #plot error vs mu
        plt.figure()
        plt.plot(Vs,Zs, 'b.')
        plt.xlabel("$\mu$")
        plt.ylabel("$\|error\|$ $[px]$")       
        plt.tight_layout()
        plt.show()        
        plt.savefig('../results/' + batch_name + '/finite_edges_error_kinematic_batch_mu.pdf')
        plt.savefig('../results/' + batch_name + '/finite_edges_error_kinematic_batch_mu.png')    
    
        #plot error vs lambda
        plt.figure()
        plt.plot(Ws,Zs, 'b.')
        plt.xlabel("$\lambda$")
        plt.ylabel("$\|error\|$ $[px]$")       
        plt.tight_layout()
        plt.show()       
        plt.savefig('../results/' + batch_name + '/finite_edges_error_kinematic_batch_lambda.pdf')
        plt.savefig('../results/' + batch_name + '/finite_edges_error_kinematic_batch_lambda.png')  

        #plot error vs knn
        plt.figure()
        plt.plot(Xs,Zs, 'b.')
        plt.xlabel("$k_n$")
        plt.ylabel("$\|error\|$ $[px]$")       
        plt.tight_layout()
        plt.show()         
        plt.savefig('../results/' + batch_name + '/finite_edges_error_kinematic_batch_knn.pdf')
        plt.savefig('../results/' + batch_name + '/finite_edges_error_kinematic_batch_knn.png')    
    
        #plot error vs omega
        plt.figure()
        plt.plot(Ys,Zs, 'b.')
        plt.xlabel("$\omega$")
        plt.ylabel("$\|error\|$ $[px]$")       
        plt.tight_layout()
        plt.show()        
        plt.savefig('../results/' + batch_name + '/finite_edges_error_kinematic_batch_omega.pdf')
        plt.savefig('../results/' + batch_name + '/finite_edges_error_kinematic_batch_omega.png')  
        
        #Mean, std and coef of variance plots###############3
        #K########
        plt.figure()
        plt.plot(Us_unique, Us_mean, 'b.')
        plt.plot(Us_unique, Us_mean, 'b:')
        if use_eta:
            plt.xlabel("$\eta$")
        else:
            plt.xlabel("$k$")
        plt.ylabel("$mean$ $\|error\|$ $[px]$")       
        plt.tight_layout()
        plt.show()  
        if use_eta:
            plt.savefig('../results/' + batch_name + '/finite_edges_error_kinematic_batch_eta_mean.pdf')
            plt.savefig('../results/' + batch_name + '/finite_edges_error_kinematic_batch_eta_mean.png')
        else:
            plt.savefig('../results/' + batch_name + '/finite_edges_error_kinematic_batch_k_mean.pdf')
            plt.savefig('../results/' + batch_name + '/finite_edges_error_kinematic_batch_k_mean.png')
        
        plt.figure()
        plt.plot(Us_unique, Us_std, 'g.')
        plt.plot(Us_unique, Us_std, 'g:')
        if use_eta:
            plt.xlabel("$\eta$")
        else:
            plt.xlabel("$k$")
        plt.ylabel("$std$ $\|error\|$ $[px]$")       
        plt.tight_layout()
        plt.show()        
        if use_eta:
            plt.savefig('../results/' + batch_name + '/finite_edges_error_kinematic_batch_eta_std.pdf')
            plt.savefig('../results/' + batch_name + '/finite_edges_error_kinematic_batch_eta_std.png')
        else:
             plt.savefig('../results/' + batch_name + '/finite_edges_error_kinematic_batch_k_std.pdf')
             plt.savefig('../results/' + batch_name + '/finite_edges_error_kinematic_batch_k_std.png')
        
        plt.figure()
        plt.plot(Us_unique, Us_std/Us_mean, 'r.')
        plt.plot(Us_unique, Us_std/Us_mean, 'r:')
        if use_eta:
            plt.xlabel("$\eta$")
        else:
            plt.xlabel("$k$")
        plt.ylabel("$cv$ $\|error\|$")       
        plt.tight_layout()
        plt.show()        
        if use_eta:
            plt.savefig('../results/' + batch_name + '/finite_edges_error_kinematic_batch_eta_cv.pdf')
            plt.savefig('../results/' + batch_name + '/finite_edges_error_kinematic_batch_eta_cv.png')  
        else:
            plt.savefig('../results/' + batch_name + '/finite_edges_error_kinematic_batch_k_cv.pdf')
            plt.savefig('../results/' + batch_name + '/finite_edges_error_kinematic_batch_k_cv.png')  
        
        #Mu#########
        plt.figure()
        plt.plot(Vs_unique, Vs_mean, 'b.')
        plt.plot(Vs_unique, Vs_mean, 'b:')
        plt.xlabel("$\mu$")
        plt.ylabel("$mean$ $\|error\|$ $[px]$")           
        plt.tight_layout()
        plt.show()        
        plt.savefig('../results/' + batch_name + '/finite_edges_error_kinematic_batch_mu_mean.pdf')
        plt.savefig('../results/' + batch_name + '/finite_edges_error_kinematic_batch_mu_mean.png')
        
        plt.figure()
        plt.plot(Vs_unique, Vs_std, 'g.')
        plt.plot(Vs_unique, Vs_std, 'g:')
        plt.xlabel("$\mu$")
        plt.ylabel("$std$ $\|error\|$ $[px]$")       
        plt.tight_layout()
        plt.show()        
        plt.savefig('../results/' + batch_name + '/finite_edges_error_kinematic_batch_mu_std.pdf')
        plt.savefig('../results/' + batch_name + '/finite_edges_error_kinematic_batch_mu_std.png')
        
        plt.figure()
        plt.plot(Vs_unique, Vs_std/Vs_mean, 'r.')
        plt.plot(Vs_unique, Vs_std/Vs_mean, 'r:')
        plt.xlabel("$\mu$")
        plt.ylabel("$cv$ $\|error\|$")         
        plt.tight_layout()
        plt.show()        
        plt.savefig('../results/' + batch_name + '/finite_edges_error_kinematic_batch_mu_cv.pdf')
        plt.savefig('../results/' + batch_name + '/finite_edges_error_kinematic_batch_mu_cv.png')  
        
        #lambda#########
        plt.figure()
        plt.plot(Ws_unique, Ws_mean, 'b.')
        plt.plot(Ws_unique, Ws_mean, 'b:')
        plt.xlabel("$\lambda$")
        plt.ylabel("$mean$ $\|error\|$ $[px]$")         
        plt.tight_layout()
        plt.show()        
        plt.savefig('../results/' + batch_name + '/finite_edges_error_kinematic_batch_lambda_mean.pdf')
        plt.savefig('../results/' + batch_name + '/finite_edges_error_kinematic_batch_lambda_mean.png')
        
        plt.figure()
        plt.plot(Ws_unique, Ws_std, 'g.')
        plt.plot(Ws_unique, Ws_std, 'g:')
        plt.xlabel("$\lambda$")
        plt.ylabel("$std$ $\|error\|$ $[px]$")       
        plt.tight_layout()
        plt.show()        
        plt.savefig('../results/' + batch_name + '/finite_edges_error_kinematic_batch_lambda_std.pdf')
        plt.savefig('../results/' + batch_name + '/finite_edges_error_kinematic_batch_lambda_std.png')
        
        plt.figure()
        plt.plot(Ws_unique, Ws_std/Ws_mean, 'r.')
        plt.plot(Ws_unique, Ws_std/Ws_mean, 'r:')
        plt.xlabel("$\lambda$")
        plt.ylabel("$cv$ $\|error\|$")             
        plt.tight_layout()
        plt.show()        
        plt.savefig('../results/' + batch_name + '/finite_edges_error_kinematic_batch_lambda_cv.pdf')
        plt.savefig('../results/' + batch_name + '/finite_edges_error_kinematic_batch_lambda_cv.png') 
        
        
        #k_n#########
        plt.figure()
        plt.plot(Xs_unique, Xs_mean, 'b.')
        plt.plot(Xs_unique, Xs_mean, 'b:')
        plt.xlabel("$k_n$")
        plt.ylabel("$mean$ $\|error\|$ $[px]$")            
        plt.tight_layout()
        plt.show()        
        plt.savefig('../results/' + batch_name + '/finite_edges_error_kinematic_batch_kn_mean.pdf')
        plt.savefig('../results/' + batch_name + '/finite_edges_error_kinematic_batch_kn_mean.png')
        
        plt.figure()
        plt.plot(Xs_unique, Xs_std, 'g.')
        plt.plot(Xs_unique, Xs_std, 'g:')
        plt.xlabel("$k_n$")
        plt.ylabel("$std$ $\|error\|$ $[px]$")          
        plt.tight_layout()
        plt.show()        
        plt.savefig('../results/' + batch_name + '/finite_edges_error_kinematic_batch_kn_std.pdf')
        plt.savefig('../results/' + batch_name + '/finite_edges_error_kinematic_batch_kn_std.png')
        
        plt.figure()
        plt.plot(Xs_unique, Xs_std/Xs_mean, 'r.')
        plt.plot(Xs_unique, Xs_std/Xs_mean, 'r:')
        plt.xlabel("$k_n$")
        plt.ylabel("$cv$ $\|error\|$")       
        plt.tight_layout()
        plt.show()        
        plt.savefig('../results/' + batch_name + '/finite_edges_error_kinematic_batch_kn_cv.pdf')
        plt.savefig('../results/' + batch_name + '/finite_edges_error_kinematic_batch_kn_cv.png') 
        
        
        #k_n#########
        plt.figure()
        plt.plot(Ys_unique, Ys_mean, 'b.')
        plt.xlabel("$\omega$")
        plt.ylabel("$mean$ $\|error\|$ $[px]$")            
        plt.tight_layout()
        plt.show()        
        plt.savefig('../results/' + batch_name + '/finite_edges_error_kinematic_batch_omega_mean.pdf')
        plt.savefig('../results/' + batch_name + '/finite_edges_error_kinematic_batch_omega_mean.png')
        
        plt.figure()
        plt.plot(Ys_unique, Ys_std, 'g.')
        plt.plot(Ys_unique, Ys_std, 'g:')
        plt.xlabel("$\omega$")
        plt.ylabel("$std$ $\|error\|$ $[px]$")        
        plt.tight_layout()
        plt.show()        
        plt.savefig('../results/' + batch_name + '/finite_edges_error_kinematic_batch_omega_std.pdf')
        plt.savefig('../results/' + batch_name + '/finite_edges_error_kinematic_batch_omega_std.png')
        
        plt.figure()
        plt.plot(Ys_unique, Ys_std/Ys_mean, 'r.')
        plt.plot(Ys_unique, Ys_std/Ys_mean, 'r:')
        plt.xlabel("$\omega$")
        plt.ylabel("$cv$ $\|error\|$")         
        plt.tight_layout()
        plt.show()        
        plt.savefig('../results/' + batch_name + '/finite_edges_error_kinematic_batch_omega_cv.pdf')
        plt.savefig('../results/' + batch_name + '/finite_edges_error_kinematic_batch_omega_cv.png') 


def plot_hypers_error_full(data_paths, use_eta_list=[False,], parameter = 'k', labels_list = ['',], colors_list = ['#1f77b4',], var = 'error', plot_name=''):
    """
    Plots error as function of hyperparameters for full batches

    Args:
        data_paths (str): data folder path
        use_eta (bool, optional): True if eta is used instead of k_neigh. Defaults to False.
        parameter (str, optional): Parameter to be plotted. k,m,l,eta. Defaults to 'k'.
        labels_list (list, optional): List with the plot labels. Defaults to ['',].
        colors_list (list, optional): List with the plot colors. Defaults to ['#1f77b4',].
        var (str, optional): variable to be plotted, error or time. Defaults to 'error'.
        plot_name (str, optional): additional name to save plots. Defaults to ''.
    """  
    
    batch_names = []
    for data_path in data_paths:
        batch_names.append(data_path.split('/')[-2])
        
    fig = plt.figure()
    if var == 'error':
        plt.ylabel("$mean$ $\|error\|$ $[px]$", fontsize = 10)
    elif var == 'time':
        plt.ylabel("$mean$ registration time $[s]$", fontsize = 10)
    c = 0            
    for use_eta, labels, batch_name, col in zip(use_eta_list, labels_list, batch_names, colors_list):
        c+=1
        print(batch_name)
        with open('../results/' + batch_name + '/finite_edges_error_kinematic_batch.json', 'r') as fp:
            dict_error_kinematic_batch = json.load(fp)
        
        data = np.empty(shape=(0,6))
        for key in dict_error_kinematic_batch['two_dofs_unique']:
            if use_eta:
                eta = dict_error_kinematic_batch['two_dofs_unique'][key][0][0]
            else:
                k = dict_error_kinematic_batch['two_dofs_unique'][key][0][0]
            m = dict_error_kinematic_batch['two_dofs_unique'][key][0][1]
            l = dict_error_kinematic_batch['two_dofs_unique'][key][0][2]
            knnor = dict_error_kinematic_batch['two_dofs_unique'][key][0][3]
            om = dict_error_kinematic_batch['two_dofs_unique'][key][0][4]
            if var == 'error':
                var_p = dict_error_kinematic_batch['two_dofs_unique'][key][1][0]
            elif var == 'time':
                var_p = dict_error_kinematic_batch['time_iteration'][key][1][0]
            if use_eta:
                data = np.concatenate((data, np.array([[eta,m,l,knnor,om,var_p]])))
            else:
                data = np.concatenate((data, np.array([[k,m,l,knnor,om,var_p]])))
        
        Us = data[:,0].astype(float)
        Vs = data[:,1].astype(float)
        Ws = data[:,2].astype(float)
        Xs = data[:,3].astype(float)
        Ys = data[:,4].astype(float)
        Zs = data[:,5].astype(float)
        
        Us_unique = np.unique(Us)        
        Us_std = []
        Us_mean = []
        for us in Us_unique:
            us_for_std = Zs[np.where(Us==us)[0]]
            Us_std.append(np.std(us_for_std))
            Us_mean.append(np.mean(us_for_std))
        Us_std = np.array(Us_std)
        Us_mean = np.array(Us_mean)
        
        Vs_unique = np.unique(Vs)        
        Vs_std = []
        Vs_mean = []
        for vs in Vs_unique:
            vs_for_std = Zs[np.where(Vs==vs)[0]]
            Vs_std.append(np.std(vs_for_std))
            Vs_mean.append(np.mean(vs_for_std))
        Vs_std = np.array(Vs_std)
        Vs_mean = np.array(Vs_mean)
        
        Ws_unique = np.unique(Ws)        
        Ws_std = []
        Ws_mean = []
        for ws in Ws_unique:
            ws_for_std = Zs[np.where(Ws==ws)[0]]
            Ws_std.append(np.std(ws_for_std))
            Ws_mean.append(np.mean(ws_for_std))
        Ws_std = np.array(Ws_std)
        Ws_mean = np.array(Ws_mean)
        
        Xs_unique = np.unique(Xs)        
        Xs_std = []
        Xs_mean = []
        for xs in Xs_unique:
            xs_for_std = Zs[np.where(Xs==xs)[0]]
            Xs_std.append(np.std(xs_for_std))
            Xs_mean.append(np.mean(xs_for_std))
        Xs_std = np.array(Xs_std)
        Xs_mean = np.array(Xs_mean)
        
        Ys_unique = np.unique(Ys)        
        Ys_std = []
        Ys_mean = []
        for ys in Ys_unique:
            ys_for_std = Zs[np.where(Ys==ys)[0]]
            Ys_std.append(np.std(ys_for_std))
            Ys_mean.append(np.mean(ys_for_std))
        Ys_std = np.array(Ys_std)
        Ys_mean = np.array(Ys_mean)
                
        
        #Mean, std and coef of variance plots###############3
        if parameter=='k' or parameter=='eta':
        #eta or k
            plt.scatter(Us_unique, Us_mean, c = col, marker='.')
            plt.plot(Us_unique, Us_mean, c = col, linestyle=':', label = labels)
        elif parameter=='mu':
        #Mu#########
            plt.scatter(Vs_unique, Vs_mean, c = col, marker='.')
            plt.plot(Vs_unique, Vs_mean, c = col, linestyle=':', label = labels)
        elif parameter=='lambda':
        #lambda#########
            plt.scatter(Ws_unique, Ws_mean, c = col, marker='.')
            plt.plot(Ws_unique, Ws_mean, c = col, linestyle=':', label = labels)
        
    plt.xticks(fontsize=10)
    plt.legend(prop={"size":8})
    
    if parameter=='k':
        plt.xlabel("$k$", fontsize=10)
        plt.tight_layout()
        plt.savefig('../results/'+plot_name+'finite_edges_'+var+'_kinematic_batch_k_mean_full_scenarios.pdf')
        plt.savefig('../results/'+plot_name+'finite_edges_'+var+'_kinematic_batch_k_mean_full_scenarios.png')
    elif parameter=='eta':
        plt.xlabel("$\eta$", fontsize=10)
        plt.tight_layout()
        plt.savefig('../results/'+plot_name+'finite_edges_'+var+'_kinematic_batch_eta_mean_full_scenarios.pdf')
        plt.savefig('../results/'+plot_name+'finite_edges_'+var+'_kinematic_batch_eta_mean_full_scenarios.png')
    elif parameter=='mu':
        plt.xlabel("$\mu$", fontsize=10)
        plt.tight_layout()
        plt.savefig('../results/'+plot_name+'finite_edges_'+var+'_kinematic_batch_mu_mean_full_scenarios.pdf')
        plt.savefig('../results/'+plot_name+'finite_edges_'+var+'_kinematic_batch_mu_mean_full_scenarios.png')
    elif parameter=='lambda':
        plt.xlabel("$\lambda$", fontsize=10)
        plt.tight_layout()
        plt.savefig('../results/'+plot_name+'finite_edges_'+var+'_kinematic_batch_lambda_mean_full_scenarios.pdf')
        plt.savefig('../results/'+plot_name+'finite_edges_'+var+'_kinematic_batch_lambda_mean_full_scenarios.png')



def plot_hypers_time(data_path, full_edges=False, use_eta=False):
    """
    Plots time as function of hyperparameters

    Args:
        data_paths (str): data folder path
        full_edges (bool, optional): True if full edges approach is applied. Defaults to False.
        use_eta (bool, optional): True if eta is used instead of k_neigh. Defaults to False.
    """    
    batch_name = data_path.split('/')[-2] #data_path has to finish in '/'; e.g.'../data/toy_examples_re/'
    
    if full_edges:
        with open('../results/' + batch_name + '/full_edges_error_kinematic_batch.json', 'r') as fp:
            dict_error_kinematic_batch = json.load(fp)
        data = np.empty(shape=(0,3))
        for key in dict_error_kinematic_batch['two_dofs_unique']:
            knnor = dict_error_kinematic_batch['two_dofs_unique'][key][0][0]
            om = dict_error_kinematic_batch['two_dofs_unique'][key][0][1]
            time = dict_error_kinematic_batch['time_iteration'][key][1][0]
            data = np.concatenate((data, np.array([[knnor,om,time]])))

            
        Xs = data[:,0]
        Ys = data[:,1]
        Zs = data[:,2]
        
        #data for mean , std and cv of hyoer params erros
        Xs_unique = np.unique(Xs)        
        Xs_std = []
        Xs_mean = []
        for xs in Xs_unique:
            xs_for_std = Zs[np.where(Xs==xs)[0]]
            Xs_std.append(np.std(xs_for_std))
            Xs_mean.append(np.mean(xs_for_std))
        Xs_std = np.array(Xs_std)
        Xs_mean = np.array(Xs_mean)
        
        Ys_unique = np.unique(Ys)        
        Ys_std = []
        Ys_mean = []
        for ys in Ys_unique:
            ys_for_std = Zs[np.where(Ys==ys)[0]]
            Ys_std.append(np.std(ys_for_std))
            Ys_mean.append(np.mean(ys_for_std))
        Ys_std = np.array(Ys_std)
        Ys_mean = np.array(Ys_mean)
         
    
        #plot error vs kn
        plt.figure()
        #ax = fig.add_subplot(111)
        plt.plot(Xs,Zs, 'b.')
        plt.xlabel("$k_n$")
        plt.ylabel("$registration$ $time$  $[s]$")       
        plt.tight_layout()
        plt.show()        
        plt.savefig('../results/' + batch_name + '/full_edges_time_kinematic_batch_knn.pdf')
        plt.savefig('../results/' + batch_name + '/full_edges_time_kinematic_batch_knn.png')  
        plt.close()
    
        #plot error vs omega
        plt.figure()
        plt.plot(Ys,Zs, 'b.')
        plt.xlabel("$\omega$")
        plt.ylabel("$registration$ $time$  $[s]$")       
        plt.tight_layout()
        plt.show()         
        plt.savefig('../results/' + batch_name + '/full_edges_time_kinematic_batch_omega.pdf')
        plt.savefig('../results/' + batch_name + '/full_edges_time_kinematic_batch_omega.png')    
        plt.close()
        
        #Plot mean, std and cv for hyperparameters error
        #k_n#########
        plt.figure()
        plt.plot(Xs_unique, Xs_mean, 'b.')
        plt.plot(Xs_unique, Xs_mean, 'b:')
        plt.xlabel("$k_n$")
        plt.ylabel("$mean$ $registration$ $time$ $[s]$")       
        plt.tight_layout()
        plt.show()        
        plt.savefig('../results/' + batch_name + '/full_edges_time_kinematic_batch_kn_mean.pdf')
        plt.savefig('../results/' + batch_name + '/full_edges_time_kinematic_batch_kn_mean.png')
        
        plt.figure()
        plt.plot(Xs_unique, Xs_std, 'g.')
        plt.plot(Xs_unique, Xs_std, 'g:')
        plt.xlabel("$k_n$")
        plt.ylabel("$std$ $registration$ $time$ $[s]$")              
        plt.tight_layout()
        plt.show()        
        plt.savefig('../results/' + batch_name + '/full_edges_time_kinematic_batch_kn_std.pdf')
        plt.savefig('../results/' + batch_name + '/full_edges_time_kinematic_batch_kn_std.png')
        
        plt.figure()
        plt.plot(Xs_unique, Xs_std/Xs_mean, 'r.')
        plt.plot(Xs_unique, Xs_std/Xs_mean, 'r:')
        plt.xlabel("$k_n$")
        plt.ylabel("$cv$ $registration$ $time$ $[s]$")              
        plt.tight_layout()
        plt.show()        
        plt.savefig('../results/' + batch_name + '/full_edges_time_kinematic_batch_kn_cv.pdf')
        plt.savefig('../results/' + batch_name + '/full_edges_time_kinematic_batch_kn_cv.png') 
        
        #k_n#########
        plt.figure()
        plt.plot(Ys_unique, Ys_mean, 'b.')
        plt.xlabel("$\omega$")
        plt.ylabel("$mean$ $registration$ $time$ $[s]$")        
        plt.tight_layout()
        plt.show()        
        plt.savefig('../results/' + batch_name + '/full_edges_time_kinematic_batch_omega_mean.pdf')
        plt.savefig('../results/' + batch_name + '/full_edges_time_kinematic_batch_omega_mean.png')
        
        plt.figure()
        plt.plot(Ys_unique, Ys_std, 'g.')
        plt.plot(Ys_unique, Ys_std, 'g:')
        plt.xlabel("$\omega$")
        plt.ylabel("$std$ $registration$ $time$ $[s]$")              
        plt.tight_layout()
        plt.show()        
        plt.savefig('../results/' + batch_name + '/full_edges_time_kinematic_batch_omega_std.pdf')
        plt.savefig('../results/' + batch_name + '/full_edges_time_kinematic_batch_omega_std.png')
        
        plt.figure()
        plt.plot(Ys_unique, Ys_std/Ys_mean, 'r.')
        plt.plot(Ys_unique, Ys_std/Ys_mean, 'r:')
        plt.xlabel("$\omega$")
        plt.ylabel("$cv$ $registration$ $time$ $[s]$")        
        plt.tight_layout()
        plt.show()        
        plt.savefig('../results/' + batch_name + '/full_edges_time_kinematic_batch_omega_cv.pdf')
        plt.savefig('../results/' + batch_name + '/full_edges_time_kinematic_batch_omega_cv.png') 
        
    else:
        with open('../results/' + batch_name + '/finite_edges_error_kinematic_batch.json', 'r') as fp:
            dict_error_kinematic_batch = json.load(fp)
        
        data = np.empty(shape=(0,6))
        for key in dict_error_kinematic_batch['two_dofs_unique']:
            if use_eta:
                eta = dict_error_kinematic_batch['two_dofs_unique'][key][0][0]
            else:
                k = dict_error_kinematic_batch['two_dofs_unique'][key][0][0]
            m = dict_error_kinematic_batch['two_dofs_unique'][key][0][1]
            l = dict_error_kinematic_batch['two_dofs_unique'][key][0][2]
            knnor = dict_error_kinematic_batch['two_dofs_unique'][key][0][3]
            om = dict_error_kinematic_batch['two_dofs_unique'][key][0][4]
            time = dict_error_kinematic_batch['time_iteration'][key][1][0]
            if use_eta:
                data = np.concatenate((data, np.array([[eta,m,l,knnor,om,time]])))
            else:
                data = np.concatenate((data, np.array([[k,m,l,knnor,om,time]])))
        
        Us = data[:,0].astype(float)
        Vs = data[:,1].astype(float)
        Ws = data[:,2].astype(float)
        Xs = data[:,3].astype(float)
        Ys = data[:,4].astype(float)
        Zs = data[:,5].astype(float)
        
        Us_unique = np.unique(Us)        
        Us_std = []
        Us_mean = []
        for us in Us_unique:
            us_for_std = Zs[np.where(Us==us)[0]]
            Us_std.append(np.std(us_for_std))
            Us_mean.append(np.mean(us_for_std))
        Us_std = np.array(Us_std)
        Us_mean = np.array(Us_mean)
        
        Vs_unique = np.unique(Vs)        
        Vs_std = []
        Vs_mean = []
        for vs in Vs_unique:
            vs_for_std = Zs[np.where(Vs==vs)[0]]
            Vs_std.append(np.std(vs_for_std))
            Vs_mean.append(np.mean(vs_for_std))
        Vs_std = np.array(Vs_std)
        Vs_mean = np.array(Vs_mean)
        
        Ws_unique = np.unique(Ws)        
        Ws_std = []
        Ws_mean = []
        for ws in Ws_unique:
            ws_for_std = Zs[np.where(Ws==ws)[0]]
            Ws_std.append(np.std(ws_for_std))
            Ws_mean.append(np.mean(ws_for_std))
        Ws_std = np.array(Ws_std)
        Ws_mean = np.array(Ws_mean)
        
        Xs_unique = np.unique(Xs)        
        Xs_std = []
        Xs_mean = []
        for xs in Xs_unique:
            xs_for_std = Zs[np.where(Xs==xs)[0]]
            Xs_std.append(np.std(xs_for_std))
            Xs_mean.append(np.mean(xs_for_std))
        Xs_std = np.array(Xs_std)
        Xs_mean = np.array(Xs_mean)
        
        Ys_unique = np.unique(Ys)        
        Ys_std = []
        Ys_mean = []
        for ys in Ys_unique:
            ys_for_std = Zs[np.where(Ys==ys)[0]]
            Ys_std.append(np.std(ys_for_std))
            Ys_mean.append(np.mean(ys_for_std))
        Ys_std = np.array(Ys_std)
        Ys_mean = np.array(Ys_mean)
  
        #plot time vs kn or eta
        plt.figure()
        plt.plot(Us,Zs, 'b.')
        if use_eta:
            plt.xlabel("$\eta$")
        else:
            plt.xlabel("$k$")
        plt.ylabel("$registration$ $time$  $[s]$")       
        plt.tight_layout()
        plt.show()
        if use_eta:        
            plt.savefig('../results/' + batch_name + '/finite_edges_time_kinematic_batch_eta.pdf')
            plt.savefig('../results/' + batch_name + '/finite_edges_time_kinematic_batch_eta.png') 
        else:
            plt.savefig('../results/' + batch_name + '/finite_edges_time_kinematic_batch_k.pdf')
            plt.savefig('../results/' + batch_name + '/finite_edges_time_kinematic_batch_k.png') 


        #plot error vs mu
        plt.figure()
        plt.plot(Vs,Zs, 'b.')
        plt.xlabel("$\mu$")
        plt.ylabel("$registration$ $time$  $[s]$")       
        plt.tight_layout()
        plt.show()        
        plt.savefig('../results/' + batch_name + '/finite_edges_time_kinematic_batch_mu.pdf')
        plt.savefig('../results/' + batch_name + '/finite_edges_time_kinematic_batch_mu.png')    
    
        #plot error vs lambda
        plt.figure()
        plt.plot(Ws,Zs, 'b.')
        plt.xlabel("$\lambda$")
        plt.ylabel("$registration$ $time$  $[s]$")       
        plt.tight_layout()
        plt.show()       
        plt.savefig('../results/' + batch_name + '/finite_edges_time_kinematic_batch_lambda.pdf')
        plt.savefig('../results/' + batch_name + '/finite_edges_time_kinematic_batch_lambda.png')  

        #plot error vs knn
        plt.figure()
        plt.plot(Xs,Zs, 'b.')
        plt.xlabel("$k_n$")
        plt.ylabel("$registration$ $time$  $[s]$")       
        plt.tight_layout()
        plt.show()         
        plt.savefig('../results/' + batch_name + '/finite_edges_time_kinematic_batch_knn.pdf')
        plt.savefig('../results/' + batch_name + '/finite_edges_time_kinematic_batch_knn.png')    
    
        #plot error vs omega
        plt.figure()
        plt.plot(Ys,Zs, 'b.')
        plt.xlabel("$\omega$")
        plt.ylabel("$registration$ $time$  $[s]$")         
        plt.tight_layout()
        plt.show()        
        plt.savefig('../results/' + batch_name + '/finite_edges_time_kinematic_batch_omega.pdf')
        plt.savefig('../results/' + batch_name + '/finite_edges_time_kinematic_batch_omega.png')  
        
        #Mean, std and coef of variance plots###############3
        
        #K########
        plt.figure()
        plt.plot(Us_unique, Us_mean, 'b.')
        plt.plot(Us_unique, Us_mean, 'b:')
        if use_eta:
            plt.xlabel("$\eta$")
        else:
            plt.xlabel("$k$")
        plt.ylabel("$mean$ $registration$ $time$ $[s]$")        
        plt.tight_layout()
        plt.show()  
        if use_eta:
            plt.savefig('../results/' + batch_name + '/finite_edges_time_kinematic_batch_eta_mean.pdf')
            plt.savefig('../results/' + batch_name + '/finite_edges_time_kinematic_batch_eta_mean.png')
        else:
            plt.savefig('../results/' + batch_name + '/finite_edges_time_kinematic_batch_k_mean.pdf')
            plt.savefig('../results/' + batch_name + '/finite_edges_time_kinematic_batch_k_mean.png')
        
        plt.figure()
        plt.plot(Us_unique, Us_std, 'g.')
        plt.plot(Us_unique, Us_std, 'g:')
        if use_eta:
            plt.xlabel("$\eta$")
        else:
            plt.xlabel("$k$")
        plt.ylabel("$std$ $registration$ $time$ $[s]$")              
        plt.tight_layout()
        plt.show()        
        if use_eta:
            plt.savefig('../results/' + batch_name + '/finite_edges_time_kinematic_batch_eta_std.pdf')
            plt.savefig('../results/' + batch_name + '/finite_edges_time_kinematic_batch_eta_std.png')
        else:
             plt.savefig('../results/' + batch_name + '/finite_edges_time_kinematic_batch_k_std.pdf')
             plt.savefig('../results/' + batch_name + '/finite_edges_time_kinematic_batch_k_std.png')
        
        plt.figure()
        plt.plot(Us_unique, Us_std/Us_mean, 'r.')
        plt.plot(Us_unique, Us_std/Us_mean, 'r:')
        if use_eta:
            plt.xlabel("$\eta$")
        else:
            plt.xlabel("$k$")
        plt.ylabel("$cv$ $registration$ $time$ $[s]$")          
        plt.tight_layout()
        plt.show()        
        if use_eta:
            plt.savefig('../results/' + batch_name + '/finite_edges_time_kinematic_batch_eta_cv.pdf')
            plt.savefig('../results/' + batch_name + '/finite_edges_time_kinematic_batch_eta_cv.png')  
        else:
            plt.savefig('../results/' + batch_name + '/finite_edges_time_kinematic_batch_k_cv.pdf')
            plt.savefig('../results/' + batch_name + '/finite_edges_time_kinematic_batch_k_cv.png')  
        
        #Mu#########
        plt.figure()
        plt.plot(Vs_unique, Vs_mean, 'b.')
        plt.plot(Vs_unique, Vs_mean, 'b:')
        plt.xlabel("$\mu$")
        plt.ylabel("$mean$ $registration$ $time$ $[s]$")        
        plt.tight_layout()
        plt.show()        
        plt.savefig('../results/' + batch_name + '/finite_edges_time_kinematic_batch_mu_mean.pdf')
        plt.savefig('../results/' + batch_name + '/finite_edges_time_kinematic_batch_mu_mean.png')
        
        plt.figure()
        plt.plot(Vs_unique, Vs_std, 'g.')
        plt.plot(Vs_unique, Vs_std, 'g:')
        plt.xlabel("$\mu$")
        plt.ylabel("$std$ $registration$ $time$ $[s]$")        
        plt.tight_layout()
        plt.show()        
        plt.savefig('../results/' + batch_name + '/finite_edges_time_kinematic_batch_mu_std.pdf')
        plt.savefig('../results/' + batch_name + '/finite_edges_time_kinematic_batch_mu_std.png')
        
        plt.figure()
        plt.plot(Vs_unique, Vs_std/Vs_mean, 'r.')
        plt.plot(Vs_unique, Vs_std/Vs_mean, 'r:')
        plt.xlabel("$\mu$")
        plt.ylabel("$cv$ $registration$ $time$ $[s]$")          
        plt.tight_layout()
        plt.show()        
        plt.savefig('../results/' + batch_name + '/finite_edges_time_kinematic_batch_mu_cv.pdf')
        plt.savefig('../results/' + batch_name + '/finite_edges_time_kinematic_batch_mu_cv.png')  
        
        #lambda#########
        plt.figure()
        plt.plot(Ws_unique, Ws_mean, 'b.')
        plt.plot(Ws_unique, Ws_mean, 'b:')
        plt.xlabel("$\lambda$")
        plt.ylabel("$mean$ $registration$ $time$ $[s]$")        
        plt.tight_layout()
        plt.show()        
        plt.savefig('../results/' + batch_name + '/finite_edges_time_kinematic_batch_lambda_mean.pdf')
        plt.savefig('../results/' + batch_name + '/finite_edges_time_kinematic_batch_lambda_mean.png')
        
        plt.figure()
        plt.plot(Ws_unique, Ws_std, 'g.')
        plt.plot(Ws_unique, Ws_std, 'g:')
        plt.xlabel("$\lambda$")
        plt.ylabel("$std$ $registration$ $time$ $[s]$")        
        plt.tight_layout()
        plt.show()        
        plt.savefig('../results/' + batch_name + '/finite_edges_time_kinematic_batch_lambda_std.pdf')
        plt.savefig('../results/' + batch_name + '/finite_edges_time_kinematic_batch_lambda_std.png')
        
        plt.figure()
        plt.plot(Ws_unique, Ws_std/Ws_mean, 'r.')
        plt.plot(Ws_unique, Ws_std/Ws_mean, 'r:')
        plt.xlabel("$\lambda$")
        plt.ylabel("$cv$ $registration$ $time$ $[s]$")          
        plt.tight_layout()
        plt.show()        
        plt.savefig('../results/' + batch_name + '/finite_edges_time_kinematic_batch_lambda_cv.pdf')
        plt.savefig('../results/' + batch_name + '/finite_edges_time_kinematic_batch_lambda_cv.png') 
        
        #k_n#########
        plt.figure()
        plt.plot(Xs_unique, Xs_mean, 'b.')
        plt.plot(Xs_unique, Xs_mean, 'b:')
        plt.xlabel("$k_n$")
        plt.ylabel("$mean$ $registration$ $time$ $[s]$")        
        plt.tight_layout()
        plt.show()        
        plt.savefig('../results/' + batch_name + '/finite_edges_time_kinematic_batch_kn_mean.pdf')
        plt.savefig('../results/' + batch_name + '/finite_edges_time_kinematic_batch_kn_mean.png')
        
        plt.figure()
        plt.plot(Xs_unique, Xs_std, 'g.')
        plt.plot(Xs_unique, Xs_std, 'g:')
        plt.xlabel("$k_n$")
        plt.ylabel("$std$ $registration$ $time$ $[s]$")        
        plt.tight_layout()
        plt.show()        
        plt.savefig('../results/' + batch_name + '/finite_edges_time_kinematic_batch_kn_std.pdf')
        plt.savefig('../results/' + batch_name + '/finite_edges_time_kinematic_batch_kn_std.png')
        
        plt.figure()
        plt.plot(Xs_unique, Xs_std/Xs_mean, 'r.')
        plt.plot(Xs_unique, Xs_std/Xs_mean, 'r:')
        plt.xlabel("$k_n$")
        plt.ylabel("$cv$ $registration$ $time$ $[s]$")          
        plt.tight_layout()
        plt.show()        
        plt.savefig('../results/' + batch_name + '/finite_edges_time_kinematic_batch_kn_cv.pdf')
        plt.savefig('../results/' + batch_name + '/finite_edges_time_kinematic_batch_kn_cv.png') 
        
        #k_n#########
        plt.figure()
        plt.plot(Ys_unique, Ys_mean, 'b.')
        #plt.plot(Ys_unique, Ys_mean, 'b:')
        plt.xlabel("$\omega$")
        plt.ylabel("$mean$ $registration$ $time$ $[s]$")        
        plt.tight_layout()
        #plt.xlim(0,0.1)
        plt.show()        
        plt.savefig('../results/' + batch_name + '/finite_edges_time_kinematic_batch_omega_mean.pdf')
        plt.savefig('../results/' + batch_name + '/finite_edges_time_kinematic_batch_omega_mean.png')
        
        plt.figure()
        plt.plot(Ys_unique, Ys_std, 'g.')
        plt.plot(Ys_unique, Ys_std, 'g:')
        plt.xlabel("$\omega$")
        plt.ylabel("$std$ $registration$ $time$ $[s]$")        
        plt.tight_layout()
        plt.show()        
        plt.savefig('../results/' + batch_name + '/finite_edges_time_kinematic_batch_omega_std.pdf')
        plt.savefig('../results/' + batch_name + '/finite_edges_time_kinematic_batch_omega_std.png')
        
        plt.figure()
        plt.plot(Ys_unique, Ys_std/Ys_mean, 'r.')
        plt.plot(Ys_unique, Ys_std/Ys_mean, 'r:')
        plt.xlabel("$\omega$")
        plt.ylabel("$cv$ $registration$ $time$ $[s]$")          
        plt.tight_layout()
        plt.show()        
        plt.savefig('../results/' + batch_name + '/finite_edges_time_kinematic_batch_omega_cv.pdf')
        plt.savefig('../results/' + batch_name + '/finite_edges_time_kinematic_batch_omega_cv.png') 


def crop_patch(data_path, mask_name, sk_pt = None, size=(256,256), return_patch=False):
    """
    Crop a patch of a given image around a given (or selected) skeleton point with
    determined size

    Args:
        data_path (str): Data folder path
        mask_name (str): Image mask name
        sk_pt (array, optional): x,y coordinates of the point of interest. Defaults to None.
        size (tuple, optional): _Size of the patch to be cropped. Defaults to (256,256).
        return_patch (bool, optional): If true returns image patch. Defaults to False.

    Returns:
        patch, patch_name, fc[0]: image patch, patch name and coordinate where image was cropped
    """
      
    img = cv2.imread(data_path+mask_name)

    #Interact with user to click four points to find bounding boxes    
    print('Please click 1 point that represent the center of the box with dim', size)
    
    if sk_pt is None:
        plt.figure()
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        fc = np.array(pylab.ginput(1,200))        
        plt.close()
    else:
        fc = np.array(sk_pt).reshape((1,2))
    #Create bounding boxes from information
    tl = np.array([fc[0][1] - int(size[0]/2), fc[0][0] - int(size[1]/2)]).astype('int')
    br = np.array([fc[0][1] + int(size[0]/2), fc[0][0] + int(size[1]/2)]).astype('int')
    bb = [tl,br]
    
    #Cropped image
    patch = img[int(bb[0][0]):int(bb[1][0]), int(bb[0][1]):int(bb[1][1]), :]
    patch_name = mask_name.split('.')[0] + "_patch_x{}_y{}.png".format(int(fc[0][0]), int(fc[0][1]))
    cv2.imwrite(data_path+ patch_name, patch)
    
    if return_patch:
        return patch, patch_name, fc[0]
    else:
        return patch_name, fc[0]

def get_tn_tt_from_results(im_name, data_folder, sk_pt, approach, hyper):
    """
    Computes the normal and tangential displacements for a given point reading the
    results previously computed

    Args:
        im_name (str): image name
        data_folder (str): data folder name
        sk_pt (array): x,y coordinate of point of interest to read kinematics
        approach (str): approached used, finite_edges or full_edges
        hyper (list): list of hyperparameters used 

    Returns:
        tn_tt_read, skl_pt: kinematic values in px and the skl_pt where was computed
    """
    
    if approach=='finite_edges':
        with open('../results/' + data_folder + im_name + '/' + approach + '/eta{}_m{}_l{}_knnor{}_omega{}/crack_kinematic.json'.format(hyper[0],hyper[1],hyper[2],hyper[3],hyper[4]), 'r') as fp:
            dict_error_kinematic_batch = json.load(fp)
        key_kin = 'kinematics_n_t_loc'
    else:
        with open('../results/' + data_folder + im_name + '/' + approach + '/knnor{}_omega{}/crack_kinematic.json'.format(hyper[0],hyper[1]), 'r') as fp:
            dict_error_kinematic_batch = json.load(fp)
        key_kin = 'kinematics_n_t'
    
    
    skeleton = []
    for sk in dict_error_kinematic_batch['0'][key_kin]:
        skeleton.append(sk[0])
    skeleton = np.array(skeleton).reshape((-1,2))
    distances = np.linalg.norm(skeleton-sk_pt, axis = 1)
    min_distance = np.min(distances)
    ind_min_dist = np.where(distances==min_distance)[0]
    sk_pt_read = skeleton[ind_min_dist[0]]
    tn_tt_read = dict_error_kinematic_batch['0'][key_kin][ind_min_dist[0]][2]
    skl_pt = dict_error_kinematic_batch['0']['dir_nor_skl'][ind_min_dist[0]]
    #t_dofs_loc = dict_error_kinematic_batch['0']['kinematics_n_t_loc'][ind_min_dist[0]]

    print("the kinematic using algorithm for the skeleton point {} is {}".format(sk_pt_read, tn_tt_read))

    return tn_tt_read, skl_pt#, t_dofs_loc   