#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 13:54:38 2021

@author: pantoja
"""

import numpy as np
from scipy.optimize import least_squares
import scipy
from crack_kinematics.tools_crack_kinematic import H_from_transformation

def fun(params, crack_edge_0, crack_edge_1, omega, H_type="euclidean", normals=False):
    """
    Residual function to compute the mean squares error

    Args:
        params (array): Array with the parameters to optimize of H
        crack_edge_0 (array): coordinates of the points of edge0
        crack_edge_1 (array): coordinates of the points of edge1
        omega (float): weight for normal features parcel in residual function
        H_type (str, optional): _transformation type to define H matrix. euclidean, similarity, affine, projective. Defaults to "euclidean".
        normals (bool, optional): True to consider normal features of edges in the residual function. Defaults to False.

    Returns:
        distances: residual value after applying transformation to edge0 with H
    """
   
    #According transformation find H with params
    H = H_from_transformation(params, H_type)

    XXB = np.concatenate((crack_edge_0[:,:2], np.ones((len(crack_edge_0[:,:2]),1))), axis=1).T
    XXB = np.dot(H, XXB)
    XXB /= XXB[2]
    
    XB = np.copy(XXB[:2].T)
    XA = np.copy(crack_edge_1[:,:2])
        
    points_distances_full = np.abs(scipy.spatial.distance.cdist(XA,XB))
    distances = np.min(points_distances_full,axis=1)
    
    #Add normals information as extra feature when registering the point sets (not inlcuded in the paper)
    if normals:
        #Method 2: computing normals once when contour is detected. Here use those normals in a normalized way
        #ids in set A that are the clossest to set B after transformed
        distances_id = np.argmin(points_distances_full,axis=1)
        #Taking the normals of the points related with the minimum distances
        nor_A = crack_edge_1[:,2]
        nor_A = nor_A[distances_id]
        nor_B = crack_edge_0[:,2]
        nor_B = nor_B[distances_id]
        #Make the normal feature descriptor as the sum of the two gradients of sequential points (before and after the point)
        nor_A = np.concatenate(((nor_A[:-1] - nor_A[1:]), np.zeros(1))) + np.concatenate((np.zeros(1),(nor_A[1:] - nor_A[:-1])))
        nor_B = np.concatenate(((nor_B[:-1] - nor_B[1:]), np.zeros(1))) + np.concatenate((np.zeros(1),(nor_B[1:] - nor_B[:-1])))
        distances += np.abs(omega*(nor_A - nor_B))
    
    return distances

def run_crack_adjustment(crack_edge_0, crack_edge_1, omega, H_type = "euclidean", normals=False):
    """
    This function finds the optimal transformation matrix H to register the edge0 over the edge1

    Args:
        crack_edge_0 (array): coordinates of the points of edge0
        crack_edge_1 (array): coordinates of the points of edge1
        omega (float): weight for normal features parcel in residual function
        H_type (str, optional): _transformation type to define H matrix. euclidean, similarity, affine, projective. Defaults to "euclidean".
        normals (bool, optional): True to consider normal features of edges in the residual function. Defaults to False.

    Returns:
        res.x, res.fun: optimal values for parameters of H and residual function
    """
    if H_type=="euclidean":
        theta = 0
        tx = 0
        ty = 0
        H = np.array([theta,tx,ty])
    elif H_type == "similarity":
        theta = 0
        tx = 0
        ty = 0
        s = 1
        H = np.array([theta,tx,ty,s])
    elif H_type == "affine":
        theta = 0
        tx = 0
        ty = 0
        m = 0
        sx = 1
        sy = 1
        H = np.array([theta,tx,ty,m, sx, sy])        
    elif H_type == "projective":
        H = np.eye(3).ravel()[:-1]
        
    #Problem numbers
    n = H.ravel().shape[0]
    m = crack_edge_0.shape[0]
    
    x0 = H.ravel()
    res = least_squares(fun, x0, verbose=0, ftol=1e-4, method='lm',\
                        args=(crack_edge_0, crack_edge_1, omega, H_type, normals))
        
    return res.x, res.fun