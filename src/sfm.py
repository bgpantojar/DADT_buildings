#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 01:46:53 2020

This script contains the codes to generate LOD3 models.
The codes are based on "Generation LOD3 models from structure-from-motion and semantic segmentation" 
by Pantoja-Rosero et., al.
https://doi.org/10.1016/j.autcon.2022.104430

This script specifically support LOD3 codes development.
These are based on codes published in:
Solem, J.E., 2012. Programming Computer Vision with Python: Tools and algorithms for analyzing images. " O'Reilly Media, Inc.".

Slightly changes are introduced to addapt to general pipeline


@author: pantoja
"""

from pylab import *
from numpy import *
from scipy import linalg
import numpy as np
import json
import matplotlib.pyplot as plt

#function that computes the least squares triangulation of a point pair
def triangulate_point(x1,x2,P1,P2):
    """ Point pair triangulation from 
        least squares solution. """
        
    M = zeros((6,6))
    M[:3,:4] = P1
    M[3:,:4] = P2
    M[:3,4] = -x1
    M[3:,5] = -x2

    U,S,V = linalg.svd(M)
    X = V[-1,:4]

    return X / X[3]

#To triangulate many points, we can add the following convenience function
#This function takes two arrays of points and returns an array of 
#3D coordinates.
def triangulate(x1,x2,P1,P2):
    """    Two-view triangulation of points in 
        x1,x2 (3*n homog. coordinates). """
        
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")

    X = [ triangulate_point(x1[:,i],x2[:,i],P1,P2) for i in range(n)]
    return array(X).T

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

def intersec_ray_plane(p0,p1,plane,epsilon=1e-6):
    u = sub_v3v3(p1, p0)
    dot = dot_v3v3(plane, u)

    if abs(dot) > epsilon:
        # Calculate a point on the plane
        p_co = mul_v3_fl(plane, -plane[3] / len_squared_v3(plane))

        w = sub_v3v3(p0, p_co)
        fac = -dot_v3v3(plane, w) / dot
        u = mul_v3_fl(u, fac)
        return np.array(add_v3v3(p0, u))
    #if plane is parallel to ray return None
    return None

def add_v3v3(v0, v1):
    return (
        v0[0] + v1[0],
        v0[1] + v1[1],
        v0[2] + v1[2],
    )
def sub_v3v3(v0, v1):
    return (
        v0[0] - v1[0],
        v0[1] - v1[1],
        v0[2] - v1[2],
    )
def dot_v3v3(v0, v1):
    return (
        (v0[0] * v1[0]) +
        (v0[1] * v1[1]) +
        (v0[2] * v1[2])
    )
def len_squared_v3(v0):
    return dot_v3v3(v0, v0)
def mul_v3_fl(v0, f):
    return (
        v0[0] * f,
        v0[1] * f,
        v0[2] * f,
    )

def plot_3D_pts(X, c='k.', fig=None, colors=None):
    '''
    Given an array that represents a point cloud, plot it
    Parameters
    ----------
    X : npy.array
        Array with the point cloud information.
    c : str, optional
        Pyplot color string. The default is 'k.'.
    fig : pyplot.fig, optional
        Object of the figure clase from pyplot. If given the 
        plot is performed over this figure. The default is None.
    Returns
    -------
    fig : pyplot.fig, optional
        Object of the figure clase from pyplot containing the ploted point cloud.
    '''

    if fig is None:
        fig = plt.figure()

    if colors is None:
        ax = fig.gca(projection='3d')
        ax.plot(X[:, 0], X[:, 1], X[:, 2], c)
        plt.axis('off')
    else:
        ax = fig.gca(projection='3d')
        i = 0
        for p1 in X:
            ax.plot(p1[0], p1[1], p1[2], color=colors[i], marker='.')
            i += 1

    plt.show()

    return fig

def find_X_x_correspondences(view_name, structure, poses, plot_X=False):
    '''
    Given the view name, its structure dictionary and poses dictionary (extracted
    from sfm.json file) find the 3D-2D correspondences between point cloud and
    keypoints.
    Parameters
    ----------
    view_name : str
        View name from camera poses.
    structure : dict
        Dictionary with the structure information after SfM.
    poses : dict
        Dictionary with the poses information of the cameras used during SfM.
    plot_X : bool, optional
        if true, plot the 3D point cloud of the 3D-2D correspondences.
        The default is False.    
    Returns
    -------
    X_x_view : list
        List with 3D-2D correspondences information. 3D id, 3D coord, 2d id,
        2d coord
    '''
    # Goes through the structure and select the 3D points that are observed
    # from the view

    #Initial structure len
    print("The initial lenght of structure is ", len(structure))

    X_x_view = []
    type_X_x_view = []

    if plot_X:
        X = []

    for X_st_id in structure:
        for obs in structure[X_st_id]['obs']:
            if obs["poseId"] == poses[view_name]["poseId"]:
                X_x_view.append(
                    [structure[X_st_id]["X_ID"], structure[X_st_id]["X"], obs["x_id"], obs["x"], ])
                type_X_x_view.append(structure[X_st_id]["descType"])
                if plot_X:
                    X.append(list(map(float, structure[X_st_id]["X"])))
    
    if plot_X:
        # print(X[0])
        X = np.array(X)
        plot_3D_pts(X)

    print("lenght correspondences X_x is: ", len(X_x_view))
    #Final lenght of structure if it is updated will change
    print("The finallenght of structure is ", len(structure))

    if plot_X:
        # print(X[0])
        X = np.array(X)
        plot_3D_pts(X)

    print("lenght correspondences X_x is: ", len(X_x_view))
    #Final lenght of structure if it is updated will change
    print("The finallenght of structure is ", len(structure))

    return X_x_view, type_X_x_view