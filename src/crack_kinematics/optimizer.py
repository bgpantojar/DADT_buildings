#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 12:07:42 2021

Pareto optimizer (run_pareto) function is based on 
Pareto-like sequential sampling heuristic for global optimisation
Mahmoud S. Shaqfa et al.
https://github.com/eesd-epfl/pareto-optimizer

The script was modified for the crack kinematics problem according
to Pantoja-Rosero et al. "Determining crack kinimatics from image crack patterns"

@author: pantoja
"""

import time
import numpy as np
import pyDOE
import copy
import matplotlib.pyplot as plt
from crack_kinematics.tools_least_squares import run_crack_adjustment
from crack_kinematics.tools_crack_kinematic import H_from_transformation

class LHSSampler:
    # This class is specialized in generating random matricies
    # LHS sampling
    def __init__(self, dim, pop_size):
        self.dim = dim
        self.pop_size = pop_size
    
    def _gen_random(self):
        np.random.seed() # Change the time seed
        if self.pop_size == 1:
            criterion = None
        else:
            criterion = 'm'
        
        random_pop = pyDOE.lhs(n = self.dim, criterion = criterion, samples = self.pop_size)
        return random_pop

class MonteCarloSampler:
    # This class is specialized in generating random matricies
    # MonteCarlo sampling
    def __init__(self, dim, pop_size):
        self.dim = dim
        self.pop_size = pop_size
    
    def _gen_random(self):
        np.random.seed() # Change the time seed
        random_pop = np.random.rand(self.pop_size, self.dim)
        #random_pop = np.random.randint(self.pop_size, self.dim)
        return random_pop


def run_pareto(lb, ub, pop_size, dim, step, sampler, acceptance_rate, iterations, objective_function, sampling_method = 'LHS', history = False, plot = False):
    """
    To run the analysis of the optimizer
    """
    
    basic_LB = np.ones((1, dim))
    basic_UB = copy.copy(basic_LB)
    steps = copy.copy(basic_LB) * step
    LB =  lb * basic_LB
    UB =   ub * basic_UB
    funval = np.zeros([pop_size, 1])
    H_val = np.zeros([pop_size, 3])
    population = np.zeros([pop_size, dim])
    funval_his = np.zeros([iterations + 1, 1])
    H_val_his = np.zeros([iterations + 1, 3])
    
    if history:
        solutions_vectors_his = np.zeros([(iterations+1) * pop_size, dim])
    
    reduced_LB = 0.
    reduced_UB = 0.
    
    # Initalize a sampling method
    if sampling_method == 'LHS':
        sampler = LHSSampler(dim, pop_size)
    elif sampling_method == 'MonteCarlo':
        sampler = MonteCarloSampler(dim, pop_size)
    else:
        raise Exception('{} samplinmg method is not defined. Please choose "LHS" or "MonteCarlo"'.format(sampling_method))
    
    limits_his = np.zeros([iterations + 1, dim * 2])
    
    ###Algorithm    
    t1 = time.time()
    # The initial population sampling
    LB_mat = np.matlib.repmat(LB, pop_size, 1)
    UB_mat = np.matlib.repmat(UB, pop_size, 1)
    steps_mat = np.matlib.repmat(steps, pop_size, 1)
    random_pop = sampler._gen_random()
    population = (np.round((LB_mat + random_pop * (UB_mat - LB_mat)) / steps_mat) * steps_mat).astype('int')
    print(population)
    del LB_mat, UB_mat, steps_mat
    
    # Evaluate the initial solutions
    for i in range(pop_size):
        H_val[i,:] , funval[i, 0] = objective_function.compute(population[i, :])
    
    # Find the best initial solution and store it
    best_index = np.argmin(funval)
    funval_his[0, 0] = funval[best_index, 0]
    H_val_his[0, :] = H_val[best_index, :]
    best_sol = copy.copy(population[best_index, :])
    best_H = copy.copy(H_val_his[0, :])
    best_loss = copy.copy(funval_his[0, 0])
    new_solution = True
    
    # Hold population memory
    if history:
        solutions_vectors_his[0:pop_size, :] = population
    
    for i in range(1, iterations+1):
        random_pop = sampler._gen_random()
        for j in range(pop_size):
            for k in range(dim):
                
                # Update the ranges
                if new_solution:
                    # The deviation is positive dynamic real number
                    deviation = abs(0.5 * (1. - acceptance_rate) * (UB[0, k] - LB[0, k])) * (1 - (i/(iterations )))
                
                reduced_LB = best_sol[k] - deviation
                reduced_LB = np.amax([reduced_LB , LB[0, k]])
                
                reduced_UB = reduced_LB + deviation * 2.
                reduced_UB = np.amin([reduced_UB, UB[0, k]])
                
                limits_his[i, k * 2] = reduced_LB
                limits_his[i, k * 2 + 1] = reduced_UB
                
                # Choose new solution
                if np.random.rand() <= acceptance_rate:
                    # choose a solution from the prominent domain
                    population[j, k] = int(reduced_LB + random_pop[j, k] * (reduced_UB - reduced_LB))
                else:
                    # choose a solution from the overall domain
                    population[j, k] = int(LB[0, k] + random_pop[j, k] * (UB[0, k] - LB[0, k]))
                
                # Round for the step size
                population[j, :] = (np.round(population[j, :] / steps) * steps).astype('int')
            # Evaluate the solution
            
            H_val[j] , funval[j] = objective_function.compute(population[j, :])
            
        # Hold population memory
        if history:
            solutions_vectors_his[i * pop_size: (i + 1) * pop_size, :] = population
        
        # Find the current best solution
        if np.amin(funval) <= funval_his[i - 1]:
            best_index = np.argmin(funval)
            funval_his[i] = funval[best_index, 0]
            H_val_his[i] = H_val[best_index, :]
            best_sol = copy.copy(population[best_index, :])
            best_H = copy.copy(H_val_his[i])
            best_loss = copy.copy(funval_his[i])
            new_solution = True
        else:
            funval_his[i] = funval_his[i - 1]
            H_val_his[i] = H_val_his[i - 1]
            new_solution = False
        
    if plot:
        plt.close('all')
        # 2D Drawing
        fig = plt.figure()
        fig.set_size_inches(4.5, dim * 2)
        
        plt.subplot(dim +1, 1, 1)
        plt.semilogx(np.arange(funval_his.size), funval_his)
        plt.xlabel("Iterations")
        plt.ylabel("The objective function evaluation")
        plt.title("The convergance curve")
        plt.xlim([1, funval_his.size])
        plt.grid()
        plt.subplots_adjust(hspace = 1)
        
        for i in range(1, dim + 1):
            plt.subplot(dim +1, 1, i+1)
            plt.semilogx(np.arange(funval_his.size),\
                         limits_his[:, 2*(i-1)], '-b',  linewidth = 0.5)
            plt.semilogx(np.arange(funval_his.size),\
                         limits_his[:, 2*i-1], '-g', linewidth = 0.5)
            plt.semilogx(np.arange(funval_his.size)\
                     , objective_function.exact * np.ones([funval_his.size, 1])\
                     ,'r--', linewidth = 0.5)
            if i == 1:
                plt.xlabel("Iterations")
                plt.ylabel("The limits value")
                plt.title("The bounding limits evaluation")
            plt.xlim([1, funval_his.size])
            plt.grid()
            plt.fill_between(np.arange(funval_his.size)\
                             , limits_his[:, 2*(i-1)], limits_his[:, 2*i-1]\
                             , facecolor='lightgray')
        plt.savefig('results.pdf', bbox_inches='tight')
        plt.savefig('results.pdf')
        plt.show()
    
    if history: return best_sol, best_H, best_loss, solutions_vectors_his
    else: return best_sol, best_H, best_loss, []
    
class objective_function_kinematics:
    """
    Objective function defined to be used in pareto algorithm
    """
    
    def __init__(self, set_crack_edge_0, set_crack_edge_1_mk_ord, omega, k_neighboors, H_type="euclidean", normals=False, local_registration = False):
        
        self.set_crack_edge_0 = set_crack_edge_0
        self.set_crack_edge_1_mk_ord = set_crack_edge_1_mk_ord
        self.omega = omega
        self.k_neighboors = k_neighboors
        self.H_type = H_type
        self.normals = normals
        self.local_registration = local_registration
        
    
    def compute(self, population_j):    
        init_ind = population_j[0]
        
        #H is calculated globally and locally. Global the origen of the transformations is the origen of the image
        #locally, the origen is the middle point of the edge_0
        set_crack_edge_1_kg = self.set_crack_edge_1_mk_ord[init_ind:init_ind+self.k_neighboors]        
    
        if not self.local_registration:    
            #GLOBAL TRANSFORMATION (COORDINATE SYSTEM ORIGEN AT IMAGE ORIGEN)
            H_op, loss = run_crack_adjustment(self.set_crack_edge_0, set_crack_edge_1_kg, self.omega, H_type=self.H_type, normals=self.normals)
            return H_op, np.array(loss).sum()
        else:
            #Local edges coordinates
            mean_edge_0 = np.mean(self.set_crack_edge_0[:,:2], axis=0)
            set_crack_edge_0_loc = np.copy(self.set_crack_edge_0) 
            set_crack_edge_0_loc[:,:2] -=  mean_edge_0
            set_crack_edge_1_loc = np.copy(set_crack_edge_1_kg) 
            set_crack_edge_1_loc[:,:2] -=  mean_edge_0
            
            H_op_loc, loss_loc = run_crack_adjustment(set_crack_edge_0_loc, set_crack_edge_1_loc, self.omega, H_type=self.H_type, normals=self.normals)
        
            return H_op_loc, np.array(loss_loc).sum()        
    
    
def kinematic_adjustment_pareto_based(k_neighboors, set_crack_edge_0, set_crack_edge_1_mk_ord, l, omega, H_type="euclidean", normals=False, ignore_global = True):
    """
    Computes the crack kinematics iteratively to register k points from edge0 (of lenght k) to k points of edge1 (of lenght mk).
    In the  pareto aproach the position lambda l where k points selected from the edge1 start is found thorugh heuristics.

    Args:
        k_neighboors (int): k neighboors used as hyperparameter in the method
        set_crack_edge_0 (array): point coordinates of finite edge edge0
        set_crack_edge_1_mk_ord (array): point coordinates of finite edge edge1
        l (int): value of hyperparameter lambda l
        omega (_type_): _description_value of hyperparameter omega
        H_type (str, optional): type of transformation represented by H. Defaults to "euclidean".
        normals (bool, optional): True to consider normal features of edges in the residual function. Defaults to False.
        ignore_global (bool, optional): True to compute just kinmeatics with local coordinates. Defaults to True.

    Returns:
        H_op, H_op_loc (array): Array with the optimal parameters values of the matrix H
    """

    #Pareto params
    lb = 0 # 1st element
    ub = len(set_crack_edge_1_mk_ord) - len(set_crack_edge_0) + 1 # (mk - k + 1) different possibilities
    pop_size = max(2, round(.05*ub)) #at least two samples
    print(pop_size,'n')
    step = 10e-10
    sampler = 'LHS'
    dim = 1
    acceptance_rate = .90
    iterations = 5
    
    if not ignore_global:

        objective_function = objective_function_kinematics(set_crack_edge_0, set_crack_edge_1_mk_ord, omega, k_neighboors, H_type=H_type, normals=normals)    
        best_sol, best_H, best_loss, _ = run_pareto(lb, ub, pop_size, dim, step, sampler, acceptance_rate, iterations, objective_function, sampling_method = 'LHS', history = False, plot = False)
        
        H_op = np.copy(best_H)
    else:
        H_op = np.array([])
    
    objective_function = objective_function_kinematics(set_crack_edge_0, set_crack_edge_1_mk_ord, omega, k_neighboors, H_type=H_type, normals=normals, local_registration=True)
    best_sol, best_H, best_loss, _ = run_pareto(lb, ub, pop_size, dim, step, sampler, acceptance_rate, iterations, objective_function, sampling_method = 'LHS', history = False, plot = False)
    H_op_loc = np.copy(best_H)
    
    return H_op, H_op_loc
    
    
def kinematic_adjustment_lambda_based(mask, mask_name, x,y, k_neighboors, set_crack_edge_0, set_crack_edge_1_mk_ord, l, omega, H_type="euclidean", normals=False, ignore_global=True):
    """
    Computes the crack kinematics iteratively to register k points from edge0 (of lenght k) to k points of edge1 (of lenght mk).
    In the lambda approach the k points selected from the edge1 start from the position l of the full edge1. The optimal H
    matrix correspond to the one with the minimal resitual function value

    Args:
        mask (array): mask image
        mask_name (str): name of the mask with the crack pattern
        x (int): x coordinate of skeleton taken in consideration
        y (int): y coordinate of skeleton taken in consideration
        k_neighboors (int): k neighboors used as hyperparameter in the method
        set_crack_edge_0 (array): point coordinates of finite edge edge0
        set_crack_edge_1_mk_ord (array): point coordinates of finite edge edge1
        l (int): value of hyperparameter lambda l
        omega (_type_): _description_value of hyperparameter omega
        H_type (str, optional): type of transformation represented by H. Defaults to "euclidean".
        normals (bool, optional): True to consider normal features of edges in the residual function. Defaults to False.
        ignore_global (bool, optional): True to compute just kinmeatics with local coordinates. Defaults to True.

    Returns:
        H_op, H_op_loc (array): Array with the optimal parameters values of the matrix H
    """
        
    if not ignore_global:
        #GLOBAL
        #Iterating through edge1 and selecting as H_op the one with minimum loss
        init_ind = 0
        best_loss = np.infty
        best_H = None
        best_ind = 0
        while k_neighboors+init_ind <= len(set_crack_edge_1_mk_ord):
            set_crack_edge_1_kg = set_crack_edge_1_mk_ord[init_ind:init_ind+k_neighboors]
            
            ###RUNING LEAST SQUARES
            #H_type: euclidean, similarity, affine, projective
            H_type = "euclidean"
            #H is calculated globally and locally. Global the origen of the transformations is the origen of the image
            #locally, the origen is the middle point of the edge_0
            
            #GLOBAL TRANSFORMATION (COORDINATE SYSTEM ORIGEN AT IMAGE ORIGEN)
            H_op, loss = run_crack_adjustment(set_crack_edge_0, set_crack_edge_1_kg, omega, H_type=H_type, normals=normals)
            if np.mean(loss)<best_loss:
                best_H = np.copy(H_op)
                best_loss = np.copy(np.mean(loss))
                best_ind = init_ind
                
            init_ind+=l #moving l points for each new group
        
        #The last kgroup (m group in the paper) has as end point the endpoint of the mk set
        if k_neighboors+init_ind > len(set_crack_edge_1_mk_ord):
            set_crack_edge_1_kg = set_crack_edge_1_mk_ord[-k_neighboors:]
            
            ###RUNING LEAST SQUARES
            #H_type: euclidean, similarity, affine, projective
            H_type = "euclidean"
            #H is calculated globally and locally. Global the origen of the transformations is the origen of the image
            #locally, the origen is the middle point of the edge_0
            #GLOBAL TRANSFORMATION (COORDINATE SYSTEM ORIGEN AT IMAGE ORIGEN)
            H_op, loss = run_crack_adjustment(set_crack_edge_0, set_crack_edge_1_kg, omega, H_type=H_type, normals=normals)
            
            if np.mean(loss)<best_loss:
                best_H = np.copy(H_op)
                best_loss = np.copy(np.mean(loss))
                best_ind = init_ind
                
            init_ind+=l #moving l points for each new group
            
        
        H_op = np.copy(best_H)
    else:
        H_op = np.array([])
    
    
    #LOCAL
    #LOCAL TRANSFORMATION (COORDINATE SYSTEM ORIGEN AT THE MEAN OF THE EDGE 0))
    #Iterating through edge1 and selecting as H_op the one with minimum loss
    init_ind = 0
    best_loss = np.infty
    best_H = None
    best_ind = 0
    while k_neighboors+init_ind <= len(set_crack_edge_1_mk_ord):
        #print("local", init_ind,k_neighboors)
        set_crack_edge_1_kg = set_crack_edge_1_mk_ord[init_ind:init_ind+k_neighboors]
        #Local edges coordinates
        mean_edge_0 = np.mean(set_crack_edge_0[:,:2], axis=0)
        set_crack_edge_0_loc = np.copy(set_crack_edge_0) 
        set_crack_edge_0_loc[:,:2] -=  mean_edge_0
        #set_crack_edge_1_loc = set_crack_edge_1_kg - mean_edge_0
        set_crack_edge_1_loc = np.copy(set_crack_edge_1_kg) 
        set_crack_edge_1_loc[:,:2] -=  mean_edge_0
        
        ###RUNING LEAST SQUARES
        H_type = "euclidean"
                    
        #LOCAL TRANSFORMATION
        H_op_loc, loss_loc = run_crack_adjustment(set_crack_edge_0_loc, set_crack_edge_1_loc, omega, H_type=H_type, normals=normals)
        
        if np.mean(loss_loc)<best_loss:
            best_H = np.copy(H_op_loc)
            best_loss = np.copy(np.mean(loss_loc))
            best_ind = init_ind
            
        init_ind+=l #moving l points for each new group
        
        
    #The last kgroup (m group in the paper) has as end point the endpoint of the mk set
    if k_neighboors+init_ind > len(set_crack_edge_1_mk_ord):
        set_crack_edge_1_kg = set_crack_edge_1_mk_ord[-k_neighboors:]
        #Local edges coordinates
        mean_edge_0 = np.mean(set_crack_edge_0[:,:2], axis=0)
        set_crack_edge_0_loc = np.copy(set_crack_edge_0) 
        set_crack_edge_0_loc[:,:2] -=  mean_edge_0
        set_crack_edge_1_loc = np.copy(set_crack_edge_1_kg) 
        set_crack_edge_1_loc[:,:2] -=  mean_edge_0            
        
        ###RUNING LEAST SQUARES
        #H_type: euclidean, similarity, affine, projective
        H_type = "euclidean"
        #H is calculated globally and locally. Global the origen of the transformations is the origen of the image
        #locally, the origen is the middle point of the edge_0
        #GLOBAL TRANSFORMATION (COORDINATE SYSTEM ORIGEN AT IMAGE ORIGEN)
        #LOCAL TRANSFORMATION
        H_op_loc, loss_loc = run_crack_adjustment(set_crack_edge_0_loc, set_crack_edge_1_loc, omega, H_type=H_type, normals=normals)
        
        if np.mean(loss_loc)<best_loss:
            best_H = np.copy(H_op_loc)
            best_loss = np.copy(np.mean(loss_loc))
            best_ind = init_ind
            
        init_ind+=l #moving l points for each new group    
    
    H_op_loc = np.copy(best_H)    
    
    H_loc = H_from_transformation(H_op_loc, H_type)  
    
    #Local edges coordinates
    mean_edge_0 = np.mean(set_crack_edge_0[:,:2], axis=0)
    set_crack_edge_0_loc = np.copy(set_crack_edge_0) 
    set_crack_edge_0_loc[:,:2] -=  mean_edge_0
    
    #Transforming one edge to overlap over the other globally
    set_crack_edge_0_op_loc = np.concatenate((np.copy(set_crack_edge_0_loc[:,:2]), np.ones((len(set_crack_edge_0_loc),1))), axis=1).T
   
    set_crack_edge_0_op_loc = H_loc @ set_crack_edge_0_op_loc 
    #print(set_crack_edge_0_op_loc)
    set_crack_edge_0_op_loc  /= set_crack_edge_0_op_loc[2]
    set_crack_edge_0_op_loc  = set_crack_edge_0_op_loc[:2].T
    
    return H_op, H_op_loc
    