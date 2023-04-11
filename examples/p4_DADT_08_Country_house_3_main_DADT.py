#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 04 20:06:06 2020

@author: pantoja
"""
import os
import sys
sys.path.append("../src")
import numpy as np
import Part
import FreeCAD
from FreeCAD import Base
import importOBJ
#from opening_builder import *
#from amuya.src.opening_builder import *
from opening_builder import *
import time

##################################USER INTERACTION#############################
t0 = time.time()
###############USER INPUT
data_folder =  'p4_DT_LOD_08_Country_house_3'
data_path = "../results/" + data_folder 
polyfit_path = "../data/" + data_folder + "/polyfit" 
#############USER CALLING FUNCTIONS
print("Building LOD3")
#Builder
op_builder(data_folder, data_path, polyfit_path)
print("Time merging LOD2 with openings {}s".format(time.time()-t0))