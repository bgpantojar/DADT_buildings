#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 04 20:06:06 2020

@author: pantoja
"""
import sys
sys.path.append("../src")
import Part
import FreeCAD
from FreeCAD import Base
import importOBJ
from opening_builder import *

##################################USER INTERACTION#############################
###############USER INPUT
data_folder =  'p4_DADT_00_La_capite'
data_path = "../results/" + data_folder 
polyfit_path = "../data/" + data_folder + "/polyfit" 
#############USER CALLING FUNCTIONS
print("Building LOD3")
#Builder
op_builder(data_folder, data_path, polyfit_path)
