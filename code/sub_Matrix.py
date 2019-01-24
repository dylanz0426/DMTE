#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 20:43:25 2018

@author: dylanz
"""

import numpy as np

def sub_Mat(P, node):
    
    sub_P = np.zeros((len(node),len(node)))
    for i in range(len(node)):
        for j in range(len(node)):
            sub_P[i,j] = P[node[i],node[j]]
    
    return sub_P