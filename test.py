#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 14:16:41 2018

@author: gtouquet
"""

from recnn.recnn import grnn_predict_gated
from recnn.evaluate import predict, build_roc
from recnn.preprocessing import apply_tf_transform, create_tf_transform
import numpy as np
from shutil import copyfile
from ROOT import TFile
import os

    
    
def test(filepath, modelpath, trainfilepath, rootfilepath, branchname, isSignal):
    X,y = np.load(trainfilepath)
    tf = create_tf_transform(X)
    X, y = np.load(filepath)
    X_tf = apply_tf_transform(X,tf)
    y_pred = predict(X_tf,modelpath, grnn_predict_gated, regression=False)
    testfile = TFile(rootfilepath,'update')
    testtree = testfile.Get('testtree')
    finaltree = testtree.CloneTree(0)
    finaltree.SetName('finaltree')
    branchval = np.zeros(1)
    finaltree.Branch(branchname, branchval, branchname+'/D')
    branchval2 = np.zeros(1)
    branchval2[0] = 1. if isSignal else 0.
    finaltree.Branch('isSignal', branchval2, 'isSignal/D')
    i = 0
    for event in testtree:
        #here test if eta still aligned
        branchval[0] = y_pred[i]
        i+=1
        finaltree.Fill()
    finaltree.Write()

if __name__ == '__main__':
    patterns = ['R=0.3_anti-kt']#,'R=1e-05_cambridge','R=1e-05_kt','R=1e-05_random','R=1e-05_seqpt','R=1e-05_seqpt_reversed','R=0.3_anti-kt','R=0.3_cambridge','R=0.3_kt','R=0.3_random','R=0.3_seqpt','R=0.3_seqpt_reversed']
    for pattern in patterns:
        tester(pattern, steps=False)
