#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import pickle
from recnn import grnn_predict_gated
from preprocessing import apply_tf_transform
from preprocessing import create_tf_transform

def compute_roc_curve(y, y_pred,density=1000):
    """return the roc curve"""
    back = np.argwhere(y == 0)
    back = back.reshape((len(back),))
    sign = np.argwhere(y == 1)
    sign = sign.reshape((len(sign),))
    #prediction
    y_pred_sign = y_pred[sign]
    y_pred_back = y_pred[back]
    t = np.linspace(0.,1.,density)
    tpr = np.zeros(density,dtype = float)
    fpr = np.zeros(density,dtype = float)
    for i in range(density):
        tpr[i] = np.sum(y_pred_sign <= t[i])
        fpr[i] = np.sum(y_pred_back <= t[i])
    tpr = 1-tpr/len(y_pred_sign)
    fpr = 1-fpr/len(y_pred_back)
    return(fpr,tpr,t)

# In[]:
def predict(X, filename, func=grnn_predict_gated,regression = True):
    """make prediction function"""
    fd = open(filename, "rb")
    params = pickle.load(fd)
    fd.close()
    y_pred = func(params, X,regression = regression)
    return(y_pred)


def transform_for_prediction(Xtrain,Xtest):
    tf = create_tf_transform(Xtrain)
    return(apply_tf_transform(Xtest,tf))

def build_roc(X, y, filename, func=None):
    """evaluates a model and build its roc curve"""
    print("Loading " + filename),
    y_pred = predict(X, filename, func=func)
    fpr, tpr, _ = compute_roc_curve(y, y_pred,density=10000)
    roc = np.trapz(-tpr,fpr)
    print("ROC AUC = %.4f" % roc)
    return(roc, fpr, tpr)
