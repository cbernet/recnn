#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import pickle
from recnn.recnn import grnn_predict_gated
from recnn.preprocessing import apply_tf_transform
from recnn.preprocessing import create_tf_transform

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



# In[]:

Xtest,ytest=np.load('/data/conda/recnn/data/npyfilesregression/Background_JEC_test_ID_preprocessed_R=0.3_anti-kt.npy')
Xtrain,_=np.load('/data/conda/recnn/data/npyfilesregression/Background_JEC_train_ID_preprocessed_R=0.3_anti-kt.npy')
X=transform_for_prediction(Xtrain,Xtest)
#xptgen=[x['genpt'] for x in X]
xptgen=ytest
xptraw=[x['pt'] for x in X]
y = predict(X, '/data/conda/recnn/data/modelsregression/Background_JEC_train_ID_preprocessed_R=0.3_anti-kt_MorePreciseStat.pickle', func=grnn_predict_gated,regression = True)
import matplotlib.pyplot as plt
#indices = [i for i in range(len(X)) if -0.2<X[i]["eta"]<0.2]
# In[]:
fig=plt.figure()
fig.set_size_inches((8,8))
funct = np.array(xptraw)/np.array(xptgen)
histraw,binsraw = np.histogram(funct,bins=1000,normed=True)
width = (binsraw[1] - binsraw[0])
center = (binsraw[:-1] + binsraw[1:]) / 2
plt.grid()
plt.xlim([0,3])
mean =np.mean(funct)
sigma=np.std(funct)
plt.bar(center, histraw, align='center', width=width,label="ptraw/ptgen, mean ={:.4f}, std = {:.4f}".format(mean,sigma),alpha=0.5,color='green')

mean =np.mean(funct)
sigma=np.std(funct)
#def f(x):
#    return(max(histraw)*np.exp(-((x-mean)**2)/(2*(sigma**2))))
#x0=np.linspace(0,3,10000)
#plt.plot(x0,f(x0))

###

funct = np.array(y)/np.array(xptgen)
histpred,binspred = np.histogram(funct,bins=100,normed=True)
width = (binspred[1] - binspred[0])
center = (binspred[:-1] + binspred[1:]) / 2

plt.xlim([0,3])
mean =np.mean(funct)
sigma=np.std(funct)

plt.bar(center, histpred, align='center', width=width,label="ptpred/ptgen, mean ={:.4f}, std = {:.4f}".format(mean,sigma),alpha=0.5,color='red')
plt.grid(True)
plt.ylabel('density')
plt.xlabel('pt/ptgen')

#def f(x):
#    return(max(histpred)*np.exp(-((x-mean)**2)/(2*(sigma**2))))
#x0=np.linspace(0,3,10000)
#plt.plot(x0,f(x0))

plt.legend()


fig.savefig("figure_histogram_end_EtaReducedStat.png",dpi=600)



# In[]:
fig=plt.figure()
fig.set_size_inches((8,8))
funct = np.array(xptraw)
histraw,binsraw = np.histogram(funct,bins=1000,normed=True)
width = (binsraw[1] - binsraw[0])
center = (binsraw[:-1] + binsraw[1:]) / 2
plt.grid()
mean =np.mean(funct)
sigma=np.std(funct)
plt.bar(center, histraw, align='center', width=width,label="ptraw, mean ={:.4f}, std = {:.4f}".format(mean,sigma),alpha=0.5,color='green')

mean =np.mean(funct)
sigma=np.std(funct)
#def f(x):
#    return(max(histraw)*np.exp(-((x-mean)**2)/(2*(sigma**2))))
#x0=np.linspace(0,3,10000)
#plt.plot(x0,f(x0))

###

funct = np.array(y)
histpred,binspred = np.histogram(funct,bins=1000,normed=True)
width = (binspred[1] - binspred[0])
center = (binspred[:-1] + binspred[1:]) / 2

mean =np.mean(funct)
sigma=np.std(funct)

plt.bar(center, histpred, align='center', width=width,label="ptpred, mean ={:.4f}, std = {:.4f}".format(mean,sigma),alpha=0.7,color='red')
plt.grid(True)
plt.ylabel('density')
plt.xlabel('pt')

#def f(x):
#    return(max(histpred)*np.exp(-((x-mean)**2)/(2*(sigma**2))))
#x0=np.linspace(0,3,10000)
#plt.plot(x0,f(x0))



plt.legend()



fig.savefig("figure_histogram_distribution_EtaReducedStat.png",dpi=600)
#histpred= numpy.histogram(y/xptgen,bins=100,normed=True)

#plt.scatter(xpt,y/ytest,s=0.1);plt.grid();plt.show()
#plt.hist(histraw,label='raw')
#plt.hist(histpred,label='predicted')
#plt.grid();plt.show()
