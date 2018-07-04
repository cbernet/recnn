#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# In[]:
from recnn.preprocessing import create_tf_transform
from recnn.evaluate import build_rocs
import matplotlib.pyplot as plt
import numpy as np
import sys
np.seterr(divide="ignore")
sys.path.append("..")

# In[]:
basepath = '/data/conda/recnn/data'
name="anti-kt"
trainfile,testfile = basepath+"/npyfiles/subjet_oriented_"+name+"_train.npy",basepath+"/npyfiles/subjet_oriented_"+name+"_test.npy"
modelpath = basepath+"/models/subjet_oriented_"+name+"_model.pickle"


/data/conda/recnn/data/modelsregression/subjet_oriented_anti-kt_with_id_model.pickle
/data/conda/recnn/data/npyfilesregression/particle_oriented_anti-kt_with_id_train.npy
# In[]:
### Load training data ###
X, y = np.load(trainfile)
X=np.array(X).astype(dict)
y = np.array(y).astype(int)

### to rescale test data ###
tf = create_tf_transform(X,y)

### Load test data ###
X1, y1 = np.load(testfile)
X1 = np.array(X1).astype(dict)
y1 = np.array(y1).astype(int)

# In[]:
### Build the roc ###
r, f, t = build_rocs(tf, X1, y1, modelpath)
print(r)

# In[]:
plt.plot(f,t,label=name)
tpr,fpr = np.load('/data/conda/recnn/data/roccurves/standardID_ROC.npy')
plt.plot(fpr[1:],tpr[1:],label='std')
#plt.xscale("log")
plt.legend()
plt.grid()
#plt.yscale("log")
#plt.xlim([0.0001,0.1])
#plt.ylim([0.2,0.8])
#plt.savefig("first_working_nn_roc.png",dpi=600)
plt.show()
print(r)

# In[]:
basepath = '/data/conda/recnn/data'
filelist=["anti-kt","kt","random","seqpt","seqpt_reversed","cambridge"]
roc_list = []

for name in filelist:
    trainfile,testfile = basepath+"/npyfiles/subjet_oriented_"+name+"_train.npy",basepath+"/npyfiles/subjet_oriented_"+name+"_test.npy"
    modelpath = basepath+"/models/subjet_oriented_"+name+"_model.pickle"
    ### Load training data ###
    X, y = np.load(trainfile)
    X=np.array(X).astype(dict)
    y = np.array(y).astype(int)

    ### to rescale test data ###
    tf = create_tf_transform(X,y)

    ### Load test data ###
    X1, y1 = np.load(testfile)
    X1 = np.array(X1).astype(dict)
    y1 = np.array(y1).astype(int)

    ### Build the roc ###
    r, f, t = build_rocs(tf, X1, y1, modelpath)
    print(r)
    roc_list.append((f,t,'subj_'+name))
    print(r)

for name in filelist:
    trainfile,testfile = basepath+"/npyfiles/particle_oriented_"+name+"_train.npy",basepath+"/npyfiles/particle_oriented_"+name+"_test.npy"
    modelpath = basepath+"/models/particle_oriented_"+name+"_model.pickle"

    ### Load training data ###
    X, y = np.load(trainfile)
    X=np.array(X).astype(dict)
    y = np.array(y).astype(int)

    ### to rescale test data ###
    tf = create_tf_transform(X,y)

    ### Load test data ###
    X1, y1 = np.load(testfile)
    X1 = np.array(X1).astype(dict)
    y1 = np.array(y1).astype(int)

    ### Build the roc ###
    r, f, t = build_rocs(tf, X1, y1, modelpath)
    print(r)
    roc_list.append((f,t,'part_'+name))
    print(r)

# In[]:
tpr,fpr = np.load('/data/conda/recnn/data/roccurves/standardID_ROC.npy')
roc_list.append((fpr[1:],tpr[1:],"std"))

# In[]:
for i in range((len(roc_list)-1)//2):
    a,b,c=roc_list[i]
    roc_list[i]=(a,b,"subj"+c)

for i in range((len(roc_list)-1)//2):
    a,b,c=roc_list[(len(roc_list)-1)//2+i]
    roc_list[i+(len(roc_list)-1)//2]=(a,b,"subj"+c)

# In[17]:
fig=plt.figure()
fig.set_size_inches((12,12))
for (i,j,k) in roc_list :
    plt.plot(i,j,label=k)
    #plt.yscale("log")
    #plt.xlim([0.001,0.1])
    #plt.ylim([0.2,0.8])
    plt.savefig("first_comparisions_dezoom.png",dpi=600)
plt.xscale("log")
plt.legend()
plt.grid()
plt.show()