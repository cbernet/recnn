#!/usr/bin/env python2
# -*- coding: utf-8 -*-

basepath = '/data/conda/recnn/data'


# In[]:
### Importing usefull packages ###
get_ipython().magic(u'load_ext cython')
import sys
import copy
import numpy as np
import multiprocessing as mp
from functools import partial
from rootpy.vector import LorentzVector
sys.path.append("..")

### Importing preprocessing functions ###
from recnn.preprocessing import ff
from recnn.preprocessing import randomize
from recnn.preprocessing import preprocess
from recnn.preprocessing import multithreadmap
from recnn.preprocessing import extract_component
from recnn.preprocessing import sequentialize_by_pt

### Importing Pyplot ###
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
plt.rcParams["figure.figsize"] = (7,6)

# In[]:
### Creating Fastjet function ###
%%cython -f -+ -I/usr/local/include --link-args=-Wl,-rpath,/usr/local/lib -lm -L/usr/local/lib -lfastjettools -lfastjet -lfastjetplugins -lsiscone_spherical -lsiscone
import numpy as np
cimport numpy as np
np.import_array()

from libcpp.pair cimport pair
from libcpp.vector cimport vector

cdef extern from "/home/yohann/Desktop/stage/recnn/notebooks/fj.cc":
    void fj(vector[double]& a, 
            vector[vector[int]]& trees, 
            vector[vector[double]]& contents, 
            vector[double]& masses, 
            vector[double]& pts, 
            double R, int jet_algorithm)
    
cpdef cluster(np.ndarray[np.double_t, ndim=2, mode="c"] a, 
              R=1.0, jet_algorithm=0):
    cdef vector[double] v
    cdef vector[vector[int]] trees
    cdef vector[vector[double]] contents
    cdef vector[double] masses
    cdef vector[double] pts 
    for value in a.ravel():
        v.push_back(value)
    fj(v, trees, contents, masses, pts, R=R, jet_algorithm=jet_algorithm)
    jets = []
    
    for tree, content, mass, pt in zip(trees, contents, masses, pts):
        tree = np.array(tree).reshape(-1, 2)
        content = np.array(content).reshape(-1, 5)
        jets.append((tree, content, mass, pt))
        
    return jets

# In[]:
### Loading and "jetting" data with ff ###
signallist = ['/Background_JEC_train_ID.npy']

signal = []

for path_file in signallist:
    events = np.array(np.load(basepath+path_file))
    signal = signal + multithreadmap(ff,events,cluster=cluster,regression=True,R=1000.)

# In[]:
### creating files to be preprocessed ###
print(len(signal))

X = np.array(signal)
y = np.array(multithreadmap(extract_component,X,component='genpt'))






for R_clustering,f in [(0.3,basepath+'/npyfilesregression/subjet_oriented_'),( 0.000001,basepath+'/npyfilesregression/particle_oriented_')]:
    
    ## In[]:
    ### eliminate single particles ###
    i=0
    while i < (len(y)):
        if X[i]['tree'].shape == (1, 2):
            X,y=np.delete(X,i),np.delete(y,i)
        else :
            i+=1
    
    ## In[]:
    ### Save all versions of the dataset ###
    
    ### anti-kt ###
    
    #random permutation
    flush = np.random.permutation(len(X))
    X_,y_ = np.copy(X[flush]),np.copy(y[flush])
    
    #preprocess
    X_ = multithreadmap(preprocess,X_,output='anti-kt',regression=True,cluster=cluster,R_clustering=R_clustering)
    
    #saving
    np.save(f+"anti-kt_with_id_train.npy",np.array([X_, y_]))
    
    ## In[]:
    ### kt ###
    flush = np.random.permutation(len(X))
    X_,y_ = np.copy(X[flush]),np.copy(y[flush])
    
    X_ = multithreadmap(preprocess,X_,output='kt',cluster=cluster,R_clustering=R_clustering)
    
    np.save(f+"kt_with_id_train.npy", np.array([X_, y_]))
    
    ## In[]:
    ### cambridge ###
    flush = np.random.permutation(len(X))
    X_,y_ = np.copy(X[flush]),np.copy(y[flush])
    
    X_ = multithreadmap(preprocess,X_,output='cambridge',cluster=cluster,R_clustering=R_clustering)
    
    np.save(f+"cambridge_with_id_train.npy", np.array([X_, y_]))
    
    ## In[]:
    ### random tree ###
    flush = np.random.permutation(len(X))
    X_,y_ = np.copy(X[flush]),np.copy(y[flush])
    
    X_=multithreadmap(randomize,multithreadmap(preprocess,X_,output="anti-kt",cluster=cluster,R_clustering=R_clustering))
    
    np.save(f+"random_with_id_train.npy", np.array([X_, y_]))
    
    ## In[]:
    ### seq by pt ###
    flush = np.random.permutation(len(X))
    X_,y_ = np.copy(X[flush]),np.copy(y[flush])
    
    X_=multithreadmap(sequentialize_by_pt,multithreadmap(preprocess,X_,output="anti-kt",cluster=cluster,R_clustering=R_clustering),reverse=False)
    
    np.save(f+"seqpt_with_id_train.npy", np.array([X_, y_]))
    
    ## In[]:
    ### seq by pt reversed ###
    flush = np.random.permutation(len(X))
    X_,y_ = np.copy(X[flush]),np.copy(y[flush])
    
    X_=multithreadmap(sequentialize_by_pt,multithreadmap(preprocess,X_,output="anti-kt",cluster=cluster,R_clustering=R_clustering),reverse=True)
    
    np.save(f+"seqpt_reversed_with_id_train.npy", np.array([X_, y_]))

# In[]:
### Verification of the formating ###
### Load data to check ###
fd = f+"anti-kt_test.npy"
X, _ = np.load(fd)

# In[]:
a1 = []
w1=[]
for i,j in enumerate(X):
    constituents = j["content"][j["tree"][:, 0] == -1]
#    if len(constituents)>1:
#        constituents = np.delete(constituents,0,0)
        w1.append([LorentzVector(c).pt() for c in constituents])
w1 = [item for sublist in w1 for item in sublist]

w1=100*np.array(w1)/sum(w1)
a1 = np.vstack(a1)

# In[]:
plt.close()
t=plt.hist2d(a1[:, 0], a1[:, 1], range=[(-0.5,0.5), (-0.5,0.5)], 
           bins=200,  cmap=plt.cm.jet,weights=w1,norm=LogNorm())
cbar = plt.colorbar()
plt.xlabel(r'$\eta$')
plt.ylabel(r'$\varphi$')
cbar.set_label(r'% of p$_t$')
#plt.savefig('tau_pfd_log_bis.png',dpi=600, transparent=True)
plt.show()