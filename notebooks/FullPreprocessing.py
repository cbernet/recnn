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
from recnn.preprocessing import recursive_format
from recnn.preprocessing import randomize
from recnn.preprocessing import preprocess
from recnn.preprocessing import multithreadmap
from recnn.preprocessing import sequentialize_by_pt

### Importing Pyplot ###
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
plt.rcParams["figure.figsize"] = (7,6)

# In[]:
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
        content = np.array(content).reshape(-1, 4)
        jets.append((tree, content, mass, pt))
        
    return jets

# In[]:
### Loading and "jetting" data with recursive_format ###
signallist = ['/Signal.npy']#'HiggsSUSYGG120',
                                  #'HiggsSUSYBB2600',
                                  #'DY1JetsToLL_M50_LO',
                                  #'HiggsSUSYBB3200',
                                  #'HiggsSUSYGG140',
                                  #'HiggsSUSYBB2300',
                                  #'HiggsSUSYGG800',
                                  #'HiggsSUSYGG160']
backgroundlist = ['/Background.npy']#'QCD_Pt15to30',
                                      #'QCD_Pt30to50',
                                      #'QCD_Pt170to300',
                                      #'QCD_Pt80to120',
                                      #'QCD_Pt80to120_ext2',
                                      #'QCD_Pt120to170',
                                      #'QCD_Pt50to80',
                                      #'QCD_Pt170to300_ext',
                                      #'QCD_Pt120to170_ext']

#def app(txt):
#    return('/'+txt+'_dataformat.npy')

#signallist = multithreadmap(app,signallist)
#backgroundlist = multithreadmap(app,backgroundlist)

background = []

for path_file in backgroundlist:
    events = np.array(np.load(basepath+path_file))
    background.extend( multithreadmap(recursive_format, events,cluster=cluster,R=1.0) )

signal = []

for path_file in signallist:
    events = np.array(np.load(basepath+path_file))
    signal.extend( multithreadmap(recursive_format, events, cluster=cluster,R=1.0) )

# In[]:
### creating files to be preprocessed ###
nmax = min(len(signal),len(background))
if nmax%2==1:
    nmax -= 1

X = np.array(background[:nmax]+signal[:nmax])
y = np.array([0]*nmax+[1]*nmax)
print(nmax)

# In[]:

#R_clustering =0.3
#f = basepath+'/npyfiles/subjet_oriented_'

R_clustering =0.0000001
f = basepath+'/npyfiles/particle_oriented_'


length = len(X)//2

### eliminate single particles ###
i=0
while i < (len(y)):
    if X[i]['tree'].shape == (1, 2):
        X,y=np.delete(X,i),np.delete(y,i)
    else :
        i+=1

# In[]:
### Save all versions of the dataset ###

### anti-kt ###

#random permutation
flush = np.random.permutation(len(X))
X_,y_ = np.copy(X[flush]),np.copy(y[flush])

#preprocess
X_ = multithreadmap(preprocess,X_,output='anti-kt',cluster=cluster,R_clustering=R_clustering)

#separate training and testing data
X_test, y_test = X_[length:],y_[length:]
X_train, y_train = X_[:length],y_[:length]

#saving
np.save(f+"anti-kt_test.npy",np.array([X_test, y_test]))

np.save(f+"anti-kt_train.npy", np.array([X_train, y_train]))

# In[]:
### kt ###
flush = np.random.permutation(len(X))
X_,y_ = np.copy(X[flush]),np.copy(y[flush])

X_ = multithreadmap(preprocess,X_,output='kt',cluster=cluster,R_clustering=R_clustering)

X_test, y_test = X_[length:],y_[length:]
X_train, y_train = X_[:length],y_[:length]

np.save(f+"kt_test.npy",np.array([X_test, y_test]))

np.save(f+"kt_train.npy", np.array([X_train, y_train]))

# In[]:
### cambridge ###
flush = np.random.permutation(len(X))
X_,y_ = np.copy(X[flush]),np.copy(y[flush])

X_ = multithreadmap(preprocess,X_,output='cambridge',cluster=cluster,R_clustering=R_clustering)

X_test, y_test = X_[length:],y_[length:]
X_train, y_train = X_[:length],y_[:length]

np.save(f+"cambridge_test.npy",np.array([X_test, y_test]))

np.save(f+"cambridge_train.npy", np.array([X_train, y_train]))
# In[]:
### random tree ###
flush = np.random.permutation(len(X))
X_,y_ = np.copy(X[flush]),np.copy(y[flush])

X_=multithreadmap(randomize,multithreadmap(preprocess,X_,output="anti-kt",cluster=cluster,R_clustering=R_clustering))

X_test, y_test = X_[length:],y_[length:]
X_train, y_train = X_[:length],y_[:length]

np.save(f+"random_test.npy",np.array([X_test, y_test]))

np.save(f+"random_train.npy", np.array([X_train, y_train]))
# In[]:
### seq by pt ###
flush = np.random.permutation(len(X))
X_,y_ = np.copy(X[flush]),np.copy(y[flush])

X_=multithreadmap(sequentialize_by_pt,multithreadmap(preprocess,X_,output="anti-kt",cluster=cluster,R_clustering=R_clustering),reverse=False)

X_test, y_test = X_[length:],y_[length:]
X_train, y_train = X_[:length],y_[:length]

np.save(f+"seqpt_test.npy",np.array([X_test, y_test]))

np.save(f+"seqpt_train.npy", np.array([X_train, y_train]))
# In[]:
### seq by pt reversed ###
flush = np.random.permutation(len(X))
X_,y_ = np.copy(X[flush]),np.copy(y[flush])

X_=multithreadmap(sequentialize_by_pt,multithreadmap(preprocess,X_,output="anti-kt",cluster=cluster,R_clustering=R_clustering),reverse=True)

X_test, y_test = X_[length:],y_[length:]
X_train, y_train = X_[:length],y_[:length]

np.save(f+"seqpt_reversed_test.npy",np.array([X_test, y_test]))

np.save(f+"seqpt_reversed_train.npy", np.array([X_train, y_train]))

# In[]:
### Verification of the formating ###
### Load data to check ###
fd = f+"anti-kt_test.npy"
X, y = np.load(fd)

# In[]:
### Check for signal ###
a1 = []
w1=[]
for i,j in enumerate(X):
    constituents = j["content"][j["tree"][:, 0] == -1]
#    if len(constituents)>1:
#        constituents = np.delete(constituents,0,0)
    if y[i]==1:
        a1.append(np.array([[LorentzVector(c).eta(), 
                            LorentzVector(c).phi()] for c in constituents]))
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

# In[]:
### For background ###
a = []
w=[]
for i,j in enumerate(X):
    constituents = j["content"][j["tree"][:, 0] == -1]
#    if len(constituents)>1:
#        constituents = np.delete(constituents,0,0)
    if y[i]==0:
        a.append(np.array([[LorentzVector(c).eta(), 
                            LorentzVector(c).phi()] for c in constituents]))
        w.append([LorentzVector(c).pt() for c in constituents])
w = [item for sublist in w for item in sublist]

w=100*np.array(w)/sum(w)
a = np.vstack(a)

# In[]:
plt.close()
t=plt.hist2d(a[:, 0], a[:, 1], range=[(-0.5,0.5), (-0.5,0.5)], 
           bins=200,  cmap=plt.cm.jet, weights=w,norm=LogNorm())
cbar = plt.colorbar()
plt.xlabel(r'$\eta$')
plt.ylabel(r'$\varphi$')
cbar.set_label(r'% of p$_t$')
#plt.savefig('non_tau_pfd_log_bis.png',dpi=600, transparent=True)
plt.show()

# In[]:
### few taus plotting ###
a = []
w=[]

njets = 10
i0=2000

i1=i0+njets

for i,j in enumerate(X[i0:i1]):
    constituents = j["content"][j["tree"][:, 0] == -1]
    if y[i+i0]==1:
        a.append(np.array([[LorentzVector(c).eta(), 
                            LorentzVector(c).phi()] for c in constituents]))
        w.append([LorentzVector(c).pt() for c in constituents])

for i in range(len(a)):
    plt.scatter(a[i][:,0],a[i][:,1],s=w[i]*100)
plt.show()