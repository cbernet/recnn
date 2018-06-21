
# coding: utf-8

# # Processing training data from raw files

# # Notebook n°1

# In[80]:


basepath = '/data/conda/recnn/data'


# In[85]:


get_ipython().magic(u'load_ext cython')


import numpy as np
import h5py
import multiprocessing as mp
import sys
sys.path.append("..")
import pickle
from functools import partial
from recnn.preprocessing import randomize
from recnn.preprocessing import sequentialize_by_pt
import copy
from recnn.preprocessing import _pt
from rootpy.vector import LorentzVector
from recnn.preprocessing import multithreadmap
from functools import partial


# In[86]:


get_ipython().run_cell_magic(u'cython', u'-f -+ -I/usr/local/include --link-args=-Wl,-rpath,/usr/local/lib -lm -L/usr/local/lib -lfastjettools -lfastjet -lfastjetplugins -lsiscone_spherical -lsiscone', u'import numpy as np\ncimport numpy as np\nnp.import_array()\n\nfrom libcpp.pair cimport pair\nfrom libcpp.vector cimport vector\n\ncdef extern from "/home/yohann/Desktop/stage/recnn/notebooks/fj.cc":\n    void fj(vector[double]& a, \n            vector[vector[int]]& trees, \n            vector[vector[double]]& contents, \n            vector[double]& masses, \n            vector[double]& pts, \n            double R, int jet_algorithm)\n    \ncpdef cluster(np.ndarray[np.double_t, ndim=2, mode="c"] a, \n              R=1.0, jet_algorithm=0):\n    cdef vector[double] v\n    cdef vector[vector[int]] trees\n    cdef vector[vector[double]] contents\n    cdef vector[double] masses\n    cdef vector[double] pts \n    for value in a.ravel():\n        v.push_back(value)\n    \n    fj(v, trees, contents, masses, pts, R=R, jet_algorithm=jet_algorithm)\n    jets = []\n    \n    for tree, content, mass, pt in zip(trees, contents, masses, pts):\n        tree = np.array(tree).reshape(-1, 2)\n        content = np.array(content).reshape(-1, 4)\n        jets.append((tree, content, mass, pt))\n        \n    return jets')


# In[87]:


def cast(event, soft=0):
    a = np.zeros((len(event)+soft, 4))
    
    for i, p in enumerate(event):
        a[i, 3] = p[0]
        a[i, 0] = p[1]
        a[i, 1] = p[2]
        a[i, 2] = p[3]
    return a


# In[88]:


def ff(e):
    t=cast(e, soft=0)
    tree, content, mass, pt = cluster(t, jet_algorithm=1)[0]  # dump highest pt jet only
    jet = {}
    
    jet["root_id"] = 0
    jet["tree"] = tree
    jet["content"] = content
    jet["mass"] = mass
    jet["pt"] = pt
    jet["energy"] = content[0, 3]

    px = content[0, 0]
    py = content[0, 1]
    pz = content[0, 2]
    p = (content[0, 0:3] ** 2).sum() ** 0.5
    eta = 0.5 * (np.log(p + pz) - np.log(p - pz))
    phi = np.arctan2(py, px)
    
    jet["eta"] = eta
    jet["phi"] = phi
    
    return(jet)


# In[89]:


events = np.array(np.load(basepath+'/QCD_Pt80to120_ext2_dataformat.npy'))
background = multithreadmap(ff,events)

events = np.array(np.load(basepath+'/HiggsSUSYGG160_dataformat.npy'))
signal = multithreadmap(ff,events)


# # Notebook n°2

# ## W vs QCD

# In[90]:


X = np.array(signal[:100000])
y = np.array([1]*100000)


# # Notebook n°3

# ### preprocessing function

# In[91]:


get_ipython().run_cell_magic(u'cython', u'-f -+ -I/usr/local/include --link-args=-Wl,-rpath,/usr/local/lib -lm -L/usr/local/lib -lfastjettools -lfastjet -lfastjetplugins -lsiscone_spherical -lsiscone', u'import numpy as np\ncimport numpy as np\nnp.import_array()\n\nfrom libcpp.pair cimport pair\nfrom libcpp.vector cimport vector\n\ncdef extern from "/home/yohann/Desktop/stage/recnn/notebooks/fj.cc":\n    void fj(vector[double]& a, \n            vector[vector[int]]& trees, \n            vector[vector[double]]& contents, \n            vector[double]& masses, \n            vector[double]& pts, \n            double R, int jet_algorithm)\n    \ncpdef cluster(np.ndarray[np.double_t, ndim=2, mode="c"] a, \n              R=0.3, jet_algorithm=0):\n    cdef vector[double] v\n    cdef vector[vector[int]] trees\n    cdef vector[vector[double]] contents\n    cdef vector[double] masses\n    cdef vector[double] pts \n    for value in a.ravel():\n        v.push_back(value)\n    \n    fj(v, trees, contents, masses, pts, R=R, jet_algorithm=jet_algorithm)\n    jets = []\n    \n    for tree, content, mass, pt in zip(trees, contents, masses, pts):\n        tree = np.array(tree).reshape(-1, 2)\n        content = np.array(content).reshape(-1, 4)\n        jets.append((tree, content, mass, pt))\n        \n    return jets')


# In[92]:


# Preprocessing algorithm:
# 1. j = the highest pt anti-kt jet (R=1)
# 2. run kt (R=0.3) on the constituents c of j, resulting in subjets sj1, sj2, ..., sjN
# 3. phi = sj1.phi(); for all c, do c.rotate_z(-phi)
# 4. bv = sj1.boost_vector(); bv.set_perp(0); for all c, do c.boost(-bv)
# 5. deltaz = sj1.pz - sj2.pz; deltay = sj1.py - sj2.py; alpha = -atan2(deltaz, deltay); for all c, do c.rotate_x(alpha)
# 6. if sj3.pz < 0: for all c, do c.set_pz(-c.pz)
# 7. finally recluster all transformed constituents c into a single jet (using kt or anti-kt? r?)

def preprocess(jet, output="kt", colinear_splits=0, trimming=0.0):
    jet = copy.deepcopy(jet)
    constituents = jet["content"][jet["tree"][:, 0] == -1]

    # run kt (R=0.3) on the constituents c of j, resulting in subjets sj1, sj2, ..., sjN
    subjets = cluster(constituents, R=0.001, jet_algorithm=0)
    # phi = sj1.phi()
    # for all c, do c.rotate_z(-phi)
    v = subjets[0][1][0]
    v = LorentzVector(v)

    phi = v.phi()
    
    for _, content, _, _ in subjets:
        for i in range(len(content)):
            v = LorentzVector(content[i])
            v.rotate_z(-phi)
            content[i, 0] = v[0]
            content[i, 1] = v[1]
            content[i, 2] = v[2]
            content[i, 3] = v[3]

    # bv = sj1.boost_vector()
    # bv.set_perp(0)
    # for all c, do c.boost(-bv)
    v = subjets[0][1][0]
    v = LorentzVector(v)
    bv = v.boost_vector()
    bv.set_perp(0)
    for _, content, _, _ in subjets:
        for i in range(len(content)):
            v = LorentzVector(content[i])
            v.boost(-bv)
            content[i, 0] = v[0]
            content[i, 1] = v[1]
            content[i, 2] = v[2]
            content[i, 3] = v[3]
    # deltaz = sj1.pz - sj2.pz
    # deltay = sj1.py - sj2.py
    # alpha = -atan2(deltaz, deltay)
    # for all c, do c.rotate_x(alpha)
    if len(subjets) >= 2:
        deltaz = subjets[0][1][0, 2] - subjets[1][1][0, 2]
        deltay = subjets[0][1][0, 1] - subjets[1][1][0, 1]
        alpha = -np.arctan2(deltaz, deltay)
        for _, content, _, _ in subjets:
            for i in range(len(content)):
                v = LorentzVector(content[i])
                v.rotate_x(alpha)
                content[i, 0] = v[0]
                content[i, 1] = v[1]
                content[i, 2] = v[2]
                content[i, 3] = v[3]
    # if sj3.pz < 0: for all c, do c.set_pz(-c.pz)
    if len(subjets) >= 3 and subjets[2][1][0, 2] < 0:
        for _, content, _, _ in subjets:
            for i in range(len(content)):
                content[i, 2] *= -1.0
                
    # finally recluster all transformed constituents c into a single jet 
    constituents = []
    
    for tree, content, _, _ in subjets:
        constituents.append(content[tree[:, 0] == -1])
        
    constituents = np.vstack(constituents)

    if output == "anti-kt":
        subjets = cluster(constituents, R=100., jet_algorithm=1)
    elif output == "kt":
        subjets = cluster(constituents, R=100., jet_algorithm=0)
    elif output == "cambridge":
        subjets = cluster(constituents, R=100., jet_algorithm=2)
    else:
        raise
    
    
    jet["tree"] = subjets[0][0]
    jet["content"] = subjets[0][1]
    
    v = LorentzVector(jet["content"][0])
    jet["phi"] = v.phi()
    jet["eta"] = v.eta()
    jet["energy"] = v.E()
    jet["mass"] = v.m()
    jet["pt"] = v.pt()
    jet["root_id"] = 0
    
    return jet


# ### Convert data

# In[93]:


f = basepath+'/picklefiles/preprocessed_training_set'

length = len(X)//2


# In[94]:
#jet = X[0]
#jet = copy.deepcopy(jet)
#constituents = jet["content"][jet["tree"][:, 0] == -1]
## run kt (R=0.3) on the constituents c of j, resulting in subjets sj1, sj2, ..., sjN
#subjets = cluster(constituents, R=0.1, jet_algorithm=0)


# In[ ]:

#Save all versions of the dataset

#anti-kt
#import pickle
#fd = open("/data/conda/recnn/data_gilles_louppe/w-vs-qcd/final/antikt-kt-test.pickle", "rb")
#X, y = pickle.load(fd)
#fd.close()

Xbis=X
ybis=y
X_,y_ = np.copy(Xbis),np.copy(ybis)

X_ = multithreadmap(preprocess,X_,output='anti-kt')



X_train, y_train = X_,y_



# In[ ]:


import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

plt.rcParams["figure.figsize"] = (7,6)


# In[96]:


#For tau

a1 = []
w1=[]
err=[]
for i,j in enumerate(X_train[:10000]):
    constituents = j["content"][j["tree"][:, 0] == -1]
    if len(constituents)>1:
        constituents = constituents[1:]
    if y_train[i]==1 or True:
        for c in constituents:
            eta=LorentzVector(c).eta()
            phi=LorentzVector(c).phi()
            if -0.18 < eta < -0.17 and -0.18<phi<-0.17:
                err.append(i)
            a1.append(np.array([eta, 
                            phi]))
        w1.append([LorentzVector(c).pt() for c in constituents])
noerr=[i for i in range(len(w1)) if i not in err]
w1=np.array(w1)[noerr]
w1 = [item for sublist in w1 for item in sublist]

w1=np.array(w1)/sum(w1)

a1=np.vstack(np.array(a1)[noerr])

# In[100]:


plt.close()
t=plt.hist2d(a1[:, 0], a1[:, 1], range=[(-0.18,-0.17), (-0.18,-0.17)], 
           bins=50,  cmap=plt.cm.jet)
cbar = plt.colorbar()
#cbar.ax.set_yticklabels((np.array(cbar.get_ticks())*100/max(cbar.get_ticks())).astype(int))
plt.xlabel(r'$\eta$')
plt.ylabel(r'$\varphi$')
cbar.set_label(r'% of p$_t$')
#plt.savefig('tau_pfd_zoom.png',dpi=600, transparent=True)
plt.show()




# In[104]:


a = []
w=[]
k=4
for i,j in enumerate([np.array(X_train)[err[k]]]):
    constituents = j["content"][j["tree"][:, 0] == -1]
    if y[i]==1 or True:
        a.append(np.array([[LorentzVector(c).eta(), 
                            LorentzVector(c).phi()] for c in constituents]))
        w.append([LorentzVector(c).pt() for c in constituents])
#w = [item for sublist in w for item in sublist]


for i in range(len(a)):
    wbis = np.array(w[i])
    plt.scatter(a[i][:,0],a[i][:,1],s=(wbis))
plt.show()

