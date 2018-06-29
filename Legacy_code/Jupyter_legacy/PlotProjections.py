
# coding: utf-8

# In[1]:


basepath = '/data/conda/recnn/data'


# In[2]:


from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import multiprocessing as mp
from scipy import interp
import numpy as np
import logging
import pickle
import glob
import sys

np.seterr(divide="ignore")
sys.path.append("..")

from recnn.recnn import grnn_transform_simple
from recnn.recnn import grnn_predict_simple
from recnn.recnn import grnn_predict_gated


from recnn.preprocessing import sequentialize_by_pt
from recnn.preprocessing import rewrite_content
from recnn.preprocessing import multithreadmap
from recnn.preprocessing import permute_by_pt
from recnn.preprocessing import randomize
from recnn.preprocessing import extract




get_ipython().magic(u'matplotlib inline')
plt.rcParams["figure.figsize"] = (6, 6)


# # Loading functions

# In[6]:


def extractcontent(jet):
    return(jet["content"])

def tftransform(jet,tf):
    jet["content"] = tf.transform(jet["content"])
    return(jet)

def load_tf(filename_train, preprocess=None, n_events_train=-1):
    # Make training data
    print("Loading training data...")

    X, y = np.load(filename_train)
    X=np.array(X).astype(dict)
    y = np.array(y).astype(int)

    if n_events_train > 0:
        indices = np.random.permutation(len(X))[:n_events_train]
        X = X[indices]
        y = y[indices]

    print("\tfilename = " + filename_train)
    print("\tX size = ")
    print(len(X))
    print("\ty size = ")
    print(len(y))

    # Preprocessing 
    print("Preprocessing...")
    X = multithreadmap(rewrite_content,X)

    if preprocess:
        X = multithreadmap(preprocess,X)

    X = multithreadmap(permute_by_pt,multithreadmap(extract,X))
    Xcontent=multithreadmap(extractcontent,X)
    tf = RobustScaler().fit(np.vstack(Xcontent))

    return(tf)

def load_test(tf, filename_test, preprocess=None):
    # Make test data 
    print("Loading test data...")

    X, y = np.load(filename_test)
    X = np.array(X).astype(dict)
    y = np.array(y).astype(int)

    print("\tfilename = " + filename_test)
    print("\tX size = ")
    print(len(X))
    print("\ty size = ")
    print(len(y))
    # Preprocessing 
    print("Preprocessing...")
    X = multithreadmap(rewrite_content,X)
    
    if preprocess:
        X = multithreadmap(preprocess,X)
        
    X = multithreadmap(permute_by_pt,X)
    X = multithreadmap(extract,X)

    X=multithreadmap(tftransform,X,tf=tf)
    i=0
    while i < len(y):
        if X[i]['tree'].shape == (1, 2):
            X,y=np.delete(X,i),np.delete(y,i)
        else :
            i+=1
    return(X, y)


# ## Loading data

# In[7]:


filename_train = "/npyfiles/25Juin_anti-kt_train.npy"
filename_test = "/npyfiles/25Juin_anti-kt_test.npy"
modelpath = "/models/model_anti-kt_25Juin.pickle"


# In[8]:


tf = load_tf(basepath+filename_train)
X, y = load_test(tf, basepath+filename_test)


# In[9]:


fd = open(basepath+modelpath, "rb")
params = pickle.load(fd)
fd.close()


# ## nice projections

# In[10]:


Xt = grnn_transform_simple(params, X[:5000])
Xtt = TSNE(n_components=2).fit_transform(Xt)

fig=plt.figure()
for i in range(5000):
    plt.scatter(Xtt[i, 0], Xtt[i, 1], color="b" if y[i] == 1 else "r", alpha=0.5)

fig.set_size_inches(6.69291,4)
plt.show()


# In[11]:


Xtt = PCA(n_components=2).fit_transform(Xt)
fig=plt.figure()

for i in range(5000):
    plt.scatter(Xtt[i, 0], Xtt[i, 1], color="b" if y[i] == 1 else "r", alpha=0.5)
fig.set_size_inches(6.69291,4)
plt.show()

