#!/usr/bin/env python2
# -*- coding: utf-8 -*-

### Importing usefull packages ###

import sys
import numpy as np
sys.path.append("..")

### Importing preprocessing functions ###
from recnn.preprocessing import ff
from recnn.preprocessing import randomize
from recnn.preprocessing import preprocess
from recnn.preprocessing import multithreadmap
from recnn.preprocessing import extract_component
from recnn.preprocessing import sequentialize_by_pt
from recnn.preprocessing import rewrite_content
from recnn.preprocessing import permute_by_pt
from recnn.preprocessing import extract
get_ipython().magic(u'load_ext cython')

# In[] :
get_ipython().run_cell_magic(u'cython', u'-f -+ -I/usr/local/include --link-args=-Wl,-rpath,/usr/local/lib  -lm -L/usr/local/lib -lfastjettools -lfastjet -lfastjetplugins -lsiscone_spherical -lsiscone', u'import numpy as np\ncimport numpy as np\nnp.import_array()\n\nfrom libcpp.pair cimport pair\nfrom libcpp.vector cimport vector\n\ncdef extern from "/home/yohann/Desktop/stage/recnn/fj.cc":\n    void fj(vector[double]& a, \n            vector[vector[int]]& trees, \n            vector[vector[double]]& contents, \n            vector[double]& masses, \n            vector[double]& pts, \n            double R, int jet_algorithm)\n    \ncpdef cluster(np.ndarray[np.double_t, ndim=2, mode="c"] a, \n              R=1., jet_algorithm=0):\n    cdef vector[double] v\n    cdef vector[vector[int]] trees\n    cdef vector[vector[double]] contents\n    cdef vector[double] masses\n    cdef vector[double] pts \n    for value in a.ravel():\n        v.push_back(value)\n    \n    fj(v, trees, contents, masses, pts, R=R, jet_algorithm=jet_algorithm)\n    jets = []\n    \n    for tree, content, mass, pt in zip(trees, contents, masses, pts):\n        tree = np.array([e for e in tree]).reshape(-1, 2)\n        content = np.array([e for e in content]).reshape(-1, 5)\n        jets.append((tree, content, mass, pt))\n        \n    return jets')

# In[]:
def preprocess_for_training(filename,regression=False,R_clustering=0.3,signal=True):
    events = np.array(np.load(filename))
    signal = multithreadmap(ff,events,cluster=cluster,regression=True,R=1000.)
    
    X = np.array(signal)
    if regression :
        y = np.array(multithreadmap(extract_component,X,component='genpt'))
    else :
        if signal:
            y = np.ones(len(X),dtype=int)
        else :
            y = np.zeros(len(X),dtype=int)
    
    ### Define paths ###
    d = filename[::-1].find('/')
    if regression :
        tosavefilename = filename[:-d]+'npyfilesregression/'+filename[-d:-4]+'_preprocessed_R='+str(R_clustering)+'_'
    else :
        tosavefilename = filename[:-d]+'npyfiles/'+filename[-d:-4]+'_preprocessed_R='+str(R_clustering)+'_'

    print('### anti-kt ###')        
    X_ = np.copy(X)
    X_ = multithreadmap(preprocess,X_,output='anti-kt',regression=regression,cluster=cluster,R_clustering=R_clustering)
    X_=multithreadmap(rewrite_content,X_)
    X_=multithreadmap(permute_by_pt,X_)
    X_=multithreadmap(extract,X_)
    np.save(tosavefilename+"anti-kt.npy",np.array([X_, y]))
    
    print('### kt ###')     
    X_ = np.copy(X)
    X_ = multithreadmap(preprocess,X_,output='kt',regression=regression,cluster=cluster,R_clustering=R_clustering)        
    X_=multithreadmap(rewrite_content,X_)
    X_=multithreadmap(permute_by_pt,X_)
    X_=multithreadmap(extract,X_)
    np.save(tosavefilename+"kt.npy", np.array([X_, y]))
    
    print('### cambridge ###')
    X_ = np.copy(X)
    X_ = multithreadmap(preprocess,X_,output='cambridge',regression=regression,cluster=cluster,R_clustering=R_clustering)        
    X_=multithreadmap(rewrite_content,X_)
    X_=multithreadmap(permute_by_pt,X_)
    X_=multithreadmap(extract,X_)
    np.save(tosavefilename+"cambridge.npy", np.array([X_, y]))


    X=multithreadmap(preprocess,X,output="anti-kt",regression=regression,cluster=cluster,R_clustering=R_clustering)    
    print('### random tree ###')
    X_ = np.copy(X)
    X_=multithreadmap(randomize,X_)
    X_=multithreadmap(rewrite_content,X_)
    X_=multithreadmap(permute_by_pt,X_)
    X_=multithreadmap(extract,X_)
    np.save(tosavefilename+"random.npy", np.array([X_, y]))
    
    print('### seq by pt ###')
    X_ = np.copy(X)
    X_=multithreadmap(sequentialize_by_pt,X_,reverse=False)
    X_=multithreadmap(rewrite_content,X_)
    X_=multithreadmap(permute_by_pt,X_)
    X_=multithreadmap(extract,X_)
    np.save(tosavefilename+"seqpt.npy", np.array([X_, y]))
    
    print('### seq by pt reversed ###')
    X_ = np.copy(X)
    X_=multithreadmap(sequentialize_by_pt,X_,reverse=True)
    np.save(tosavefilename+"seqpt_reversed.npy", np.array([X_, y]))
    X_=multithreadmap(rewrite_content,X_)
    X_=multithreadmap(permute_by_pt,X_)
    X_=multithreadmap(extract,X_)
    return(None)

# In[]:
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='lol')
    parser.add_argument("filename", help="",type=str)
    parser.add_argument("--regression", help="", action="store_true")
    parser.add_argument("--R_clustering", help="0.3 for subjet, 0 for particle", type=float, default=0.3)
    parser.add_argument("--signal", help="", action="store_true")
    args = parser.parse_args()

    if args.R_clustering == 0.:
        args.R_clustering = 0.0000001
    
    preprocess_for_training(args.filename,regression=args.regression,R_clustering=args.R_clustering,signal=args.signal)
