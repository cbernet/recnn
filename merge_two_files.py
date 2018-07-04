#!/usr/bin/env python2
# -*- coding: utf-8 -*-
if __name__=='__main__':
    import numpy as np
    import sys

    if len(sys.argv)>2:
        X1,y1=np.load(sys.argv[1])
        X2,y2=np.load(sys.argv[2])
        X,y=np.concatenate((X1,X2)),np.concatenate((y1,y2))
        np.save(sys.argv[1][:-4]+'_merged.npy',np.array([X,y]))
    else:
        print("not enough files")