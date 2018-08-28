#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 14:16:41 2018

@author: gtouquet
"""

from recnn.recnn import grnn_predict_gated
from recnn.evaluate import predict
from recnn.preprocessing import apply_tf_transform
from recnn.preprocessing import create_tf_transform
import numpy as np
from shutil import copyfile
from ROOT import TFile, TH1F
import os

def test(testarray, resultarray, model, regression=True):
    if isinstance(model, list):
        y = []
        for mod in model:
            y.append(predict(testarray, mod, func=grnn_predict_gated, regression=regression))
            print 'predicted from:', mod
    else:
        y = predict(testarray, model, func=grnn_predict_gated, regression=regression)
    print 'test data predicted'
    copyfile('/data/gtouquet/samples_root/Test_Train_splitted_Background_JEC.root',
             '/data/gtouquet/samples_root/'+model[0][model[0].rfind('/')+1:model[0].rfind('_iteration')]+'.root')
    f = TFile('/data/gtouquet/samples_root/'+model[0][model[0].rfind('/')+1:model[0].rfind('_iteration')]+'.root', 'update')
    print 'output:', '/data/gtouquet/samples_root/'+model[0][model[0].rfind('/')+1:model[0].rfind('_iteration')]+'.root'
    testtree = f.Get('testtree')
    print 'tree cloned'
    n_events = testtree.GetEntries()
    # print 'n_events :', n_events
    # print 'len(y) :', len(y)
    # if len(y) != n_events:
    #     print "Oups! root and numpy files don't have the same nevents!"
    #     import pdb;pdb.set_trace()
    predicted = {'predicted':np.ones(1)}
    name = 'pt_rec' if regression else 'passTauID'
    branch = testtree.Branch(name, predicted['predicted'], name+'/D')
    if isinstance(y, list):
        other_branches = []
        for i, pred in enumerate(y):
            n_it = model[i][model[i].rfind('iteration')+len('iteration'):model[i].rfind('.')]
            predicted[n_it] = np.ones(1)
            name = 'pt_rec'+n_it
            other_branches.append(testtree.Branch(name, predicted[n_it], name+'/D'))
    for i, event in enumerate(testtree):
        if i%10000==0:
            print i, '/', n_events
        #if i==len(y):
        #    print 'breaking, nentries>len(y)!'
        #    break
        if event.genpt!= resultarray[i]:
            print 'genpt does not match!'
            import pdb;pdb.set_trace()
        if isinstance(y, list):
            for i, pred in enumerate(y):
                n_it = model[i][model[i].rfind('iteration')+len('iteration'):model[i].rfind('.')]
                predicted[n_it][0] = pred[i]
                other_branches[i].Fill()
        else:
            predicted['predicted'][0] = y[i]
            branch.Fill()
#        testtree.Fill()
    print 'newtree filled'
    testtree.SetName('tested_tree')
    print 'after pointbreak'
    testtree.Write()
    
def tester(pattern='R=1e-05_anti-kt', steps=False):
    X,y = np.load('data/npyfilesregression/Background_JEC_train_ID_preprocessed_{}.npy'.format(pattern))
    print 'train data loaded'
    tf = create_tf_transform(X)
    print 'tf created'
    model = 'data/modelsregression/Model_{}.pickle'.format(pattern)
    if steps:
        model = []
        for i in [54]:#range(40):
            path = 'data/modelsregression/Model_{}_iteration{}.pickle'.format(pattern,i)
            if os.path.isfile(path):
                model.append(path)
    X, y = np.load('data/npyfilesregression/Background_JEC_test_ID_preprocessed_{}.npy'.format(pattern))
    print 'test data loaded'
    X = apply_tf_transform(X,tf)
    test(X, y, model,regression=True)

if __name__ == '__main__':
    patterns = ['R=1e-05_anti-kt']#,'R=1e-05_cambridge','R=1e-05_kt','R=1e-05_random','R=1e-05_seqpt','R=1e-05_seqpt_reversed','R=0.3_anti-kt','R=0.3_cambridge','R=0.3_kt','R=0.3_random','R=0.3_seqpt','R=0.3_seqpt_reversed']
    for pattern in patterns:
        tester(pattern, steps=True)
