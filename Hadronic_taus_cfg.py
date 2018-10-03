from root_to_npy import makerecnnfiles_classification
from ROOT import TLorentzVector
import numpy as np
import os
from shutil import copyfile
from preprocess_for_training import preprocess_for_training
from train import train
from test import test


import argparse
parser = argparse.ArgumentParser(description = 'cfg file for fadronic tau recnn training and testing')
parser.add_argument("output_path", help = "directory that will be created and be used as based directory.",type = str)
args = parser.parse_args()

#OPTIONS
def traincut(event):
    vect = TLorentzVector()
    vect.SetPxPyPzE(event.GenJet[0],event.GenJet[1],event.GenJet[2],event.GenJet[3])
    if abs(vect.Eta())>0.8 or vect.Pt()<20 or vect.Pt()>100:
        return False
    vect.SetPxPyPzE(event.Jet[0],event.Jet[1],event.Jet[2],event.Jet[3])
    if abs(vect.Eta())>0.8 or vect.Pt()<20:
        return False
    else:
        return True

outputdir = args.output_path
R_clustering = 0.0000001
orderings = ['anti-kt']#'kt','cambridge',,'random','seqpt','seqpt_reversed'

###################outputdir
if not os.path.exists(outputdir):
    os.makedirs(outputdir)

####################original rootfiles
if not os.path.exists(outputdir+'/rawSignal.root'):
    copyfile('/data/gtouquet/samples_root/RawSignal_21Sept.root',outputdir+'/rawSignal.root')
    copyfile('/data/gtouquet/samples_root/RawBackground_17july.root',outputdir+'/rawBackground.root')
    
####################raw numpy arrays 
print '################converting to npy################' 
if os.path.exists(outputdir+'/unpreproc_test_background.npy'):
    train_array_signal = np.load(outputdir+'/unpreproc_train_signal.npy')
    test_array_signal = np.load(outputdir+'/unpreproc_test_signal.npy')
    train_array_background = np.load(outputdir+'/unpreproc_train_background.npy')
    test_array_background = np.load(outputdir+'/unpreproc_test_background.npy') 
else:
    train_array_signal, test_array_signal, train_array_background, test_array_background = makerecnnfiles_classification(outputdir+'/rawSignal.root', outputdir+'/rawBackground.root', traincut)
    np.save(outputdir+'/unpreproc_train_signal.npy', train_array_signal)
    np.save(outputdir+'/unpreproc_test_signal.npy', test_array_signal)
    np.save(outputdir+'/unpreproc_train_background.npy', train_array_background)
    np.save(outputdir+'/unpreproc_test_background.npy', test_array_background )
    
###################preprocessing
print '################preprocessing################'
ordering_dict = {}
if not os.path.exists(outputdir+'/preprocessed_test_background_{}.npy'.format(orderings[-1])):
    os.system('ipython preprocess_for_training.py -- {}  --R_clustering {} --issignal --tosavefilename {}'.format(outputdir+'/unpreproc_train_signal.npy',R_clustering,outputdir+'/preprocessed_train_signal_'))
    os.system('ipython preprocess_for_training.py -- {}  --R_clustering {} --issignal --tosavefilename {}'.format(outputdir+'/unpreproc_test_signal.npy',R_clustering,outputdir+'/preprocessed_test_signal_'))
    os.system('ipython preprocess_for_training.py -- {}  --R_clustering {} --tosavefilename {}'.format(outputdir+'/unpreproc_train_background.npy',R_clustering,outputdir+'/preprocessed_train_background_'))
    os.system('ipython preprocess_for_training.py -- {}  --R_clustering {} --tosavefilename {}'.format(outputdir+'/unpreproc_test_background.npy',R_clustering,outputdir+'/preprocessed_test_background_'))
    # preprocess_for_training(outputdir+'/unpreproc_train_signal.npy',R_clustering=R_clustering,issignal=True,tosavefilename=outputdir+'/preprocessed_train_signal_')
    # preprocess_for_training(outputdir+'/unpreproc_test_signal.npy',R_clustering=R_clustering,issignal=True,tosavefilename=outputdir+'/preprocessed_test_signal_')
    # preprocess_for_training(outputdir+'/unpreproc_train_background.npy',R_clustering=R_clustering,issignal=False,tosavefilename=outputdir+'/preprocessed_train_background_')
    # preprocess_for_training(outputdir+'/unpreproc_test_background.npy',R_clustering=R_clustering,issignal=False,tosavefilename=outputdir+'/preprocessed_test_background_')
ordering_dict = {}
for ordering in orderings:
    train_signal = np.load(outputdir+'/preprocessed_train_signal_{}.npy'.format(ordering))
    test_signal = np.load(outputdir+'/preprocessed_test_signal_{}.npy'.format(ordering))
    train_background = np.load(outputdir+'/preprocessed_train_background_{}.npy'.format(ordering))
    test_background = np.load(outputdir+'/preprocessed_test_background_{}.npy'.format(ordering))
    X1,y1 = train_signal
    X2,y2 = train_background
    X,y=np.concatenate((X1,X2)),np.concatenate((y1,y2))
    np.save(outputdir+'/preprocessed_train_merged_{}.npy'.format(ordering), np.array([X,y]))
    ordering_dict[ordering] = [train_signal,
                               test_signal,
                               train_background,
                               test_background,
                               [X,y]]


#################training
print '################training################'
#n_epochs=10 --step_size=0.001 --decay=0.1 --verbose
if not os.path.exists(outputdir+'/model_{}.npy'.format(orderings[0])):
    for ordering in orderings:
        train(outputdir+'/preprocessed_train_merged_{}.npy'.format(ordering),
              outputdir+'/model_{}.npy'.format(ordering),
              regression=False,
              simple=False,
              n_features=14,
              n_hidden=40,
              n_epochs=10,
              batch_size=64,
              step_size=0.001,
              decay=0.5,
              verbose=True)


#################testing
print '################testing################'
for ordering in orderings:
    test(outputdir+'/preprocessed_test_signal_{}.npy'.format(ordering),
         outputdir+'/model_{}.npy'.format(ordering),
         outputdir+'/preprocessed_train_merged_{}.npy'.format(ordering),
         outputdir+'/rawSignal.root',
         ordering if ordering != 'anti-kt' else 'antikt',
         True)
    test(outputdir+'/preprocessed_test_background_{}.npy'.format(ordering),
         outputdir+'/model_{}.npy'.format(ordering),
         outputdir+'/preprocessed_train_merged_{}.npy'.format(ordering),
         outputdir+'/rawBackground.root',
         ordering if ordering != 'anti-kt' else 'antikt',
         False)
