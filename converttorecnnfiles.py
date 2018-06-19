"""
Preprocessing our files to a usable format for the recnn program.

input format :
	[ [[ p1, p2, p3, ... ]] , [[ p1, p2, p3, ... ]] , [[ p1, p2, p3, ... ]] , ...]
	pi = [E,px,py,pz,id ...]
	with "fake particles", i.e. pi=[0.,0.,0.,0.,...]
output format :
	[ [ p1, p2, ... ] , ...]
	pi = [E,px,py,pz,id=0]
	without fake particles
"""


import numpy as np
import sys
import os
import multiprocessing as mp
from ROOT import TFile, TLorentzVector
from root_numpy import tree2array
from recnn.preprocessing import multithreadmap
from functools import partial




#now useless
def extract_first_element(jet):
	"""
	extract first element of the jet : its components.
	"""
	return(jet[0])


def find_first_non_particle(jet):
	"""
	dichotomy search of first  non particle of the jet
	"""
	n=len(jet)
	found = False
	first = 0
	last = n-1
	while not found and first <= last :
		m = (last+first)//2
		if jet[m][0] == 0. and jet[m-1][0] != 0. :
			found = True
		elif jet[m][0] != 0. :
			first = m+1
		else :
			last = m-1
	return(m)

def select_particle_features(jet, addID=False):
	"""
	select E,px,py,pz of the particle.
	"""
        if addID:
	        return(jet[:,[3,0,1,2,0]])
        else:
                return(jet[:,[3,0,1,2]])


def etatrimtree(tree):
        newtree = tree.CloneTree(0)
        for event in tree:
                vect = TLorentzVector()
                vect.SetPxPyPzE(event.Jet[0],event.Jet[1],event.Jet[2],event.Jet[3])
                if vect.Eta()<2.6:
                        newtree.Fill()
        return newtree
        
def converttorecnnfiles(input_file, addID=False, etacut=True):
        #load data
        f = TFile(input_file)
        tree = f.Get('tree')
        if etacut:
                tmpfile = TFile('tmp.root','recreate')
                tree = etatrimtree(tree)
        jets_array = tree2array(tree,'ptcs')
        if etacut:
                os.remove('tmp.root')
        
        #process
        indexes = multithreadmap(find_first_non_particle, jets_array)
        
        jets_array = list(jets_array)
        
        for i in range(len(jets_array)):
	        jets_array[i] = jets_array[i][:indexes[i]]
                
        jets_array = multithreadmap(select_particle_features,jets_array, addID=addID)

        np.save(input_file[:input_file.find('.')],
                jets_array)



if __name__ == '__main__':
        import argparse
        parser = argparse.ArgumentParser(description='Convert Dataformated ROOTfiles to numpy files usable in the notebooks.')
        parser.add_argument("rootfile", help="path to the ROOTfile to be converted",type=str)
        # parser.add_argument("--branchlist", help="list of branches to keep", nargs='+') # for now respect naming scheme used to produce trees
        parser.add_argument("--addID", help="wether or not to add the pdgID in the output", action="store_true")
        # parser.add_argument("--cut", help="what cut to apply to the rootfile events e.g. ",type=str)
        args = parser.parse_args()
        
        converttorecnnfiles(args.rootfile, addID=args.addID)
