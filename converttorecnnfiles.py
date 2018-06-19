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
import multiprocessing as mp
from ROOT import TFile
from root_numpy import tree2array
from functools import partial



def multithreadmap(f,X,ncores=20, **kwargs):
	"""
	multithreading map of a function, default on 20 cpu cores.
	"""
	p=mp.Pool(ncores)
	Xout = p.map(f,X)
	p.terminate()
	return(Xout)


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


def converttorecnnfiles(input_file, addID=False):
        #load data
        f = TFile(input_file)
        tree = f.Get('tree')
        jets_array = tree2array(tree,'ptcs')
        
        #process
        indexes = multithreadmap(find_first_non_particle, jets_array)
        
        jets_array = list(jets_array)
        
        for i in range(len(jets_array)):
	        jets_array[i] = jets_array[i][:indexes[i]]
                
        select_particle_features_ID = partial(select_particle_features, addID=addID)
        jets_array = multithreadmap(select_particle_features_ID,jets_array)

        np.save(input_file[:input_file.find('.')]+'_processed_for_recnn',jets_array)



if __name__ == '__main__':
        import argparse
        parser = argparse.ArgumentParser(description='Convert Dataformated ROOTfiles to numpy files usable in the notebooks.')
        parser.add_argument("rootfile", help="path to the ROOTfile to be converted",type=str)
        # parser.add_argument("--branchlist", help="list of branches to keep", nargs='+') # for now respect naming scheme used to produce trees
        parser.add_argument("--addID", help="wether or not to add the pdgID in the output", action="store_true")
        args = parser.parse_args()
        
        converttorecnnfiles(args.rootfile, addID=args.addID)
