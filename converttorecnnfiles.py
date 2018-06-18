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



def multithreadmap(f,X,ncores=20):
	"""
	multithreading map of a function, default on 20 cpu cores.
	"""
	p=mp.Pool(ncores)
	Xout = p.map(f,X)
	p.terminate()
	return(Xout)


#now useless
#def extract_first_element(jet):
#	"""
#	extract first element of the jet : its components.
#	"""
#	return(jet[0])


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

def select_particle_features(jet):
	"""
	select E,px,py,pz of the particle.
	"""
	return(jet[:,[3,0,1,2,0]])


#Load file
if len(sys.argv)>1:
	input_file = sys.argv[1]
else :
	input_file = '/data/yohann/QCD_Pt170to300_dataformat.npy'

#load data
jets_array = np.load(input_file)

#process
#jets_array = multithreadmap(extract_first_element, jets_array)

indexes = multithreadmap(find_first_non_particle, jets_array)

for i in range(len(jets_array)):
	jets_array[i] = jets_array[i][:indexes[i]]


jets_array = multithreadmap(select_particle_features,jets_array)



np.save(input_file[:-4]+'_processed_for_recnn',jets_array)
