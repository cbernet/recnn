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

import os
import numpy as np
from ROOT import TFile, TLorentzVector, TH1F
from root_numpy import tree2array
from recnn.preprocessing import multithreadmap

norm_hist = TH1F('normhist','normhist',150,0.,150.)
background_norm_hist = TH1F('backgroundnormhist','backgroundnormhist',150,0.,150.)


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


def etatrimtree(tree, isSignal=False):
        newtree = tree.CloneTree(0)
        i=0
        for event in tree:
                vect = TLorentzVector()
                vect.SetPxPyPzE(event.Jet[0],event.Jet[1],event.Jet[2],event.Jet[3])
                if abs(vect.Eta())>2.3 or vect.Pt()<20.:
                        continue
                is_Signal = (event.dRs[0] < 0.3 and event.dRs[0]!=0.)
                if isSignal:
                        if not is_Signal:
                                continue
                        vect.SetPxPyPzE(event.GenTau[0],event.GenTau[1],event.GenTau[2],event.GenTau[3])
                        if abs(vect.Eta())>2.3 or vect.Pt()<20 or vect.Pt()>100:
                                continue
                        vect.SetPxPyPzE(event.GenJet[0],event.GenJet[1],event.GenJet[2],event.GenJet[3])
                        if abs(vect.Eta())>2.3 or vect.Pt()<20 or vect.Pt()>100:
                                continue
                else:
                        if is_Signal:
                                continue
                        vect.SetPxPyPzE(event.GenJet[0],event.GenJet[1],event.GenJet[2],event.GenJet[3])
                        if abs(vect.Eta())>2.3 or vect.Pt()<20 or vect.Pt()>100:
                                continue
                if isSignal:
                    vect.SetPxPyPzE(event.GenJet[0],event.GenJet[1],event.GenJet[2],event.GenJet[3])
                    norm_hist.Fill(vect.Pt())
                if not isSignal:
                    vect.SetPxPyPzE(event.GenJet[0],event.GenJet[1],event.GenJet[2],event.GenJet[3])
                    the_bin = norm_hist.FindBin(vect.Pt())
                    nsig = norm_hist.GetBinContent(the_bin)
                    nback = background_norm_hist.GetBinContent(the_bin)
                    if nback>=nsig:
                        continue
                    else:
                        background_norm_hist.Fill(vect.Pt())
                newtree.Fill()
                i+=1
        return newtree
        
def converttorecnnfiles(input_file, addID=False, etacut=True, isSignal=False):
        #load data
        f = TFile(input_file)
        tree = f.Get('tree')
        if etacut:
                tmpfile = TFile('tmp.root','recreate')
                tree = etatrimtree(tree, isSignal=isSignal)
        jets_array = tree2array(tree,'ptcs')
        if etacut:
                os.remove('tmp.root')
        
        #process
        indexes = multithreadmap(find_first_non_particle, jets_array)
        
        jets_array = list(jets_array)
        
        for i in range(len(jets_array)):
	        jets_array[i] = jets_array[i][:indexes[i]]
                
        jets_array = multithreadmap(select_particle_features,jets_array, addID=addID)
        if input_file=='QCD_Pt120to170_ext':
            import pdb;pdb.set_trace()
        return jets_array
#        np.save('data/{}'.format(input_file[input_file.rfind('/')+1:input_file.find('.')]),
#                jets_array)



if __name__ == '__main__':
        import argparse
        parser = argparse.ArgumentParser(description='Convert Dataformated ROOTfiles to numpy files usable in the notebooks. \n Usage : \n python converttorecnnfiles.py /data/gtouquet/samples_root/QCD_Pt80to120_ext2_dataformat.root --isSignal False \n OR \n python converttorecnnfiles.py all')
        parser.add_argument("rootfile", help="path to the ROOTfile to be converted OR 'all' which converts all usual datasets from gael's directory (for now, maybe put the rootfiles in data/ too?)",type=str)
        parser.add_argument("--isSignal", help=" if is it a signal file (hadronic taus) or background file (QCD jets)?",type=bool, default=False)
        parser.add_argument("--addID", help="whether or not to add the pdgID in the output", action="store_true")
        args = parser.parse_args()

        if args.rootfile == 'all':
                signal_samples = ['HiggsSUSYGG120',
                                  'HiggsSUSYBB2600',
                                  'DY1JetsToLL_M50_LO',
                                  'HiggsSUSYBB3200',
                                  'HiggsSUSYGG140',
                                  'HiggsSUSYBB2300',
                                  'HiggsSUSYGG800',
                                  'HiggsSUSYGG160']
                background_samples = ['QCD_Pt15to30',
                                      'QCD_Pt30to50',
                                      'QCD_Pt170to300',
                                      'QCD_Pt80to120',
                                      'QCD_Pt80to120_ext2',
                                      'QCD_Pt120to170',
                                      'QCD_Pt50to80',
                                      'QCD_Pt170to300_ext',
                                      'QCD_Pt120to170_ext']
                signals = []
                for s in signal_samples:
                        print 'Converting', s
                        rootfile = '{}{}_dataformat.root'.format('/data/gtouquet/samples_root/',s)
                        signals.extend(converttorecnnfiles(rootfile,
                                                           addID=args.addID,
                                                           isSignal=True))
                np.save('data/{}'.format('Signal'), signals)
                backgrounds = []
                for s in background_samples:
                        print 'Converting', s
                        rootfile = '{}{}_dataformat.root'.format('/data/gtouquet/samples_root/',s)
                        backgrounds.extend(converttorecnnfiles(rootfile,
                                                               addID=args.addID,
                                                               isSignal=False))
                np.save('data/{}'.format('Background'), backgrounds)
                
                
        else:
                converttorecnnfiles(args.rootfile,
                            addID=args.addID,
                            isSignal=args.isSignal)
