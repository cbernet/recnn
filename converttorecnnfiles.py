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


def etatrimtree(tree, isSignal=False, JEC=False):
        newtree = tree.CloneTree(0)
        if JEC:
            testtree = tree.CloneTree(0)
            genpt = np.zeros(1)
            testtree.Branch('genpt', genpt, 'genpt/D')
            newtree.Branch('genpt', genpt, 'genpt/D')
        i=0
        j=0
        nentries = tree.GetEntries()
        for event in tree:
                j+=1
                if j%10000==0:
                    print 'entry', j, '/', nentries
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
                if isSignal and not JEC:
                    vect.SetPxPyPzE(event.GenJet[0],event.GenJet[1],event.GenJet[2],event.GenJet[3])
                    norm_hist.Fill(vect.Pt())
                if not isSignal and not JEC:
                    vect.SetPxPyPzE(event.GenJet[0],event.GenJet[1],event.GenJet[2],event.GenJet[3])
                    the_bin = norm_hist.FindBin(vect.Pt())
                    nsig = norm_hist.GetBinContent(the_bin)
                    nback = background_norm_hist.GetBinContent(the_bin)
                    if nback>=nsig:
                        continue
                    else:
                        background_norm_hist.Fill(vect.Pt())
                if JEC:
                    vect.SetPxPyPzE(event.GenJet[0],event.GenJet[1],event.GenJet[2],event.GenJet[3])
                    genpt[0] = vect.Pt()
                    if i%2==0:
                        testtree.Fill()
                    else:
                        newtree.Fill()
                else:
                    newtree.Fill()
                i+=1
        if JEC:
            newtree.SetName('traintree')
            testtree.SetName('testtree')
            newtree.Write()
            testtree.Write()
            return newtree, testtree
        else:
            tree.Write()
            return newtree, None
        
def converttorecnnfiles(tree, addID=False, isSignal=False, JEC=False):
        #load data
        tree, testtree = etatrimtree(tree, isSignal=isSignal, JEC=JEC)
        if JEC:
            jets_array = tree2array(tree,'ptcs')
            genpt_array = tree2array(tree,'genpt')
        else:
            jets_array = tree2array(tree,'ptcs')
        #process
        indexes = multithreadmap(find_first_non_particle, jets_array)
        
        jets_array = list(jets_array)
        
        for i in range(len(jets_array)):
	        jets_array[i] = jets_array[i][:indexes[i]]
                
        jets_array = multithreadmap(select_particle_features,jets_array, addID=addID)
        
        if JEC:
            genpt_array = list(genpt_array)
            jets_array = zip(jets_array, genpt_array)
        
        return jets_array


if __name__ == '__main__':
        import argparse
        parser = argparse.ArgumentParser(description='Convert Dataformated ROOTfiles to numpy files usable in the notebooks. \n Usage : \n python converttorecnnfiles.py /data/gtouquet/samples_root/QCD_Pt80to120_ext2_dataformat.root --isSignal False \n OR \n python converttorecnnfiles.py all')
        parser.add_argument("--rootfile", help="path to the ROOTfile to be converted OR 'all' which converts all usual datasets from gael's directory (for now, maybe put the rootfiles in data/ too?)",type=str, default='all')
        parser.add_argument("--isSignal", help=" if is it a signal file (hadronic taus) or background file (QCD jets)?", action="store_true")
        parser.add_argument("--addID", help="whether or not to add the pdgID in the output", action="store_true")
        parser.add_argument("--JEC", help="to make the files needed for JEC regression", action="store_true")
        args = parser.parse_args()

        if args.rootfile == 'all':
                if not args.JEC:
                    #Signal files
                    sinputfile = TFile('/data/gtouquet/samples_root/RawSignal.root')
                    stree = sinputfile.Get('tree')
                    outROOTfiles = TFile('/data/gtouquet/samples_root/Test_Train_splitted_{}.root'.format('Signal'),'recreate')
                    signals = converttorecnnfiles(stree,
                                                  addID=args.addID,
                                                  isSignal=True,
                                                  JEC=args.JEC)
                    outROOTfiles.Write()
                    outROOTfiles.Close()
                    np.save('data/{}{}'.format('Signal', 'JEC' if args.JEC else ''), signals)
                    
                #Background files
                binputfile = TFile('/data/gtouquet/samples_root/RawBackground.root')
                btree = binputfile.Get('tree')
                outROOTfileb = TFile('/data/gtouquet/samples_root/Test_Train_splitted_{}.root'.format('Background'),'recreate')
                backgrounds = converttorecnnfiles(btree,
                                                  addID=args.addID,
                                                  isSignal=False,
                                                  JEC=args.JEC)
                outROOTfileb.Write()
                outROOTfileb.Close()
                np.save('data/{}{}'.format('Background', 'JEC' if args.JEC else ''), backgrounds)
                
                
        else:
                inputfile = TFile(args.rootfile)
                tree = inputfile.Get('tree')
                converttorecnnfiles(tree,
                                    addID=args.addID,
                                    isSignal=args.isSignal,
                                    JEC=args.JEC)
