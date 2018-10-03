from ROOT import TH1F, TTree, TFile, TLorentzVector
import numpy as np

def histfill(tree,selection, score_retrieval):
    hist = TH1F('dummy','dummy',10000,0.,1.)
    for event in tree:
        if selection(event):
            hist.Fill(score_retrieval(event))
    return hist
            
def ROC(stree,btree,selection,score_retrieval,outname):
    bhist = histfill(btree,selection,score_retrieval)
    shist = histfill(stree,selection,score_retrieval)
    outtree = TTree(outname,outname)
    tpr = np.zeros(1)
    fpr = np.zeros(1)
    outtree.Branch("tpr",tpr,"tpr/D")
    outtree.Branch("fpr",fpr,"fpr/D")
    n_signal = shist.Integral()
    n_background = bhist.Integral()
    for i in range(10000):
        tpr[0] = 1.-(shist.Integral(0,i+1)/n_signal)
        fpr[0] = 1.-(bhist.Integral(0,i+1)/n_background)
        outtree.Fill()
    outtree.Write()
    
def selection(event):
    vect = TLorentzVector()
    vect.SetPxPyPzE(event.GenJet[0],event.GenJet[1],event.GenJet[2],event.GenJet[3])
    if abs(vect.Eta())>0.5 or vect.Pt()<50 or vect.Pt()>70:
        return False
    vect.SetPxPyPzE(event.Jet[0],event.Jet[1],event.Jet[2],event.Jet[3])
    if abs(vect.Eta())>0.5 or vect.Pt()<50:
        return False
    else:
        return True

def score_retrieval_NN(event):
    return event.antikt


def score_retrieval_std(event):
    iso_score = event.standardID[0]
    
    if iso_score != 0.: #iso_score==0. here means there was no reconstructed tau from jet => we want it to stay 0. (to say it is considered as a jet)
        iso_score += 1
        iso_score /= 2
            
    decaymode_finding = event.standardID[1] > 0.5
    if not decaymode_finding:
        iso_score *= 0.
            
    dz = abs(event.standardID[2]) < 0.2
    if not dz:
        iso_score *= 0.
            
    return iso_score

    
if __name__ == '__main__':
    workdir = '/data/gtouquet/testdir_23Sept'
    sfile = TFile(workdir+'/rawSignal.root')
    bfile = TFile(workdir+'/rawBackground.root')
    stree = sfile.Get('finaltree')
    btree = bfile.Get('finaltree')

    outfile = TFile(workdir+'/ROCs.root','recreate')

    ROC(stree,btree,selection,score_retrieval_NN,'NN_ROC')

    ROC(stree,btree,selection,score_retrieval_std,'std_ROC')
