from ROOT import TFile, TH1F, TF1
import numpy as np

pts = []
for i in range(60):
    if i*5.>20.:
        pts.append([i*5.,(i+1.)*5.])

etas = [[0.,0.25],[0.25,0.5],[0.5,0.75],[0.75,1.],[1.,1.3]]

def plots(tree, var='rawpt'):
    f2 = TFile('/data/gtouquet/samples_root/plots/outfile{}.root'.format(var),'recreate')
    hists = []
    for pt,ptmax in pts:
        hists.append(TH1F('hist{}'.format(int(pt)),
                          'hist{}'.format(int(pt)),
                          200,0.,2.))
    print 'starting turning on tree'
    for event in tree:
        if not hasattr(event, var):
            print 'var not in tree'
            return False
        for i, pt in enumerate(pts):
            if event.genpt > pt[0] and event.genpt < pt[1]:
                val = getattr(event,var)
                hists[i].Fill(val/event.genpt)
    print 'finished turning on tree'
    resphist = TH1F('resphist','resphist',len(pts),np.array([x[0] for x in pts]+[pts[-1][-1]]))
    reshist = TH1F('reshist','reshist',len(pts),np.array([x[0] for x in pts]+[pts[-1][-1]]))
    for i, hist in enumerate(hists):
        hist.Draw()
        rms = hist.GetRMS()
        maximum = hist.GetMaximumBin()
        f = TF1("gaus","gaus",maximum-(2*rms),maximum+(2*rms))
        f.SetParameter(0,hist.GetMaximum())
        f.SetParameter(1,hist.GetMean())
        f.SetParameter(2,rms)
        hist.Fit("gaus")
        mean = f.GetParameter(1)
        resphist.SetBinContent(i+1,mean)
        resphist.SetBinError(i+1,f.GetParError(1))
        if mean==0.:
            reshist.SetBinContent(i+1,f.GetParameter(2))
            reshist.SetBinError(i+1,f.GetParError(2))
        else:
            reshist.SetBinContent(i+1,f.GetParameter(2)/(1.-abs(1.-mean)))
            reshist.SetBinError(i+1,f.GetParError(2)/(1.-abs(1.-mean)))
    #     hist.Write()
    # resphist.Write()
    # reshist.Write()
    f2.Write()

def plots_eta(tree, var='rawpt'):
    f2 = TFile('/data/gtouquet/samples_root/plots/outfile{}_eta.root'.format(var),'recreate')
    hists = []
    for eta,etamax in etas:
        hists.append(TH1F('hist{}'.format(int(eta*100)),
                          'hist{}'.format(int(eta*100)),
                          200,0.,2.))
    print 'starting turning on tree'
    for event in tree:
        if not hasattr(event, var):
            print 'var not in tree'
            return False
        for i, eta in enumerate(etas):
            if event.geneta > eta[0] and event.geneta < eta[1] and event.genpt>150. and event.genpt<200.:
                val = getattr(event,var)
                hists[i].Fill(val/event.genpt)
    print 'finished turning on tree'
    resphist = TH1F('resphist','resphist',len(etas),np.array([x[0] for x in etas]+[etas[-1][-1]]))
    reshist = TH1F('reshist','reshist',len(etas),np.array([x[0] for x in etas]+[etas[-1][-1]]))
    for i, hist in enumerate(hists):
        hist.Draw()
        rms = hist.GetRMS()
        maximum = hist.GetMaximumBin()
        f = TF1("gaus","gaus",maximum-(2*rms),maximum+(2*rms))
        f.SetParameter(0,hist.GetMaximum())
        f.SetParameter(1,hist.GetMean())
        f.SetParameter(2,rms)
        hist.Fit("gaus")
        mean = f.GetParameter(1)
        resphist.SetBinContent(i+1,mean)
        resphist.SetBinError(i+1,f.GetParError(1))
        if mean==0.:
            reshist.SetBinContent(i+1,f.GetParameter(2))
            reshist.SetBinError(i+1,f.GetParError(2))
        else:
            reshist.SetBinContent(i+1,f.GetParameter(2)/(1.-abs(1.-mean)))
            reshist.SetBinError(i+1,f.GetParError(2)/(1.-abs(1.-mean)))
    #     hist.Write()
    # resphist.Write()
    # reshist.Write()
    f2.Write()
    

if __name__ == '__main__':
    f = TFile("/data/gtouquet/samples_root/Model_R=1e-05_anti-kt.root")
    tree = f.Get('tested_tree')
    plots(tree, var='rawpt')
    plots(tree, var='recopt')
    plots(tree, var='pt_rec')
    plots_eta(tree, var='rawpt')
    plots_eta(tree, var='recopt')
    plots_eta(tree, var='pt_rec')
