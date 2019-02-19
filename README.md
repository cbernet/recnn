# QCD Aware Recursive Neural Network

## Installation

```
git clone git@github.com:cbernet/recnn.git 
cd recnn
ln -s /data/conda/recnn/data
ln -s /data/conda/recnn/data_gilles_louppe
ln -s /data/conda/recnn/models
```

Pour executer certain script de plotting il vous faudra aussi recuperer le package cpyroot:

```
cd ..
git clone git@github.com:cbernet/cpyroot.git
cd cpyroot
source init.sh
cd ../recnn
```

## Original paper from Gilles Louppe et al

[Original paper](https://arxiv.org/abs/1702.00748)

## Input data

The signal and background samples come from the CMS simulation, and are stored into ROOT files: 

```
/data2/conda/recnn/data/rootfiles/RawSignal_21Sept.root
/data2/conda/recnn/data/rootfiles/RawBackground_17july.root  
```

All Jets in the Background rootfile are to be considered as background (not hadronic taus) whereas jets in the Signal rootfile are to be considered signal only if a gen-level Tau is found very close to the jet:

```
dRs[0] != 0. && dRs[0] < 0.3
```

Original samples :

* signal :

```
  /SUSYGluGluToHToTauTau_M-*_TuneCUETP8M1_13TeV-pythia8/RunIISummer16MiniAODv2-PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/MINIAODSIM
  /SUSYGluGluToBBHToTauTau_M-*_TuneCUETP8M1_13TeV-pythia8/RunIISummer16MiniAODv2-PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/MINIAODSIM
  with * in [80,90,100,110,120,130,140,160,180,200,250,350,400,450,500,600,700,800,900,1000,1200,1400,1600,1800,2000,2300,2600,2900,3200]
```

* background :

```
  /QCD_Pt_15to30_TuneCUETP8M1_13TeV_pythia8/RunIISummer16MiniAODv2-PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/MINIAODSIM
  /QCD_Pt_30to50_TuneCUETP8M1_13TeV_pythia8/RunIISummer16MiniAODv2-PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/MINIAODSIM
  /QCD_Pt_50to80_TuneCUETP8M1_13TeV_pythia8/RunIISummer16MiniAODv2-PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/MINIAODSIM
  /QCD_Pt_80to120_TuneCUETP8M1_13TeV_pythia8/RunIISummer16MiniAODv2-PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/MINIAODSIM
  /QCD_Pt_80to120_TuneCUETP8M1_13TeV_pythia8/RunIISummer16MiniAODv2-PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6_ext2-v1/MINIAODSIM
  /QCD_Pt_120to170_TuneCUETP8M1_13TeV_pythia8/RunIISummer16MiniAODv2-PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/MINIAODSIM
  /QCD_Pt_120to170_TuneCUETP8M1_13TeV_pythia8/RunIISummer16MiniAODv2-PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6_ext1-v1/MINIAODSIM
  /QCD_Pt_170to300_TuneCUETP8M1_13TeV_pythia8/RunIISummer16MiniAODv2-PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/MINIAODSIM
  /QCD_Pt_170to300_TuneCUETP8M1_13TeV_pythia8/RunIISummer16MiniAODv2-PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6_ext1-v1/MINIAODSIM
```

These files contain a TTree called `tree` with the following branches:

All variables names are the names of the object's method.

* Jet[1][9] : any reconstructed jet with pt>20. GeV
  	* variables in order : [px,py,pz,energy,pdgId,charge,numberOfDaughters,hadronFlavour,partonFlavour]

* ptcs[200][8] : the 200 first particles of the Jet in decreasing order of pt
  	* variables in order : [px,py,pz,energy,pdgId,charge,dxy,dz]

* GenJet[1][7] : closest found gen-level Jet
	 * variables in order : [px,py,pz,energy,pdgId,charge,numberOfDaughters]

* genptcs[200][8] : the 200 first particles of the gen-level Jet in decreasing order of pt
 	 * variables in order : [px,py,pz,energy,pdgId,charge,dxy,dz]

* RawJet[1][6] : the PF jet before jet energy correction (in CMSSW : jet.uncorrectedJet(0))
  	* variables in order : [px,py,pz,energy,pdgId,charge]

* Tau[1][6] : closest reconstructed Tau
  	* variables in order : [px,py,pz,energy,pdgId,charge]

* GenTau[1][7] : closest reconstructed gen-level Tau
  	* variables in order : [px,py,pz,energy,pdgId,charge,decayMode]

* dRs[5] : all potentially interesting dRs (when it is not specified, the words "the closest" means the closest to the reconstructed jet )
  	* dRs[0] = dR between the Jet and the closest gen-level Tau
  	* dRs[1] = dR between the closest gen-level Tau and its closest reconstructed Tau
  	* dRs[2] = dR between the gen-level Jet and the closest gen-level Tau
  	* dRs[3] = dR between the Jet and its closest reconstructed Tau
  	* dRs[4] = dR between the Jet and the closest gen-level Jet

* standardID[6] : needed variables for standard ID algorithm
  	* standardID[0] = tauID('byIsolationMVArun2v1DBoldDMwLTraw')
  	* standardID[1] = tauID('decayModeFinding')
  	* standardID[2] = tau.leadChargedHadrCand().dz()
  	* standardID[3] = tauID('againstElectronVLooseMVA6')
  	* standardID[4] = tauID('againstMuonLoose3')
  	* standardID[5] = tau.decayMode()

* event[3] : event identification ( = serial numbers of the collision in which the jet was recorded)
  	* event[0] = eventId
  	* event[1] = lumi (lumisection)
  	* event[2] = run

## Preprocessing, Training, Testing

Those three steps can be done in a single cfg file that can be used like :

    nohup ipython Hadronic_taus_cfg.py <path to the working directory that will be created> > Hadronic_taus_cfg.out &
    
The rootfiles with the input and the test results are in <workdir>/rawBackground.root and <workdir>/rawSignal.root for background (QCD jets) and signal (hadronic taus) respectively. We advise to put <workdir> on a disk with a lot of space.
    
## Plot ROC curves

    python ROC.py <path to working directory>
    
The produced ROC curves will be in <workdir>/ROCs.root .
