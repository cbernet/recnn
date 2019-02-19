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

These files contain a TTree called `tree` with the following branches:

* RawJet[1][6] : the PF jet before jet energy correction
   * blah
   * ...  
* genptcs[200][8]
* Jet[1][9]
* Tau[1][6]
* GenJet[1][7]
* GenTau[1][7]
* ptcs[200][8]
* dRs[5]
* standardID[6]
* event[3]

## Preprocessing, Training, Testing

Those three steps can be done in a single cfg file that can be used like :

    nohup ipython Hadronic_taus_cfg.py <path to the working directory that will be created> > Hadronic_taus_cfg.out &
    
The rootfiles with the input and the test results are in <workdir>/rawBackground.root and <workdir>/rawSignal.root for background (QCD jets) and signal (hadronic taus) respectively. We advise to put <workdir> on a disk with a lot of space.
    
## Plot ROC curves

    python ROC.py <path to working directory>
    
The produced ROC curves will be in <workdir>/ROCs.root .
