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

## Preprocessing, Training, Testing

Those three steps can be done in a single cfg file that can be used like :

    nohup ipython Hadronic_taus_cfg.py <path to the working directory that will be created> > Hadronic_taus_cfg.out &
    
The rootfiles with the input and the test results are in <workdir>/rawBackground.root and <workdir>/rawSignal.root for background (QCD jets) and signal (hadronic taus) respectively. We advise to put <workdir> on a disk with a lot of space.
    
## Plot ROC curves

    python ROC.py <path to working directory>
    
The produced ROC curves will be in <workdir>/ROCs.root .
