# Recursive Neural Networks for Jet Physics
## Jet Vs Tau discrimination and energy calibration


### Requirements

- python 2.7
           - numpy
           - multiprocessing
           - ROOT
           - root_numpy
           - functools
           - autograd
           - click
           - copy
           - logging
           - pickle
           - sklearn
           - cython
           - ipython
- FastJet

### Data

(mirror to be released)

### Usage for jet classification
### Rebuilding the data

input file : npy format containing an array of shape :
```
array =[jet1,...,jetN]
jet = [particle1,...,particleM]
particle = [E,px,py,pz]
```
The output files are produced using the FullPreprocessing NotreBook, you need to adapt the path.

You need to specify which file is background and which is signal, using the lists in the 6th cell (signallist and backgroundlist).

### Training
```
python train.py training_data.npy model.pickle
```
### Testing

There is still work to do here.

### This work is inspired by
QCD-Aware Recursive Neural Networks for Jet Physics
https://arxiv.org/abs/1702.00748

* Gilles Louppe
* Kyunghyun Cho
* Cyril Becot
* Kyle Cranmer

---

Please cite using the following BibTex entry:

```
@article{louppe2017qcdrecnn,
           author = {{Louppe}, G. and {Cho}, K. and {Becot}, C and {Cranmer}, K.},
            title = "{QCD-Aware Recursive Neural Networks for Jet Physics}",
          journal = {ArXiv e-prints},
    archivePrefix = "arXiv",
           eprint = {1702.00748},
     primaryClass = "hep-th",
             year = 2017,
            month = feb,
}
```

---
