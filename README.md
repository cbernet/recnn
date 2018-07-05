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
           - logging
           - pickle
           - sklearn
           - cython
           - ipython
- FastJet

### Data

(mirror to be released)

### How to use it
#### Rebuilding the data

input file : npy format containing an array of shape :
```
array =[jet1,...,jetN]
jet = [particle1,...,particleM]
particle = [E,px,py,pz,standard_id]
```

To preprocess your files for the RECNN, you need to use the Python file `preprocess_for_training.py`. It needs to be run with **iPython**. You can either import the needed function with `from preprocess_for_training import preprocess_for_training` or use it directly. Syntax :
```
ipython preprocess_for_training.py /path/to/myfile.npy -- Other_options
```



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
