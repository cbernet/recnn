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
#### Input file

input file : npy format containing an array of shape :
- for discrimination
```
array =[jet1,...,jetN]
jet = [particle1,...,particleM]
particle = [E,px,py,pz,standard_id]
```

- for regression (energy calibration)
```
array =[jet1,...,jetN]
jet = ([particle1,...,particleM],genpt)
particle = [E,px,py,pz,standard_id]
```

#### Preprocessing

To preprocess your files for the RECNN, you need to use the Python file `preprocess_for_training.py`. It needs to be run with **iPython**. You can either import the needed function with `from preprocess_for_training import preprocess_for_training` or use it directly.

Syntax :
```
ipython preprocess_for_training.py /path/to/myfile.npy -- Other_options
```

The options are :
- `--regression`, allows you to switch to regression mode
- `--R_clustering 0.3`, allows you to chose the re-clustering radius of your jet. 0,3 would be for subjet and 0. for second particle orientation.
- `--signal`, is usefull with regression, if your file is a background file, it need to be predicted as zero, while for signal it would be one. Use this option to preprocess the signal file.

The preprocessed files are saved under the subdirectory `npyfiles` for discrimination and `npyfilesregression` for regression. The name of a preprocessed file is `initial_name_preprocessed_R=(value)_(clustering_method).npy`.

Once you preprocessed your background and signal for discrimination, you need to combine them into a single file with `merge_two_files.py`.

Usage : `python merge_two_files.py /path/to/file1.npy /path/to/file2.npy`

#### Training

To train your RECNN you need to run `python train.py /path/to/preprocessed_train_set.npy /path/to/save/model.pickle Other_options`.

The options are :
- `--regression`, add this for regression use.
- `--n_features 12`, depends on your use-case, it's the number of features the sample contains. With our data formating, it's 12.
- `--n_hidden 40`, number of neurons in a layer of the final network.
- `--n_epochs 15`, number of epochs of training.
- `--batch_size 64`, size of your batch for gradient descent computing.
- `--step_size 0.0005`, size of you gradient descent step.
- `--decay 0.9`, decay of your step size. Each epoch will do `step_size := decay * step_size`.
- `--random_state 42`, set a random state. Default is 42.
- `--statlimit 100000`, limit sample size (usefull for ram usage).
- `--verbose`, verbose run, highly recommended.
- `--simple`, add this to use a simple network instead of a gated one.

#### Testing

Once your model have been trained, you can test it using the same type of data. Testing it means produce predictions from your model. The file `evaluate.py` contans the required functions.

Example :
```
Xtrain, ytrain = np.load('/path/to/preprocessed/trainfile.npy')
Xtest, ytest = np.load('/path/to/preprocessed/testfile.npy')
X = transform_for_prediction(Xtrain,Xtest)
y_pred = predict(X, '/path/to/model.pickle', func=grnn_predict_gated,regression = ... )
```

### This work was done by

* Gaël Touquet
* Yohann Faure

under the supervision of

* Colin Bernet

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
