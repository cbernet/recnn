basepath = '/data/conda/recnn/data'
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
np.seterr(divide="ignore")
sys.path.append("..")
from recnn.recnn import grnn_predict_simple
from recnn.recnn import grnn_predict_gated
from recnn.preprocessing import rewrite_content
from recnn.preprocessing import multithreadmap
from recnn.preprocessing import permute_by_pt
from recnn.preprocessing import extract



# In[26]:


def extractcontent(jet):
    return(jet["content"])
def tftransform(jet,tf):
    jet["content"] = tf.transform(jet["content"])
    return(jet)

def load_tf(X, y):
    # Make training data
    X = multithreadmap(rewrite_content,X)
    X = multithreadmap(extract,X)
    Xcontent=multithreadmap(extractcontent,X)
    tf = RobustScaler().fit(np.vstack(Xcontent))
    return(tf)

def load_test(tf, X, y):
    # Make test data 
    shuf = np.random.permutation(1000)
    X = X[:1000]
    y = y[:1000]
    X=X[shuf]
    y=y[shuf]
    print("Preprocessing...")
    X = multithreadmap(rewrite_content,X)
    X = multithreadmap(permute_by_pt,X)
    X = multithreadmap(extract,X)
    X=multithreadmap(tftransform,X,tf=tf)
    return(X, y)

def roc_curve_perso(y, y_pred,density=10000):
    back = np.argwhere(y==0)
    back = back.reshape((len(back),))
    sign = np.argwhere(y==1)
    sign=sign.reshape((len(sign),))
    y_pred_sign=y_pred[sign]
    y_pred_back=y_pred[back]
    t=np.linspace(0.,1.,density)
    tpr=np.zeros(density,dtype=float)
    fpr=np.zeros(density,dtype=float)
    for i in range(density):
        tpr[i]=np.sum(y_pred_sign<=t[i])
        fpr[i]=np.sum(y_pred_back<=t[i])
    tpr=1-tpr/len(y_pred_sign)
    fpr=1-fpr/len(y_pred_back)
    return(fpr,tpr,t)


# In[27]:


def predict(X, filename, func=grnn_predict_simple):
    fd = open(filename, "rb")
    params = pickle.load(fd)
    fd.close()
    y_pred = func(params, X)
    return(y_pred)

def evaluate_models(X, y, filename, func=grnn_predict_simple):
    print("Loading " + filename),
    y_pred = predict(X, filename, func=func)
    fpr, tpr, _ = roc_curve_perso(y, y_pred,density=100)
    roc = np.trapz(-tpr,fpr)
    print("ROC AUC = %.4f" % roc)
    return(roc, fpr, tpr)

def build_rocs(tf, X1, y1, model):
    X, y = load_test(tf, X1, y1) 
    roc, fpr, tpr = evaluate_models(X, y, model, func=grnn_predict_gated)
    return(roc, fpr, tpr)

# In[]



trainfile,testfile = '/npyfiles/25Juin_anti-kt_train.npy','/npyfiles/25Juin_anti-kt_test.npy'
name = "anti-kt"
modelpath = '/models/model_anti-kt_25Juin.pickle'

trainfile=basepath+trainfile
testfile=basepath+testfile

X, y = np.load(trainfile)
X=np.array(X).astype(dict)
y = np.array(y).astype(int)
tf = load_tf(X,y)

X1, y1 = np.load(testfile)
X1 = np.array(X1).astype(dict)
y1 = np.array(y1).astype(int)


r, f, t = build_rocs(tf, X, y, basepath+modelpath)
for i in range(9):
    r1, f1, t1 = build_rocs(tf, X, y, basepath+modelpath)
    r,f,t=r1+r,f1+f,t1+t
plt.plot(f/10,t/10,label=name)
plt.show()
print(r/10)
