import numpy as np
import copy
import pickle
from functools import partial
from rootpy.vector import LorentzVector
from sklearn.preprocessing import RobustScaler
import multiprocessing as mp

# Data loading related

def multithreadmap(f,X,ncores=20, **kwargs):
	"""
	multithreading map of a function, default on 20 cpu cores.
	"""
	func = partial(f, **kwargs)
	p=mp.Pool(ncores)
	Xout = p.map(func,X)
	p.terminate()
	return(Xout)


def create_tf_transform(X):
    """loads training data and make a robustscaler transform"""
    # Make training data
    Xcontent=multithreadmap(extract_component,X,component="content")
    tf = RobustScaler().fit(np.vstack(Xcontent))
    return(tf)

def tftransform(jet,tf):
    """applies a robustscaler transform to one jet"""
    jet["content"] = tf.transform(jet["content"])
    return(jet)

def apply_tf_transform(X,tf):
    """applies a robustscaler transform to a jet array"""
    return(multithreadmap(tftransform,X,tf=tf))


def extract_component(e,component):
    return(e[component])

def cast(event):
    """
    Converts an envent into a list of p4, usable by fastjet
    """
    a = np.zeros((len(event), 5))
    for i, p in enumerate(event):
        a[i, 3] = p[0]
        a[i, 0] = p[1]
        a[i, 1] = p[2]
        a[i, 2] = p[3]
        a[i, 4] = p[4]
    return(a)

def create_jet_dictionary(e,cluster=None,regression=False,R=1.0):
    """
    create the Jet dictionary stucture from fastjet
    """
    jet = {}
    if regression:
        ye=e[-1]
        e=e[0]
        jet["genpt"]   = ye
    t=cast(e)
    tree, content, mass, pt = cluster(t, jet_algorithm=1,R=R)[0]  # dump highest pt jet only
    
    jet["root_id"] = 0
    jet["tree"]    = tree    # tree structure, tree[i] constains [left son, right son] of subjet i
    jet["content"] = content # list of every p4 of every subjet used to create the full jet
    jet["mass"]    = mass
    jet["pt"]      = pt
    jet["energy"]  = content[0, 3]

    px = content[0, 0]
    py = content[0, 1]
    pz = content[0, 2]
    p = (content[0, 0:3] ** 2).sum() ** 0.5
    eta = 0.5 * (np.log(p + pz) - np.log(p - pz))
    phi = np.arctan2(py, px)
    
    jet["eta"]     = eta
    jet["phi"]     = phi
    
    return(jet)

def preprocess(jet, cluster, output="kt", regression=False,R_clustering=0.3):
    """
    preprocesses the data to make it usable by the recnn
    Preprocessing algorithm:
    1. j = the highest pt anti-kt jet (R=1)
    2. run kt (R=0.3) on the constituents c of j, resulting in subjets sj1, sj2, ..., sjN
    3. phi = sj1.phi(); for all c, do c.rotate_z(-phi)
    4. bv = sj1.boost_vector(); bv.set_perp(0); for all c, do c.boost(-bv)
    5. deltaz = sj1.pz - sj2.pz; deltay = sj1.py - sj2.py; alpha = -atan2(deltaz, deltay); for all c, do c.rotate_x(alpha)
    6. if sj3.pz < 0: for all c, do c.set_pz(-c.pz)
    7. finally recluster all transformed constituents c into a single jet
    """
    jet = copy.deepcopy(jet)
    constituents = jet["content"][jet["tree"][:, 0] == -1]
    if regression :
        genpt=jet["genpt"]

    ### run kt (R=0.3) on the constituents c of j, resulting in subjets sj1, sj2, ..., sjN ###
    subjets = cluster(constituents, R=R_clustering, jet_algorithm=0)
    oldeta=jet["eta"]
    oldpt=jet['pt']
    ### Rot phi ###
    # phi = sj1.phi()
    # for all c, do c.rotate_z(-phi)
    v = subjets[0][1][0]
    v = LorentzVector(v)

    phi = v.phi()
    
    for _, content, _, _ in subjets:
        for i in range(len(content)):
            v = LorentzVector(content[i][:4])
            v.rotate_z(-phi)
            content[i, 0] = v[0]
            content[i, 1] = v[1]
            content[i, 2] = v[2]
            content[i, 3] = v[3]

    ### boost ###
    # bv = sj1.boost_vector()
    # bv.set_perp(0)
    # for all c, do c.boost(-bv)
    v = subjets[0][1][0]
    v = LorentzVector(v)
    bv = v.boost_vector()
    bv.set_perp(0)
    for _, content, _, _ in subjets:
        for i in range(len(content)):
            v = LorentzVector(content[i][:4])
            v.boost(-bv)
            content[i, 0] = v[0]
            content[i, 1] = v[1]
            content[i, 2] = v[2]
            content[i, 3] = v[3]
    
    ### Rot alpha ###
    # deltaz = sj1.pz - sj2.pz
    # deltay = sj1.py - sj2.py
    # alpha = -atan2(deltaz, deltay)
    # for all c, do c.rotate_x(alpha)
    if len(subjets) >= 2:
        deltaz = subjets[0][1][0, 2] - subjets[1][1][0, 2]
        deltay = subjets[0][1][0, 1] - subjets[1][1][0, 1]
        alpha = -np.arctan2(deltaz, deltay)
        for _, content, _, _ in subjets:
            for i in range(len(content)):
                v = LorentzVector(content[i][:4])
                v.rotate_x(alpha)
                content[i, 0] = v[0]
                content[i, 1] = v[1]
                content[i, 2] = v[2]
                content[i, 3] = v[3]
    
    ### flip if necessary ###
    # if sj3.pz < 0: for all c, do c.set_pz(-c.pz)
    if len(subjets) >= 3 and subjets[2][1][0, 2] < 0:
        for _, content, _, _ in subjets:
            for i in range(len(content)):
                content[i, 2] *= -1.0
                
    ### finally recluster all transformed constituents c into a single jet ###
    constituents = []
    
    for tree, content, _, _ in subjets:
        constituents.append(content[tree[:, 0] == -1])
        
    constituents = np.vstack(constituents)

    if output == "anti-kt":
        subjets = cluster(constituents, R=100., jet_algorithm=1)
    elif output == "kt":
        subjets = cluster(constituents, R=100., jet_algorithm=0)
    elif output == "cambridge":
        subjets = cluster(constituents, R=100., jet_algorithm=2)
    else:
        raise
    
    jet["tree"]    = subjets[0][0]
    jet["content"] = subjets[0][1]
    v = LorentzVector(jet["content"][0])
    jet["phi"]     = v.phi()
    jet["eta"]     = v.eta()
    jet["energy"]  = v.E()
    jet["mass"]    = v.m()
    jet["pt"]      = v.pt()
    jet["root_id"] = 0
    jet['oldeta']  = oldeta
    jet['oldpt']   = oldpt
    if regression:
        jet["genpt"]   = genpt
    return(jet)

def load_from_pickle(filename, n_jets):
    """loads a pickle file"""
    jets = []
    fd = open(filename, "rb")

    for i in range(n_jets):
        jet = pickle.load(fd)
        jets.append(jet)

    fd.close()

    return jets



# Jet related

def _pt(v):
    """computes the pt of a LorentzVector"""
    pz = v[2]
    p = (v[0:3] ** 2).sum() ** 0.5
    eta = 0.5 * (np.log(p + pz) - np.log(p - pz))
    pt = p / np.cosh(eta)
    return pt


def permute_by_pt(jet, root_id=None):
    """Makes the hightest pt subjet the right subjet"""
    # ensure that the left sub-jet has always a larger pt than the right

    if root_id is None:
        root_id = jet["root_id"]

    if jet["tree"][root_id][0] != -1:
        left = jet["tree"][root_id][0]
        right = jet["tree"][root_id][1]

        pt_left = _pt(jet["content"][left])
        pt_right = _pt(jet["content"][right])

        if pt_left < pt_right:
            jet["tree"][root_id][0] = right
            jet["tree"][root_id][1] = left

        permute_by_pt(jet, left)
        permute_by_pt(jet, right)

    return jet


def rewrite_content(jet):
    """computes successive fusions and ids."""
    jet = copy.deepcopy(jet)

#    if jet["content"].shape[1] == 5:
#        pflow = jet["content"][:, 4].copy()
    content = np.zeros((len(jet["content"]),4+5))
    content[:,:4] = jet["content"][:,:-1]
    ids = np.abs(jet['content'][:,-1])
    content[:,4:] = np.array([np.isclose(ids,211.),np.isclose(ids,130.),np.isclose(ids,11.),np.isclose(ids,13.),np.isclose(ids,22.)],dtype=float).T
    tree = jet["tree"]

    def _rec(i):
        if tree[i, 0] == -1:
            pass
        else:
            _rec(tree[i, 0])
            _rec(tree[i, 1])
            c = content[tree[i, 0]] + content[tree[i, 1]]
            c[4:]=((content[tree[i, 0],3])*content[tree[i, 0],4:]+(content[tree[i, 1],3])*content[tree[i, 1],4:])/(content[tree[i, 0],3]*content[tree[i, 1],3])
            content[i] = c

    _rec(jet["root_id"])

#    if jet["content"].shape[1] == 5:
#        jet["content"][:, 4] = pflow

    return jet


def extract(jet, pflow=False):
    """per node feature extraction"""

    jet = copy.deepcopy(jet)

    s = jet["content"].shape

#    if not pflow:
    content = np.zeros((s[0], 7+5+2))
#    else:
#        # pflow value will be one-hot encoded
#        content = np.zeros((s[0], 7+4))

    for i in range(len(jet["content"])):
        px = jet["content"][i, 0]
        py = jet["content"][i, 1]
        pz = jet["content"][i, 2]

        p = (jet["content"][i, 0:3] ** 2).sum() ** 0.5
        eta = 0.5 * (np.log(p + pz) - np.log(p - pz))
        theta = 2 * np.arctan(np.exp(-eta))
        pt = p / np.cosh(eta)
        phi = np.arctan2(py, px)

        content[i, 0] = p
        content[i, 1] = eta if np.isfinite(eta) else 0.0
        content[i, 2] = phi
        content[i, 3] = jet["content"][i, 3]
        content[i, 4] = (jet["content"][i, 3] /
                         jet["content"][jet["root_id"], 3])
        content[i, 5] = pt if np.isfinite(pt) else 0.0
        content[i, 6] = theta if np.isfinite(theta) else 0.0
        content[i, 7] = jet["oldeta"]
        content[i, 8] = jet["oldpt"]
        content[i,9:] = jet["content"][i, -5:]
#        if pflow:
#            if jet["content"][i, 4] >= 0:
#                content[i, 7+int(jet["content"][i, 4])] = 1.0

    jet["content"] = content

    return jet


def randomize(jet):
    """build a random tree"""

    jet = copy.deepcopy(jet)

    leaves = np.where(jet["tree"][:, 0] == -1)[0]
    nodes = [n for n in leaves]
    content = [jet["content"][n] for n in nodes]
    nodes = range(len(nodes))
    tree = [[-1, -1] for n in nodes]
    pool = [n for n in nodes]
    next_id = len(nodes)

    while len(pool) >= 2:
        i = np.random.randint(len(pool))
        left = pool[i]
        del pool[i]
        j = np.random.randint(len(pool))
        right = pool[j]
        del pool[j]

        nodes.append(next_id)
        c = (content[left] + content[right])

#        if len(c) == 5:
#            c[-1] = -1

        content.append(c)
        tree.append([left, right])
        pool.append(next_id)
        next_id += 1

    jet["content"] = np.array(content)
    jet["tree"] = np.array(tree).astype(int)
    jet["root_id"] = len(jet["tree"]) - 1

    return jet


def sequentialize_by_pt(jet, reverse=False):
    """transform the tree into a sequence ordered by pt"""

    jet = copy.deepcopy(jet)

    leaves = np.where(jet["tree"][:, 0] == -1)[0]
    nodes = [n for n in leaves]
    content = [jet["content"][n] for n in nodes]
    nodes = [i for i in range(len(nodes))]
    tree = [[-1, -1] for n in nodes]
    pool = sorted([n for n in nodes],
                  key = lambda n: _pt(content[n]),
                  reverse = reverse)
    next_id = len(pool)

    while len(pool) >= 2:
        right = pool[-1]
        left = pool[-2]
        del pool[-1]
        del pool[-1]

        nodes.append(next_id)
        c = (content[left] + content[right])

        if len(c) == 5:
            c[-1] = -1

        content.append(c)
        tree.append([left, right])
        pool.append(next_id)
        next_id += 1

    jet["content"] = np.array(content)
    jet["tree"] = np.array(tree).astype(int)
    jet["root_id"] = len(jet["tree"]) - 1

    return jet
