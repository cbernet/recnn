import uproot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

fname = 'data/rootfiles/RawSignal_21Sept.root'
nevts = 10000


rfile = uproot.rootio.open(fname)
tree = rfile.get('tree')

print tree.keys()

def load(branches):
    event = dict()
    for branch in branches:
        pass

# ptcs = tree['ptcs'].array(entrystop=nevts)
# deltars = tree['dRs'].array(entrystop=nevts)

event = tree.arrays(['ptcs','dRs', 'Tau', 'Jet', 'GenTau', 'standardID'], entrystop=nevts)

# select signal entries
event_mgt = dict()
# cut on deltar between pf jet and gen tau:
matched_to_gentau = event['dRs'][:,0] < 0.1
for key, branch in event.iteritems():
    event_mgt[key] = branch[matched_to_gentau]
    print key, event_mgt[key].shape

# plot jet pt
jets = event_mgt['Jet']
print 'plot jet pt'
jets = np.squeeze(jets, axis=1)
print jets.shape
df = pd.DataFrame(jets, columns=['px','py','pz','e','pdgid','charge','ndau','hadflav','partflav'])
print df
df['pt'] = np.sqrt(df['px']**2 + df['py']**2)
plt.hist(df['pt'], bins=100)

# plot ptc pt
ptcs = event_mgt['ptcs']

# ptcs = ptcs.reshape(-1, 200*8)
# but the following does not work... 
# df = pd.DataFrame(ptcs, columns=['px','py','pz','e','pdgid','charge','dxy','dz']*200)
# df['pt'] = np.sqrt(df['px']**2 + df['py']**2)
# plt.hist(df['pt'], bins=100)


ptcs = ptcs.reshape(9930*200, 8)
df = pd.DataFrame(ptcs, columns=['px','py','pz','e','pdgid','charge','dxy','dz'])
df['pt'] = np.sqrt(df['px']**2 + df['py']**2)   
ptcs_pose =  df.loc[df['e']>0]
plt.hist(ptcs_pose['pt'], bins=100, range=(0,10))





