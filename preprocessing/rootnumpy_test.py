from ROOT import TFile

rfile = TFile('data/rootfiles/RawSignal_21Sept.root')
tree = rfile.Get('tree')

from root_numpy import tree2array
array = tree2array(tree,'ptcs')
