from train import train

typelist=["subjet","particle"]

filelist=["anti-kt","kt","random","seqpt","seqpt_reversed","cambridge"]




for i in typelist:
	for j in filelist:
		print("\n\n\n\n")
		print(i,j)
		print("\n\n\n\n")
		nametrain="/data/conda/recnn/data/npyfilesregression/"+i+"_oriented_"+j+"_with_id_train.npy"
		namemodel="/data/conda/recnn/data/modelsregression/"+i+"_oriented_"+j+"_with_id_model.pickle"
		train(nametrain,namemodel,n_features=12,n_hidden=40,n_epochs=10,batch_size=64,step_size=0.0005,decay=0.9,statlimit=100000,regression=True,verbose=True)

