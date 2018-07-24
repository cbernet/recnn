from train import train

typelist=["0.3","1e-05"]

filelist=["anti-kt","kt","random","seqpt","seqpt_reversed","cambridge"]




for i in typelist:
	for j in filelist:
		print("\n\n\n\n")
		print(i,j)
		print("\n\n\n\n")
		nametrain="/data/conda/recnn/data/npyfilesregression/Background_JEC_train_ID_preprocessed_R="+i+"_"+j+".npy"
		namemodel="/data/conda/recnn/data/modelsregression/Model_R="+i+"_"+j+".pickle"
		train(nametrain,namemodel,n_features=14,n_hidden=40,n_epochs=5,batch_size=64,step_size=0.001,decay=0.5,regression=True,verbose=True)

