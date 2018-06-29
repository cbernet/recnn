from trainbis import train

typelist=["subjet","particle"]

filelist=["anti-kt","kt","random","seqpt","seqpt_reversed","cambridge"]




for i in typelist:
	for j in filelist:
		print("\n\n\n\n")
		print(i,j)
		print("\n\n\n\n")
		nametrain="/data/conda/recnn/data/npyfiles/"+i+"_oriented_"+j+"_train.npy"
		namemodel="/data/conda/recnn/data/models/"+i+"_oriented_"+j+"_model.pickle"
		train(nametrain,namemodel,n_features=7,n_hidden=40,n_epochs=20,batch_size=64,step_size=0.0005,decay=0.9,regression=False)

