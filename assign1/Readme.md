Following are the files and their uses:
	- dump_sift_features.py : for extracting local feature using sift.
	- dump_kmeans.py : applying kmeans on local feature extracted above
	- dump_sift_model : computing tf_idf vectors and dumping it
	- dump_delf_feature : extracting delf local features
	- predict.py : Final aggregate to predict ranking for query image.

We assume all these files to be present in same directory as the train folder given to us. predict.py uses a pickle model named model.pickle with hash ee5a08eb70cb958b686b48df81bd3687

To predict the final ranking:
	python3 predict.py <addr of query image> <addr of output txt file>

The outputs of given sample test images are present in outputs directory

Requirements:
	- numpy
	- tensorflow
	- tensorflow-hub
	- scipy
	- skimage
	- sklearn
	- cv2
	- matplotlib
	- pickle
	

