# Multiple Instance Recognition

### Problem Statement
Given a dataset of images of several objects your job is to train a model on it such that it is able to retrieve all the instances of a given test object. More specifially the given dataset consists of images of several objects and multiple instances of each object. When tested on an image/object your model's task is to retrieve all instances of the image in the dataset. In other words your model must give a similarity ranking on all images in the dataset such that instances of the object being tested have the highest rankings. Please note that in training data each image contains only one object but in test data images will contain multiple objects. You will have to retrieve an image if any one of the object present in the query image matches with the object present in that image.

### Dataset
The training data can be downloaded [here](https://web.cse.iitk.ac.in/users/cs783/asm1/dataset_train.zip)
The sample test data is present in the repo [here](sample_test)  
The test data is present in the repo [here](test)

### Approach
We used **Text Retreival Approach to Image Instance Recognition**  to solve the problem using **SIFT** and **DELF** descriptor. The report can be found [here](report.pdf).

### Code
#### Training
The src folder contains model generation code and these expect training data to be present there. Model realated files are:
- dump_sift_features.py : for extracting local feature using sift.
- dump_kmeans.py : applying kmeans on local feature extracted above
- dump_sift_model : computing tf_idf vectors and dumping it
- dump_delf_feature : extracting delf local features

The final model is basically pickle of both sift and delf model pickle.

#### Prediction
Prediction (final aggregate to predict ranking for query image) assumes the names of all training images. This assumes model.pickle to be present in same folder.   To predict the final ranking:

```
python3 src/predict.py <addr of query image> <addr of output txt file>
```
