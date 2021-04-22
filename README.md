# AI-393-Spring-2021

### PROJECT MEMBERS ###
StdID | Name
------------ | -------------
**8886** | **Afraz Ul Huq** 
61705 | Roshaan Mehmood
63361 | Naufil Bin Majid

# Abstract #
In this project, we have used the following technologies:

## 1. Training Data ##
We used train.csv file to train data. For reading our training data. Pandas is mostly used to help with data organization and visualization. Since you can put data into dataframes and datasets. We used pandas library tool here to read train.csv file.


## Cross Validation ##
**Cross-validation** is a resampling procedure used to evaluate machine learning models on a limited data sample. The procedure has a single parameter called k that refers to the number of groups that a given data sample is to be split into. As such, the procedure is often called k-fold cross-validation.

For cross validation we have to split data and to acquire this we use **train_test_split** method. For splitting data we have to give a range of specific value.

## Naive Bayes ##
**Naive Bayes** is a probabilistic algorithm that’s typically used for classification problems. Naive Bayes is simple, intuitive, and yet performs surprisingly well in many cases. For example, spam filters Email app uses are built on Naive Bayes. In this article, I’ll explain the rationales behind Naive Bayes and build a spam filter in Python. (For simplicity, I’ll focus on binary classification problems).


## Convolution ##
**convolution** the most common type of convolution that is used is the 2D convolution layer, and is usually abbreviated as conv2DIt has a really small kernel, essentially a window of pixel values, that slides along those two dimensions. The rgb channel is not handled as a small window of depth, but rather, is obtained from beginning to end, first channel to last. That is, even a convolution with a small spatial window of 1x1, which takes a single pixel spatially in the width/height dimensions, would still take all 3 RGB channels.we use it to produce the better images on it to recognize the digit in more better.



## KNN ##
**kNN** KNN algorithm is one of the simplest classification algorithm and it is one of the most used learning algorithms. ... KNN is a non-parametric, lazy learning algorithm. Its purpose is to use a database in which the data points are separated into several classes to predict the classification of a new sample point.The KNN algorithm can compete with the most accurate models because it makes highly accurate predictions. we use it to produce the best accuracy and give us the best result regarding accuracy.

## Some info about parameters ## 

X_train.shape
y_train.shape
X_test.shape
mnb = MultinomialNB()


**param_grid = {'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 10.0],**
              **'fit_prior': [True,False]}**

**mnb_GS = GridSearchCV(mnb, param_grid, cv=5, verbose=2, n_jobs=1)**

**tiny_model.fit(X_train, y_train, epochs=5, batch_size=32)**

**test_loss, test_acc = tiny_model.evaluate(X_test,  y_test, verbose=2)**

For the first algorithm, I am choosing to use the **Naive-Bayes classifier**. Assuming that all predictors have an equal effect on the outcome and one predictor variable does not affect the presence of another, the algorithm seems like a good choice.

Since our case is a classification problem with multple possible outcomes (digits 0 to 9), the **Multinomial** type of **Naive-Bayes Classifier** would be a better fit compared to the other 2 types. The Bernoulli type recommends a binary outcome whereas the Gaussian type requires predictors to be continous. 

**train_images = np.reshape(X_train, (len(X_train),5,5))**
**test_images = np.reshape(X_test, (len(X_test),5,5))**

at this point we try to filter out and reshape the images in 5x5 which give us the better result and more accuracy to point.

**train,test=train_test_split(train,test_size=0.25)**

at this parameter we just split our data to train it seperately which guide us to **cross-validation** to split data.

##Short Description About Techniques##
we using the 5 techniques to improve the training of our model and make it produce the more better results further **(svm,knn,neural,linear regression,convo)**
by usig these 5 techniques our understanding is much more improve rearding how can 

This repository contains assignments and project submitted to AI course offered in Spring 2021 at KIET.

