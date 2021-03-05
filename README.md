# mle-capstone
Repository for the Machine Learning Engineer Capstone Project

This project is based in the Natural Language Processing with Disaster Tweets Competition from Kaggle (https://www.kaggle.com/c/nlp-getting-started/overview). The original data set may be downloaded from this site.

## Required Libraries
Data managing libraries: Common libraries used to store, display and manage data
- Pandas
- Numpy
- SKLearn
- Matplotlib
- OS
- Pickle

Data processing libraries: libraries used to pre-process our data and manage information (ex: word processing, vocabulary creation, etc.)
- NLTK (Natural Language Toolkit)
- Collections
- re
- BeautifulSoup

Model implementation libraries: Libraries used to implement and evaluate our different models
- Sagemaker
- SKLearn
- PyTorch

## Project Motivation
Our project objective is to detect if tweets are reporting a disaster or not based in the words contained in them. This model could be useful in cases were there is no official information or it is too slow to alert nearby population.

## Files Description
- ./Project Report.pdf: Report of the project submitted
- ./proposal.pdf: approved project proposal
- ./Capstone project.ipynb: Jupyter notebook containing all of our project implementations and results. It is the main file and it can be directly executed in a Jupyter notebook environment.
- ./cache/tweet_data/preprocessed_data.pkl: cache file storing our pre-processed words to avoid reprocessing in following executions. May be deleted for your own implementation.
- ./data/train.csv: original train data from the Kaggle Competition
- ./data/pytorch/train.csv: file containing our processed training set
- ./data/pytorch/validation.csv: file containing our processed validation set
- ./data/pytorch/word_dict.pkl: file containing the created dictionary used to transform words to integer values
- ./train/model.py: definition of our neural network model
- ./train/train.py: model initialization, default parameter specification and training functions
- ./train/requirements.txt: libraries required for our model implementations.

## Results Summary
Our proposed RNN model showed to be superior under the F1-score metric than a Linear Ridge Regression Classifier. Moreover, hyperparameter tuning of the epoch values showed to improve the results of an initial implementation with a default value of epochs.

## Acknowledments
- Kaggle site for the competition description and datasets: https://www.kaggle.com/c/nlp-getting-started/overview
- Public tutorial for Natural Language Processing by Phil Culliton: https://www.kaggle.com/philculliton/nlp-getting-started-tutorial
- Public submission by Shahules786: https://www.kaggle.com/shahules/basic-eda-cleaning-and-glove
