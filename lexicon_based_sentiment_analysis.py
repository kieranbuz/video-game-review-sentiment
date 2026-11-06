# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 13:53:45 2020

@author: Kieran Burrell
"""
#import the pandas library to read the dataset from a CSV file
import pandas as pd
#import the numpy library to convert data into arrays
import numpy as np
#import string module to be used to remove punctuation from entries
import string
#import time module to be used for timing the AI algorithms
import time
#import the classifier
from sklearn.neural_network import MLPClassifier
#import classifier evaluation metrics
from sklearn.metrics import classification_report

#LOADING DATA FILES
print()
print('Loading Data Files...')
#load training data
#open training data file
training_file = open('reviews_Video_Games_training.raw.tsv')
#read all file lines into training_dataset
training_dataset = training_file.readlines()
#declare training instance list
training_instances = []
#declare training label
training_labels = []
#for each line in the training dataset
for line in training_dataset:
    #split the line where there is a tab
    entry = line.split('\t')
    #instance is first part of the entry split
    instance = entry[0]
    #label is the second part of the entry split
    label = int(entry[1])
    #append new instance to training_instances 
    training_instances.append(instance)
    #append new label to training_labels
    training_labels.append(label)
    
#load test data
testing_file = open('reviews_Video_Games_test.raw.tsv')
#read all file lines into testing_dataset
testing_dataset = testing_file.readlines()
#declare testing instance list
testing_instances = []
#declare testing label
testing_labels = []
#for each line in the testing dataset
for line in testing_dataset:
    #split the line where there is a tab
    entry = line.split('\t')
    #instance is first part of the entry split
    instance = entry[0]
    #label is the second part of the entry split
    label = int(entry[1])
    #append new instance to testing_instances 
    testing_instances.append(instance)
    #append new label to testing_labels
    testing_labels.append(label)

#load sentiment lexicon
sentiment_lexicon = pd.read_csv('Games_senti_lexicon.tsv', delimiter='\\t', header=None, engine='python')
#get words in sentiment lexicon\n",
sentiment_words = list(sentiment_lexicon[0])
#get sentiment scores
sentiment_scores = list(sentiment_lexicon[1])
print('Files have been loaded succesfully')

#CALCUALTING POLARITY
print()
print('Calculating sentiment polarity...')
#declare training instance polarity
training_instances_polarity = []
#declare testing instance polarity
testing_instances_polarity = []
#declare temporary word values 
words_values = []

#for every training instance
for entry in training_instances:
    #remove punctuation from entry
    entry = entry.translate(str.maketrans('', '', string.punctuation))
    #split entry into words
    words = entry.split(" ")
    #for every word 
    for word in words:
        #if word is in sentiment_words
        if word in sentiment_words:
            #get index of word in sentiment_words
            index = sentiment_words.index(word)
            #append word value with sentiment score based on index
            words_values.append(float(sentiment_scores[index]))
    #multiplies words value by 100
    multiplied_list = [element * 100 for element in words_values]     
    #append polarity with the sum of the multiplied list
    training_instances_polarity.append(sum(multiplied_list))
    #clear words_values 
    words_values = []
    
#for every testing instance
for entry in testing_instances:
    #remove punctuation from entry
    entry = entry.translate(str.maketrans('', '', string.punctuation))
    #split entry into words
    words = entry.split(" ")
    #for every word 
    for word in words:
        #if word is in sentiment_words
        if word in sentiment_words:
            #get index of word in sentiment_words
            index = sentiment_words.index(word)
            #append word value with sentiment score based on index
            words_values.append(float(sentiment_scores[index]))
    #multiplies words value by 100
    multiplied_list = [element * 100 for element in words_values]     
    #append polarity with the sum of the multiplied list
    testing_instances_polarity.append(sum(multiplied_list))
    #clear words_values 
    words_values = []

print('Sentiment polarity calculated successfully')

#makes training instance polarity list into array and reshapes array for classifier
training_instances_polarity = np.array(training_instances_polarity).reshape(-1,1)
#makes training labels list into array and reshapes array for classifier
training_labels = np.array(training_labels).reshape(-1,1)
#makes testing instance polarity list into array and reshapes array for classifier
testing_instances_polarity = np.array(testing_instances_polarity).reshape(-1,1)
#makes testing labels list into array and reshapes array for classifier
testing_labels = np.array(testing_labels).reshape(-1,1)

#TRAINING
print()
print('Training the algorithm...')
#records start time
timerStart = time.perf_counter()
#defines the classifier
classifier = MLPClassifier(random_state=1000, max_iter=10)
#trains the classifier
classifier.fit(training_instances_polarity, training_labels.ravel())
print('Training finished successfully')

#TESTING
print()
print('Testing the algorithm...')
#gets predicted labels of test dataset
predicted_test_labels = classifier.predict(testing_instances_polarity)
#records finish time
timerFinish = time.perf_counter()
print('Testing finished successfully')

#EVALUATION
print()
print('Evaluating the algorithm...')
#prints evaluation metrics based on the actual and the predicted test labels
print(classification_report(testing_labels,predicted_test_labels,digits=4))
#prints time taken
print("Time taken: " + str(round(timerFinish-timerStart,4)) + " Secs")      
 