# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 13:53:45 2020

@author: Kieran Burrell
"""
#import the pandas library to read the dataset from a CSV file
import pandas as pd
#import time module to be used for timing the AI algorithms
import time
#import the classifier
from sklearn.naive_bayes import CategoricalNB
#import classifier evaluation metrics
from sklearn.metrics import classification_report

#GETTING DATA FROM CSV FILE
def load_data_from_csv(input_csv):
    #uses the read_csv function to load the csv file into a dataframe
    #the first line of the csv (header=0) contains headings
    df = pd.read_csv(input_csv, header=0)
    #stores CSV headings into a list
    csv_headings = list(df.columns.values)
    #the first n-1 iteams of the csv_headings are the feature names
    feature_names = csv_headings[:len(csv_headings) - 1]
    #gets the numeric data of the CSV file and store it in a numpy array
    df = df._get_numeric_data()
    numpy_array = df.to_numpy()
    #gets the number of rows and columns from numpy array
    number_of_rows, number_of_columns = numpy_array.shape
    #n-1 columns of the numpy array corresponds to the instances
    instances = numpy_array[:, 0:number_of_columns - 1]
    #the last column of the numpy array corresponds to the class labels
    labels = []
    for label in numpy_array[:, number_of_columns - 1:number_of_columns].tolist():
        labels.append(label[0])
    return feature_names, instances, labels   

#LOADING DATA FILES
print()
print('Loading Data Files...')
#file path for the training dataset
input_training_csv = 'reviews_Video_Games_training.csv'
#file path for the testing dataset
input_testing_csv = 'reviews_Video_Games_test.csv'
#reading the training dataset
training_feature_names, training_instances, training_labels = load_data_from_csv(input_csv=input_training_csv)
#reading the testing dataset
testing_feature_names, testing_instances, testing_labels = load_data_from_csv(input_csv=input_testing_csv)
print('Files have been loaded succesfully')

#TRAINING
print()
print('Training the algorithm...')
#records start time
timerStart = time.perf_counter()
#defines the classifier
classifier = CategoricalNB()
#trains the classifier
classifier.fit(training_instances, training_labels)
print('Training finished successfully')

#TESTING
print()
print('Testing the algorithm...')
#gets predicted labels of test dataset
predicted_test_labels = classifier.predict(testing_instances)
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
 