# -*- coding: utf-8 -*-
"""
NB sentiment analyser. 

Start code.
"""
import argparse, csv, re
import pandas as pd
import Classsifier as Classifier


"""
IMPORTANT, modify this part with your details
"""
USER_ID = "aca19rmn" #your unique student ID, i.e. the IDs starting with "acp", "mm" etc that you use to login into MUSE 

def parse_args():
    parser=argparse.ArgumentParser(description="A Naive Bayes Sentiment Analyser for the Rotten Tomatoes Movie Reviews dataset")
    parser.add_argument("training")
    parser.add_argument("dev")
    parser.add_argument("test")
    parser.add_argument("-classes", type=int)
    parser.add_argument('-features', type=str, default="all_words", choices=["all_words", "features"])
    parser.add_argument('-output_files', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-confusion_matrix', action=argparse.BooleanOptionalAction, default=False)
    args=parser.parse_args()
    return args




def main():

    inputs=parse_args()
    
    #input files
    training = inputs.training
    dev = inputs.dev
    test = inputs.test
    
    #number of classes
    number_classes = inputs.classes
    
    #accepted values "features" to use your features or "all_words" to use all words (default = all_words)
    features = inputs.features
    
    #whether to save the predictions for dev and test on files (default = no files)
    output_files = inputs.output_files
     
    
    #whether to print confusion matrix (default = no confusion matrix)
    confusion_matrix = inputs.confusion_matrix
    
    """
    ADD YOUR CODE HERE
    Create functions and classes, using the best practices of Software Engineering
    """
    # read input files as Panda DataFrame
    df = pd.read_csv(training, sep='\t', header=0)
    devf = pd.read_csv(dev, sep='\t', header=0)
    testf = pd.read_csv(test, sep='\t', header=0)

    # convert from 5 classes to 3 classes
    if number_classes == 3:
        df = Classifier.mapping(df)
        devf = Classifier.mapping(devf)
    
    # apply preprocessing
    df = Classifier.pre_process(df, features)
    devf = Classifier.pre_process(devf, features)
    testf = Classifier.pre_process(testf, features)

    # calculate the likelihood of every term and the vocabulary
    likelihood_dict, v, words = Classifier.likelihood(df, number_classes)

    # calculate the prior probability of every class
    prior_dict = Classifier.prior_prob(df, number_classes)

    # make predictions for dev and store as a confusion matrix 
    confusion_arr = Classifier.save_results_dev(devf, prior_dict, likelihood_dict, number_classes, v, USER_ID, output_files, words)

    # make predications for test
    Classifier.save_results_test(testf, prior_dict, likelihood_dict, number_classes, v, USER_ID, output_files, words)

    # show the confusion matrix
    if confusion_matrix:
        Classifier.confusion_matrix(confusion_arr, number_classes, features)
    
    # calculate f1 score
    f1_score = Classifier.calculate_f1(confusion_arr, number_classes)
    

    """
    IMPORTANT: your code should return the lines below. 
    However, make sure you are also implementing a function to save the class predictions on dev and test sets as specified in the assignment handout
    """
    #print("Student\tNumber of classes\tFeatures\tmacro-F1(dev)\tAccuracy(dev)")
    print("%s\t%d\t%s\t%f" % (USER_ID, number_classes, output_files, f1_score))



if __name__ == "__main__":
    main()