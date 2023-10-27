import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import seaborn as sn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer

nltk.download('omw-1.4')



def mapping(df):
    """Take in a dataframe with 5 classes and convert it to 3 classes."""
    df.loc[df.Sentiment <= 1, 'Sentiment'] = 0
    df.loc[df.Sentiment == 2, 'Sentiment'] = 1
    df.loc[df.Sentiment >= 3, 'Sentiment'] = 2
    
    return df

def pre_process(df, features):
    """Preprocess text in dataframe.""" 
    #lemmatizer = WordNetLemmatizer()
    #lancaster=LancasterStemmer()
    porter = PorterStemmer()
 
    # lower
    df['Phrase'] = df['Phrase'].str.lower()

    # remove punctuation
    df['Phrase'] = df['Phrase'].str.translate(str.maketrans('', '', "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~)")) 
   
    # remove words from stoplist
    if features == "features":  
        stop = stopwords.words('english')
        df['Phrase'] = df['Phrase'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
        #all_nouns = [word for synset in wn.all_synsets('n') for word in synset.lemma_names()]
        #df['Phrase'] = df['Phrase'].apply(lambda x: ' '.join([word for word in x.split() if word not in (all_nouns)]))
    
    # tokenise
    df['Phrase'] = df['Phrase'].str.split()

    # stem
    for review in df['Phrase']:
        for w in review:
            w = porter.stem(w)
        
    return df    


def likelihood(df, number_classes):
    """Return likelikood of every term and vocabulary size."""  
    sent_dict = {}
    words_set = set()

    # calculate vocabulary
    for ind in df.index:              
        for w in df['Phrase'][ind]:
            words_set.add(w)

    # number of words in sentiment (3 classes)
    if number_classes == 3:
        number_words = [0] * 3 

    # number of words in sentiment (5 classes)  
    if number_classes == 5:
        number_words = [0] * 5   
    
    # term count dictionary {sentiment : { word : word_count}}
    for ind in df.index:     
        if df['Sentiment'][ind] not in sent_dict:
            sent_dict[df['Sentiment'][ind]] = {}
        for w in df['Phrase'][ind]:    
            number_words[df['Sentiment'][ind]] += 1
            if w not in sent_dict[df['Sentiment'][ind]]:
                sent_dict[df['Sentiment'][ind]][w] = 0
            sent_dict[df['Sentiment'][ind]][w] += 1

    # calculate likelihood of every term with Laplace smoothing
    for s in sent_dict:
        for w in sent_dict[s]:
            sent_dict[s][w] = (sent_dict[s][w] + 1) / (number_words[s] + len(words_set))
    
    return sent_dict, len(words_set), number_words


def prior_prob(df, number_classes):
    """Calculate prior probability of each class."""
    prior = {}

    # store prior probability in dictionary {sentiment : probability}
    if number_classes == 5:
        for i in range(5):
            prior[i] = df[df.Sentiment == i].shape[0] / df.shape[0]

    # maps 5-value sentiment scale to a 3-value sentiment scale 
    else: 
        for i in range(3):
            prior[i] = df[df.Sentiment == i].shape[0] / df.shape[0]

    return prior

def evaluation(text, prior, likelihood, number_classes, v, words):
    """Take a text and assign a class to it."""
    if number_classes == 3:
        size = 3     
    elif number_classes == 5:
        size = 5
    
    # indexes represent sentiment class
    results = [0] * size  

    for s in range(size):
        results[s] = prior[s]
        for w in text:
            if w in likelihood[s]:
                results[s] *= likelihood[s][w] # posterior probability
            else: 
                results[s] *= 1 / (v + words[s]) # posterior probability if word not in vocabulary
    
    #Bayes classifier
    return results.index(max(results)) 


def confusion_matrix(confusion, number_classes, features):
    """Display confusion matrix with heat map."""
    s = sn.heatmap(confusion, annot=True, fmt='g')
    s.set(xlabel='Predicted', ylabel='Actual', title='{} Sentiment Confusion Matrix - '.format(number_classes) + features)
    plt.show()


def save_results_dev(devf, prior_dict, likelihood_dict, number_classes, v, user, output, words):
    """Classify all phrases in dev dataframe and return confusion matrix."""
    text_id = []
    text_sentiment = []
    confusion = np.zeros((number_classes, number_classes))
    for ind in devf.index:  
        #list of all phrase ids
        text_id.append(devf['SentenceId'][ind]) 
        # list of all phrases
        txt = devf['Phrase'][ind] 
        # classify phrase
        sentiment = evaluation(txt, prior_dict, likelihood_dict, number_classes, v, words)
        # add results to confusion matrix
        confusion[devf['Sentiment'][ind]][sentiment] += 1
        # list of results
        text_sentiment.append(sentiment)
    
    # create dataframe with results
    results_dic = {'SentenceID':text_id, 'Sentiment':text_sentiment}
    results_df = pd.DataFrame(results_dic)

    # create output file with results
    if output:
        results_df.to_csv(f'dev_predictions_{number_classes}classes_{user}.tsv', sep="\t", index=False)

    # return conusion matrix
    return confusion

def save_results_test(testf, prior_dict, likelihood_dict, number_classes, v, user, output, words):
    """Classify all phrases in test dataframe."""
    text_id = []
    text_sentiment = []

    for ind in testf.index:  
        #list of all phrase ids
        text_id.append(testf['SentenceId'][ind]) 
        # list of all phrases
        txt = testf['Phrase'][ind] 
        # classify phrase
        sentiment = evaluation(txt, prior_dict, likelihood_dict, number_classes, v, words)
        # list of results
        text_sentiment.append(sentiment)

    # create dataframe with results
    results_dic = {'SentenceID':text_id, 'Sentiment':text_sentiment}
    results_df = pd.DataFrame(results_dic)

    # create output file with results
    if output:
        results_df.to_csv(f'test_predictions_{number_classes}classes_{user}.tsv', sep="\t", index=False)
        

def calculate_f1(confusion, number_classes):
    """Calculate macro-F1 score."""
    f1 = 0
    # calculate F1 score for each class from the confusion matrix
    for i in range(number_classes):
        precision = confusion[i][i] / confusion[:, i].sum()
        recall = confusion[i][i] / confusion[i, :].sum()
        f1 += 2*((precision*recall)/(precision+recall))

    # calculate macro-F1 score
    f1 = f1/number_classes
    return f1
