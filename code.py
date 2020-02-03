from nltk.corpus import brown
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
import nltk
import string
import random
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from spellchecker import SpellChecker

PUNC = [x for x in string.punctuation] + ["''", "``"]

##to get data from text file for reading 
# def get_data(data_path):
#     with open(data_path, 'r') as data_file:
#         data = data_file.read()
#         #return [x for x in data_file.readlines()]
#         return data

def download_data(target_path):
    brown_data = brown.sents()
    i = 0; n = 0
    label = []
    sentence = []
    while n < 40000:
        for sent in sent_tokenize(' '.join(brown_data[i])):
            updated_list = [x for x in word_tokenize(sent) if x not in PUNC]
            if n > 25000:
                random.shuffle(updated_list)
                label.append('incorrect')
            else:
                label.append('correct')
            sent = ' '.join(updated_list)
               
            if sent != '\n':
                sentence.append(sent)
                n += 1
            i += 1
    output = pd.DataFrame({'Sentence': sentence,'Label': label})
    output.to_csv('data_all.csv', index=False)

def parse_data(i,data):
    parsed_data = {}
    parsed_data[data[i].replace('\n', '')] = True
    return parsed_data

if __name__ == '__main__':
   
   ### shuffling of data
    # with open('data_all.csv', 'r') as r, open('data_all_shuffled.csv', 'w') as w:
    #     data = r.readlines()
    #     header, rows = data[0], data[1:]
    #     random.shuffle(rows)
    #     rows = '\n'.join([row.strip() for row in rows])
    #     w.write(header + rows)


    df = pd.read_csv('data_all_shuffled.csv',names=['Sentence','Label'])

    sentences = df['Sentence'].values.astype('U')
    y = df['Label'].values.astype('U')


    sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)
    
    vectorizer = CountVectorizer()
    vectorizer.fit(sentences_train)

    X_train = vectorizer.transform(sentences_train)

    ##to test custom values define the path first, and then sent_tokenize it so that it breaks in sentences as shown below with example
    # path_test = 'essay.txt'
    # sentences_get = get_data(path_test)
    # #print(sentences_get)
    # sentence_pred = sent_tokenize(sentences_get)
    # #print(len(sentence_pred))
    # for i in range(len(sentence_pred)):
    #     sentence_pred[i] = sentence_pred[i].replace(',',' ')
    # print(sentence_pred)
    # X_test  = vectorizer.transform(sentence_pred)

    
   # classifier = LogisticRegression()
    classifier_f = open("LR.pickle", "rb")
    classifier = pickle.load(classifier_f)
    classifier_f.close()
    classifier.fit(X_train, y_train)
    #score = classifier.score(X_test, y_test)
    #print("Accuracy:", score)

    ##checking custom value
    sentences_predict = 'Hello how are you? I am fine how about you.'
    sentences_predict2 = sent_tokenize(sentences_predict)
    #print(sentences_predict2)
    x = vectorizer.transform(sentences_predict2)
    pred = classifier.predict(x)
    #print(pred)
    count_unknown = list(pred).count('incorrect')
    print('The solution of above problem like this')
    print('1.No of sentence incorrect in the Essay.\n',count_unknown)
    count_known = len(pred) - count_unknown
    percent_accuracy_sent = (count_known/len(pred))*100

    ###saving pickle file
    # save_classifier = open("LR.pickle","wb")
    # pickle.dump(classifier, save_classifier)
    # save_classifier.close()
    
    ##put string in split_word_text to check whether all spellings are correct or not
    split_word_text = word_tokenize(sentences_predict)

    spell = SpellChecker()
    misspelled = split_word_text
    #if word is in dictionary
    word = spell.known(misspelled)
    #if word not in dictionary
    word2 = spell.unknown(misspelled)
    #print(word)
    #print(word2,len(word2))
    print('2.No of spelling mistake in the Essay.\n',len(word2))
    percent_accuracy_word = (len(word)/(len(word)+len(word2)))*100
    #print(percent_accuracy_word)
    print('Accuracy of Essay.', (percent_accuracy_sent+percent_accuracy_word)/2)
