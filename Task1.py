# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 22:19:47 2017

@author: Abdul Rehman
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 13:55:42 2017

@author: Abdul Rehman
"""

from nltk.corpus import udhr
import numpy as np
import matplotlib.pyplot as plt

'''Function to convert a string of text into ngram of n. Simply replace '\n' (newline) with '' (zero space). Then pick chunks 
from string of size n.'''
#"gram" is ngrams of "txt" based on "n"
def ngram(txt,n):
    text=txt.replace('\n','')
    chunk=int(len(text)/n)
    gram=[]
    x=0
    for x in range(chunk):
        gram.append(text[n*x:n*(x+1)])
    if(len(text)%n!=0):
        gram.append(text[n*(x+1):])
    return gram

    
'''Function to convert a text into a combination of multi ngrams. For example we want to use the combination of 1-gram and 
2-gram for a text T. This function will call 'ngram' function twice for 1-gram and 2-gram respectively and then concatenate
the result into one vector. Thus combining all combinations of n-grams for a text.'''
#"freqCorpus" is mult-ngrams of "multiText" on "n". This "n" is list of multiple natural numbers.
def multiNgram(multiText,n):
    freqCorpus=list()
    if(len(multiText[0])!=1):
        for x in multiText:
            freq=[]
            for y in n:
                freq=freq+ngram(x,y)
            freqCorpus.append(list(set(freq)))
    else:
        freq=[]
        for y in n:
            freq=freq+ngram(multiText,y)
        freqCorpus.append(freq)
    return freqCorpus

'''Function to calculate the cosine distance between any two vectors a and b.'''    
def cosine(a,b):
    ab=sum(a*b)
    sqrta2=np.sqrt(sum(a*a))
    sqrtb2=np.sqrt(sum(b*b))
    return ab/(sqrta2*sqrtb2)
    
'''Function to calculate the consine distance multiple vectors one by one and the store the result in
a list'''
#"cosVal" consists of list of cosine distance for each vector with parent.
def multiCos(vec):
    cosVal=[]
    for x in vec:
        cosVal.append(cosine(x,np.ones(len(x))))
    return cosVal

'''Function to train the model. Input consists list of langugages and n-grams combinations for which we want to train our
model. Output of this function is multidimensional vector array, which has a multi n-grams for each language'''
#Function returns the multi n-grmas of each language which is to be tested..
def train(lang,n):
    langCorpus=[]
    for x in lang:
        langCorpus.append(udhr.raw(x+'-Latin1'))
    return multiNgram(langCorpus,n)
    
'''Function to test the model. Input consists of multi n-grams vector for each language for which we want to be tested. Other
inputs are test sentence and n-grams combinations we want to use in the test.'''
#Function returns the closes neighbor or best index of the language against the given "test"
def test(ngramCorpus,test,n):
    testnGram=multiNgram(test,n)[0]
    testVec=[]
    for x,y in zip(ngramCorpus,range(len(ngramCorpus))):
        testVec.append(np.zeros(len(x)))
        for z in testnGram:
            if(z in x):
                ind=x.index(z)
                testVec[y][ind]=testVec[y][ind]+1               
    return np.argmax(multiCos(testVec))
    
#Basic input of the whole model. "textTxt" is the string we want to test, "n" is the list of n-grams combinations we want
#to have and "lang" is the list of languages we want to test again. Note that these langauges are named according to
#NLTK corpus naming conventions. If you want to add more langauge, please use NLTK corpus naming.
def main():

    testTxt='Du bist ein mann. Er ist auch ein man'
    n=[1,2,3,4,5,6,7,8,9,10]
    lang=['English', 'German_Deutsch',
         'Danish_Dansk','Dutch_Nederlands','Spanish','Swedish_Svenska',
          'French_Francais','Italian_Italiano','Norwegian','Portuguese_Portugues']

    model=train(lang,n)
    ind=test(model,testTxt,n)
    print(lang[ind])
    
    '''------------Folowing Code is only for model testing. Testing is done on movie subtitles of differentmovies------------------'''
    testingTheModel(n,lang,model) 
    
    
'''Function to setup the subtitle file names and path. Also for controlling the whole test procedure''' 
def testingTheModel(n,lang,model):        
    testLang=['danish','dutch','english','french','german','italian','norwegian',
              'portuguese','spanish','swedish'] 
              
    actualTestLang=['Danish_Dansk','Dutch_Nederlands','English','French_Francais',
                'German_Deutsch','Italian_Italiano','Norwegian','Portuguese_Portugues',
                'Spanish','Swedish_Svenska']
                
    dictionary={'Danish_Dansk':'Danish','Dutch_Nederlands':'Dutch',
                'English':'English','French_Francais':'French',
                'German_Deutsch':'German','Italian_Italiano':'Italian',
                'Norwegian':'Norwegian','Portuguese_Portugues':'Portuguese',
                'Spanish':'Spanish','Swedish_Svenska':'Swedish'}

    place=["C:/Users\\Abdul Rehman\\Downloads\\testData\\2\\","C:/Users\\Abdul Rehman\\Downloads\\testData\\1\\"]
    #Following function does whole testing
    startModelTest(testLang,actualTestLang,place,n,model,lang,dictionary)


''' Function to read .srt files, parse text and concatenate language wise'''    
def subTitle(testLang,place):
    import pysrt as srt
    test=[]
    for x,y in zip(testLang,range(len(testLang))):
        obj=srt.open(place+x+".srt",encoding='iso-8859-1')
        test.append([])
        for z in obj:
            test[y].append(z.text)
    return test

'''Function to test the each subtitle. "sub" input contains subtitles in different languages.
This function will return the predicted langauge for each subtitle'''
def modelTest(test,model,n,sub,lang):
    language=[]
    for x in sub:
        ind=test(model,x,n)
        language.append(lang[ind])
    return language

'''Function to test multiple combinations of n-gram. It will call "modelTest" function for
different combinations of n-grams and then store the result. This function is also 
responsible for calculating the prediction accuracy and plotting the acquried observations '''    
def startModelTest(testLang,actualTestLang,place,n,model,lang,dic):
    accuracy=[]
    wrong=[]
    plt.figure(1)
    for a,b in zip(place,range(len(place))):
        sub=subTitle(testLang,a)
        accuracy.append([])
        for c in range(1,len(n)):
            language=modelTest(test,model,n[:c],sub,lang)
            count=0
            for x,y in zip(language,actualTestLang):
                if(x==y):
                    count=count+1
                else:
                    wrong.append(dic[x]+'<==>'+dic[y])
            accuracy[b].append(count/len(testLang))

            
    plt.plot(range(1,len(accuracy[b])+1),accuracy[b])
    frequency,wrongPairs=confusedPairs(wrong)
    plt.xlabel('n-grams combinations from 1 till n')
    plt.ylabel('Prediction Accuracy')
    plt.title('combinations of n-grams vs prediction accuracy')
    plt.show()
    plt.figure(figsize=(5,12))
    y_axis=np.arange(len(frequency),dtype=int)
    plt.barh(y_axis, frequency, align='center')
    plt.yticks(y_axis, wrongPairs)
    plt.xlabel('Frequency of predicting wrong')
    plt.ylabel('Predicted Langauge <==> Actual Language')
    plt.title('Wrong Prediction Pair Frequency')
#    print(frequency,wrongPairs)
#    print(accuracy)
    plt.show()
    
'''This function returns the frequency for each wrong prediction of a language'''
def confusedPairs(wrong):
    new=list(set(wrong))
    freq=[0]*len(new)
    for x in wrong:
        if(x in new):
            ind=new.index(x)
            freq[ind]=freq[ind]+1
    return freq,new

    
if(__name__=="__main__"):
    main()