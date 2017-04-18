# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 13:55:42 2017

@author: Abdul Rehman
"""

from nltk.corpus import udhr
import numpy as np

def ngram(txt,n):
    text=txt.replace('\n','')
    chunk=int(len(text)/n)
    gram=[]
    for x in range(chunk):
        gram.append(text[n*x:n*(x+1)])
    if(len(text)%n!=0):
        gram.append(text[n*(x+1):])
    return gram
    
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
    
def cosine(a,b):
    ab=sum(a*b)
    sqrta2=np.sqrt(sum(a*a))
    sqrtb2=np.sqrt(sum(b*b))
    return ab/(sqrta2*sqrtb2)
    
def testFreqVec(ngramCorpus,test,n):
    testnGram=multiNgram(test,n)[0]
    testVec=[]
    for x,y in zip(ngramCorpus,range(len(ngramCorpus))):
        testVec.append(np.zeros(len(x)))
        for z in testnGram:
            if(z in x):
                ind=x.index(z)
                testVec[y][ind]=testVec[y][ind]+1
    return testVec
    
def multiCos(vec):
    cosVal=[]
    for x in vec:
        cosVal.append(cosine(x,np.ones(len(x))))
    return cosVal

def main():
    n=[1,2,3,4]
    lang=['English', 'German_Deutsch',
         'Greenlandic_Inuktikut', 'Hungarian_Magyar','Danish_Dansk','Dutch_Nederlands','Finnish_Suomi',
          'French_Francais','Italian_Italiano','Norwegian','Portuguese_Portugues','Spanish','Swedish_Svenska']
    langCorpus=[]
    for x in lang:
        langCorpus.append(udhr.raw(x+'-Latin1'))
    ngramCorpus=multiNgram(langCorpus,n)
    test='Du bist nicht gut'
    vec=testFreqVec(ngramCorpus,test,n)
    ind=np.argmax(multiCos(vec))
    print(lang[ind])
    #lang[np.argmax(multiCos(testFreqVec(multiNgram(langCorpus,n),test,n)))]
    
if(__name__=="__main__"):
    main()